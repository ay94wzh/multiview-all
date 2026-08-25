"""ACT action head — Action Chunking with Transformers (Zhao et al. 2023, CVAE).

Adapted from ManiSkill's reference implementation
(~ /repos/ManiSkill/examples/baselines/act/act/detr/{detr_vae,transformer}.py),
as a drop-in replacement for the baseline arch's DiffusionActionHead:
``compute_loss(cond, actions) -> scalar`` / ``sample(cond, num_steps=None) ->
(B, horizon, action_dim)``.

Flat-cond variant: the head reconstructs its token sequence from the baseline's
global cond vector (obs tokens | proprio embed), so the policy surface is
unchanged. Token flow:

  CVAE encoder (training only): [CLS; proprio; horizon action tokens] -> CLS
      output -> Linear -> mu/logvar (latent_dim*2) -> reparametrized z.
  Decoder: memory = [z; proprio; obs tokens] (learned position per slot),
      horizon learned query embeddings decode a (B, horizon, action_dim) chunk
      by direct linear regression (query i = chunk row i).
  Loss: L1(pred, actions).mean() + kl_weight * KL(mu, logvar).
  Sampling: deterministic z = 0 (the prior mean) — same as the reference.

Deliberate deviations from the reference, all documented:
  * memory positions: a learned table over the flat slots (the reference uses
    learned positions for its prepended tokens and 2D sine for image tokens;
    our ResNet frame tokens carry no 2D structure).
  * decoder output: the FINAL layer's output (the reference's ``hs[0]`` after
    its transpose/stack indexing actually selects the FIRST layer's output —
    an indexing quirk of detr_vae.py, not an intended design).
  * deterministic z=0 decoding (the reference has no stochastic flag either).
  * layer positions are added to q/k only (DETR ``with_pos_embed``); torch's
    stock TransformerDecoder pre-adds positions into the value stream and
    defaults to pre-norm, so we use small handwritten post-norm layers.
  * no padding/causal masks anywhere (learned queries, no padding).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ActionHeadConfig


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, var) || N(0, I)): sum over latent dim, mean over
    batch (matches the reference's kl_divergence)."""
    klds = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(dim=-1).mean()


def _sinusoid_table(n_pos: int, d: int) -> torch.Tensor:
    """(1, n_pos, d) sinusoidal position table (reference
    get_sinusoid_encoding_table: even dims sin, odd dims cos)."""
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)  # (n, 1)
    i = torch.arange(d, dtype=torch.float32).unsqueeze(0)        # (1, d)
    angle = pos / (10000.0 ** (i / d))
    table = torch.zeros(n_pos, d)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.unsqueeze(0)


class _EncoderLayer(nn.Module):
    """DETR post-norm self-attention layer; positions are added to q/k only
    (the value stream stays position-free)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # pos: (1, n, d) — broadcasts over the batch
        h = self.self_attn(src + pos, src + pos, src)[0]
        src = self.norm1(src + self.dropout1(h))
        h = self.ff(src)
        return self.norm2(src + self.dropout2(h))


class _DecoderLayer(nn.Module):
    """DETR post-norm decoder layer: self-attention over queries, then
    cross-attention from queries to the (frozen) memory."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor,
        query_pos: torch.Tensor, mem_pos: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_attn(tgt + query_pos, tgt + query_pos, tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(h))
        h = self.cross_attn(tgt + query_pos, memory + mem_pos, memory)[0]
        tgt = self.norm2(tgt + self.dropout2(h))
        h = self.ff(tgt)
        return self.norm3(tgt + self.dropout3(h))


class ACTActionHead(nn.Module):
    """CVAE action-chunking head over the baseline's flat cond vector.

    cond layout (built by BaselinePolicy._cond): [obs tokens (num_obs_tokens *
    obs_token_dim) | proprio embed (proprio_dim)] — the constructor asserts
    cond_dim matches. Training runs the CVAE encoder (reparametrized z);
    sampling decodes deterministically with z = 0.

    ``num_train_steps`` / ``inference_steps`` exist only for surface parity
    with DiffusionActionHead (scripts/eval.py and rollout_record.py set
    ``model.global_head.inference_steps`` unconditionally). ``sample`` does not
    switch to eval mode — rollouts call ``policy.eval()`` (eval/rollout.py)
    which disables the transformer dropout.
    """

    def __init__(
        self,
        cfg: ActionHeadConfig,
        cond_dim: int,
        obs_token_dim: int,
        num_obs_tokens: int,
        proprio_dim: int,
    ):
        super().__init__()
        if not cfg.use_proprio:
            raise ValueError(
                "head_type='act' requires action_head.use_proprio=true "
                "(the CVAE encoder and decoder need a state token)"
            )
        if cond_dim != num_obs_tokens * obs_token_dim + proprio_dim:
            raise ValueError(
                f"cond_dim {cond_dim} != num_obs_tokens {num_obs_tokens} * "
                f"obs_token_dim {obs_token_dim} + proprio_dim {proprio_dim}"
            )
        act = cfg.act
        d = act.hidden_dim
        self.horizon = cfg.horizon
        self.action_dim = cfg.action_dim
        self.num_train_steps = cfg.num_train_steps  # inert (diffusion parity)
        self.inference_steps = cfg.num_inference_steps  # inert (script parity)
        self.latent_dim = act.latent_dim
        self.kl_weight = act.kl_weight
        self.num_obs_tokens = num_obs_tokens
        self.obs_token_dim = obs_token_dim
        self.proprio_dim = proprio_dim

        # decoder: [z; proprio; obs tokens] memory, horizon learned queries
        self.obs_proj = nn.Linear(obs_token_dim, d)
        self.proprio_proj = nn.Linear(proprio_dim, d)
        self.query_embed = nn.Embedding(self.horizon, d)
        self.mem_pos_embed = nn.Embedding(num_obs_tokens + 2, d)  # z, proprio, obs...
        self.decoder = nn.ModuleList(
            [_DecoderLayer(d, act.nhead, act.dim_feedforward, act.dropout)
             for _ in range(act.dec_layers)]
        )
        self.action_head = nn.Linear(d, cfg.action_dim)  # shared across queries

        # CVAE encoder: [CLS; proprio; actions], sinusoidal positions
        self.cls_embed = nn.Embedding(1, d)
        self.encoder_action_proj = nn.Linear(cfg.action_dim, d)
        self.latent_proj = nn.Linear(d, act.latent_dim * 2)
        self.latent_out_proj = nn.Linear(act.latent_dim, d)
        self.encoder = nn.ModuleList(
            [_EncoderLayer(d, act.nhead, act.dim_feedforward, act.dropout)
             for _ in range(act.enc_layers)]
        )
        self.register_buffer(
            "enc_pos_table", _sinusoid_table(1 + 1 + self.horizon, d)
        )

    def _split_cond(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (obs_tokens (B, N, obs_token_dim), proprio_raw (B, proprio_dim)).
        Matches BaselinePolicy._cond's layout: obs tokens first, proprio last."""
        n = self.num_obs_tokens * self.obs_token_dim
        obs = cond[:, :n].reshape(cond.shape[0], self.num_obs_tokens, self.obs_token_dim)
        return obs, cond[:, -self.proprio_dim:]

    def _decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) z + cond -> (B, horizon, action_dim)."""
        d = self.proprio_proj.out_features
        z = self.latent_out_proj(z)                                  # -> (B, d)
        obs_tokens = self.obs_proj(self._split_cond(cond)[0])        # (B, N, d)
        proprio_tok = self.proprio_proj(self._split_cond(cond)[1])   # (B, d)
        memory = torch.cat(
            [z[:, None], proprio_tok[:, None], obs_tokens], dim=1
        )  # (B, 2 + N, d)
        tgt = torch.zeros(cond.shape[0], self.horizon, d, device=cond.device)
        qpos = self.query_embed.weight[None]        # (1, horizon, d)
        mpos = self.mem_pos_embed.weight[None]      # (1, 2 + N, d)
        for layer in self.decoder:
            tgt = layer(tgt, memory, qpos, mpos)
        return self.action_head(tgt)

    def compute_loss(self, cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """L1 action loss + kl_weight * KL(mu, logvar); scalar."""
        B = actions.shape[0]
        d = self.proprio_proj.out_features
        # CVAE encoder input: [CLS; proprio; action tokens]. Positions are
        # NOT pre-added here — _EncoderLayer adds pos to q/k internally.
        cls = self.cls_embed.weight.expand(B, -1, -1)                 # (B, 1, d)
        proprio_tok = self.proprio_proj(self._split_cond(cond)[1])    # (B, d)
        enc_in = torch.cat(
            [cls, proprio_tok[:, None], self.encoder_action_proj(actions)],
            dim=1,
        )  # (B, 2 + horizon, d)
        pos = self.enc_pos_table
        for layer in self.encoder:
            enc_in = layer(enc_in, pos)
        mu_logvar = self.latent_proj(enc_in[:, 0])                    # CLS output
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = reparametrize(mu, logvar)                                 # (B, latent_dim)
        pred = self._decode(z, cond)                                  # projects to d
        l1 = F.l1_loss(pred, actions)
        return l1 + self.kl_weight * kl_divergence(mu, logvar)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """Deterministic z=0 decoding; num_steps accepted for surface parity
        with DiffusionActionHead.sample and ignored."""
        z = torch.zeros(cond.shape[0], self.latent_dim, device=cond.device)
        return self._decode(z, cond)
