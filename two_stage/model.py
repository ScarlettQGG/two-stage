"""Models for the two-stage pipeline.

Stage 1 — ``Stage1``: a mask-aware multimodal autoencoder that
fuses the per-modality latents into one joint co-embedding (the static map).
Stage 2 — ``NeighborhoodAdapter``: a neighbourhood-aware delta adapter that
learns, per perturbation, how that map remodels (leave-one-out neighbour
prediction → coherence-weighted, drift-removed movement on the unit sphere).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # only for type hints — avoids a model<->cache import cycle
    from .stage1_bridge import Stage1Cache


# ===================== Stage 1: multimodal model =====================

class _MLP(nn.Module):
    """Compact 2-layer MLP with optional dropout, used for encoders/decoders/fusion."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0,
                 last_activation: Optional[str] = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )
        self.last_activation = last_activation

    def forward(self, x):
        h = self.net(x)
        if self.last_activation == "tanh":
            h = torch.tanh(h)
        return h


class Stage1(nn.Module):
    """ joint embedding for >=2 modalities with missing-data masking.

    Parameters
    ----------
    modality_dims : dict {modality_name: input_dim}
        Input feature dimensionality per modality.
    latent_dim_per_modality : int
        Dim of per-modality latents h_m. Default 64.
    joint_dim : int
        Dim of the joint embedding z. Default 256.
    hidden_dim : int
        Hidden width inside encoders/decoders/fusion. Default 256.
    dropout : float
        Dropout on encoder MLPs. Default 0.0.
    """

    def __init__(self,
                 modality_dims: Dict[str, int],
                 latent_dim_per_modality: int = 64,
                 joint_dim: int = 256,
                 hidden_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.modality_names: List[str] = list(modality_dims.keys())
        self.modality_dims = dict(modality_dims)
        self.latent_dim = int(latent_dim_per_modality)
        self.joint_dim = int(joint_dim)

        # Per-modality encoders: input -> h_m
        self.encoders = nn.ModuleDict({
            m: _MLP(d, hidden_dim, self.latent_dim, dropout=dropout)
            for m, d in modality_dims.items()
        })
        # Fusion: [h_1 | ... | h_M | mask_1 | ... | mask_M] -> z
        fusion_in = self.latent_dim * len(self.modality_names) + len(self.modality_names)
        self.fusion = _MLP(fusion_in, hidden_dim, self.joint_dim,
                           dropout=dropout, last_activation=None)
        # Per-modality decoders: z -> x_hat_m
        self.decoders = nn.ModuleDict({
            m: _MLP(self.joint_dim, hidden_dim, d, dropout=0.0)
            for m, d in modality_dims.items()
        })
        # Kendall homoscedastic uncertainty (one per modality, per loss term).
        # Initialised at 0 -> exp(-0) = 1 weight; will be learned during training.
        n = len(self.modality_names)
        self.log_sigma_recon = nn.Parameter(torch.zeros(n))
        self.log_sigma_struct = nn.Parameter(torch.zeros(n))

    def encode(self, inputs: Dict[str, torch.Tensor],
               masks: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode and fuse.

        inputs : dict {m: (B, d_m)}. For missing entries the tensor must be
                 present (zero-filled is fine).
        masks  : dict {m: (B,) float/bool}, 1 if present, 0 if missing.

        Returns
        -------
        z      : (B, joint_dim) — the fused joint embedding.
        h_dict : dict {m: (B, latent_dim)} — per-modality latents (mask-gated).
        """
        h_blocks: List[torch.Tensor] = []
        mask_blocks: List[torch.Tensor] = []
        h_dict: Dict[str, torch.Tensor] = {}
        for m in self.modality_names:
            x = inputs[m]
            mk = masks[m].float().unsqueeze(-1)        # (B,1)
            h = self.encoders[m](x) * mk                # mask-gated
            h_dict[m] = h
            h_blocks.append(h)
            mask_blocks.append(mk)
        h_cat = torch.cat(h_blocks + mask_blocks, dim=-1)  # (B, M*d + M)
        z = self.fusion(h_cat)
        return z, h_dict

    def decode(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """Decode joint z back to one modality's input space."""
        return self.decoders[modality](z)

    def forward(self, inputs, masks):
        return self.encode(inputs, masks)


def make_model(modality_dims: Dict[str, int],
               latent_dim_per_modality: int = 64,
               joint_dim: int = 256,
               hidden_dim: int = 256,
               dropout: float = 0.0) -> Stage1:
    return Stage1(modality_dims=modality_dims,
                      latent_dim_per_modality=latent_dim_per_modality,
                      joint_dim=joint_dim,
                      hidden_dim=hidden_dim,
                      dropout=dropout)

# ===================== Stage 2: neighbourhood adapter =====================


# ───────────────────────────────────────────────────────────────────────────
# Tiny building blocks
# ───────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1, dropout: float = 0.0
         ) -> nn.Sequential:
    """Compact MLP. The Dropout layer is ALWAYS included (with p=0 if asked)
    so the nn.Sequential index layout is stable across dropout values — this
    matters for state_dict reload (an adapter trained with dropout=0.1 must
    be reloadable into one constructed with dropout=0.1, otherwise indices
    drift). PyTorch's nn.Dropout(0.0) is a cheap no-op."""
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(max(0, n_hidden - 1)):
        layers.append(nn.Dropout(dropout))                     # always present
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class ResidualMLP(nn.Module):
    """δ_raw_i → R^d_z with zero-init output layer so it starts as no-op."""
    def __init__(self, d_in: int, hidden: int, d_z: int, n_hidden: int = 2):
        super().__init__()
        self.body = _mlp(d_in, hidden, hidden, n_hidden=n_hidden)
        self.gelu = nn.GELU()
        self.out  = nn.Linear(hidden, d_z)
        nn.init.zeros_(self.out.weight)            # KEY: residual starts at 0
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.gelu(self.body(x)))


# ───────────────────────────────────────────────────────────────────────────
# Sub-modules
# ───────────────────────────────────────────────────────────────────────────

class SelfContext(nn.Module):
    """Build per-protein context vector from z + per-modality h_m + cluster + cond."""
    def __init__(self, d_z: int, d_h: int, n_modalities: int,
                 K: int, n_conds: int, d_clust: int, d_cond: int,
                 hidden: int, d_ctx: int, dropout: float = 0.0):
        super().__init__()
        self.cluster_emb = nn.Embedding(K, d_clust)
        self.cond_emb    = nn.Embedding(n_conds, d_cond)
        in_dim = d_z + n_modalities * d_h + d_clust + d_cond
        self.net = _mlp(in_dim, hidden, d_ctx, n_hidden=2, dropout=dropout)

    def forward(self, z: torch.Tensor, h_per_mod: List[torch.Tensor],
                cluster_id: torch.Tensor, cond_id: torch.Tensor) -> torch.Tensor:
        ce = self.cluster_emb(cluster_id)
        de = self.cond_emb(cond_id)
        x = torch.cat([z] + h_per_mod + [ce, de], dim=-1)
        return self.net(x)


class DeltaEncoder(nn.Module):
    """Encode a single protein's (δ_raw, h_*) into the per-neighbour token e_j ∈ R^d_e."""
    def __init__(self, d_epic: int, d_h: int, n_modalities: int,
                 hidden: int, d_e: int, dropout: float = 0.0):
        super().__init__()
        in_dim = d_epic + n_modalities * d_h
        self.net = _mlp(in_dim, hidden, d_e, n_hidden=2, dropout=dropout)

    def forward(self, delta_raw: torch.Tensor, h_per_mod: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([delta_raw] + h_per_mod, dim=-1)
        return self.net(x)


class NeighbourAttention(nn.Module):
    """Single-head attention from ctx (query) over neighbour tokens (k/v),
    modulated multiplicatively by the cache's edge weights."""
    def __init__(self, d_ctx: int, d_e: int, hidden: int):
        super().__init__()
        self.q = nn.Linear(d_ctx, hidden)
        self.k = nn.Linear(d_e,   hidden)
        self.v = nn.Linear(d_e,   hidden)
        self.scale = 1.0 / math.sqrt(hidden)

    def forward(self, ctx: torch.Tensor, E: torch.Tensor, edge_w: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ctx     : (B, d_ctx)
        E       : (B, k, d_e)              — neighbour tokens
        edge_w  : (B, k)                   — non-negative, row-normalised
        Returns:
          agg   : (B, hidden)              — weighted neighbour aggregation
          alpha : (B, k)                   — final mixing weights (for diagnostics)
        """
        Q = self.q(ctx).unsqueeze(1)                     # (B,1,h)
        K = self.k(E)                                     # (B,k,h)
        V = self.v(E)                                     # (B,k,h)
        s = (Q * K).sum(-1) * self.scale                  # (B,k)  attention logits
        a = F.softmax(s, dim=-1)
        # Multiplicative blend with graph weights (downweights low-conf edges
        # even if attention loved them).
        alpha = a * (edge_w + 1e-9)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        agg = (alpha.unsqueeze(-1) * V).sum(dim=1)        # (B, h)
        return agg, alpha


class DeltaHead(nn.Module):
    """[ctx, agg] → (δ̂, log σ²_pred). σ² is a scalar per protein (isotropic noise)."""
    def __init__(self, d_ctx: int, d_agg: int, hidden: int, d_z: int,
                 dropout: float = 0.0, log_sigma2_init: float = 0.0):
        super().__init__()
        self.body = _mlp(d_ctx + d_agg, hidden, hidden, n_hidden=2, dropout=dropout)
        self.gelu = nn.GELU()
        self.delta_head      = nn.Linear(hidden, d_z)
        self.log_sigma2_head = nn.Linear(hidden, 1)
        # Initialise the log-σ² head to a sensible value so training is stable
        nn.init.zeros_(self.log_sigma2_head.weight)
        with torch.no_grad():
            self.log_sigma2_head.bias.fill_(log_sigma2_init)

    def forward(self, ctx: torch.Tensor, agg: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.gelu(self.body(torch.cat([ctx, agg], dim=-1)))
        delta = self.delta_head(h)
        log_s2 = self.log_sigma2_head(h).squeeze(-1)
        # clamp for numerical stability
        log_s2 = log_s2.clamp(-6.0, 6.0)
        return delta, log_s2


# ───────────────────────────────────────────────────────────────────────────
# Top-level adapter
# ───────────────────────────────────────────────────────────────────────────

class NeighborhoodAdapter(nn.Module):
    """Stage 2 v3.

    Parameters
    ----------
    cache : Stage1Cache       — must have .attach_modules() called before use
    cond_names : list[str]    — e.g. ["cisplatin", "vorinostat"]
    d_e : int                 — neighbour token width (default 128)
    d_ctx : int               — self-context width (default 128)
    d_attn : int              — attention key/value width (default 128)
    hidden : int              — hidden width inside MLPs (default 256)
    dropout : float           — dropout in MLPs (default 0.1)
    sigma2_raw_floor : float  — minimum σ²_raw to avoid division blow-up
    sigma2_raw_scale : float  — multiplier on cache.sigma2_epic when used as σ²_raw
    """
    def __init__(self, cache: Stage1Cache,
                 cond_names: List[str],
                 d_e: int = 128, d_ctx: int = 128, d_attn: int = 128,
                 d_clust: int = 16, d_cond: int = 8,
                 hidden: int = 256, dropout: float = 0.1,
                 sigma2_raw_floor: float = 0.05,
                 sigma2_raw_scale: float = 1.0,
                 sigma2_pred_init: Optional[float] = None,
                 coherence_gate: bool = False,
                 coherence_gate_gamma: float = 1.0,
                 spherical: bool = False,
                 factorized: bool = False,
                 drift_remove: bool = False,
                 unified: bool = False):
        super().__init__()
        if cache.E_EPIC is None:
            raise RuntimeError("Stage1Cache.attach_modules() must be called before "
                               "constructing NeighborhoodAdapter — adapter needs "
                               "access to the frozen E_EPIC/D_* modules.")
        self.cache = cache
        self.cond_names = list(cond_names)
        self.cond_to_id = {c: i for i, c in enumerate(cond_names)}
        self.sigma2_raw_floor = float(sigma2_raw_floor)
        self.sigma2_raw_scale = float(sigma2_raw_scale)
        # Coherence gate: scale a protein's movement by the agreement between its
        # observed differential (δ_raw_proj) and the neighbourhood consensus (δ̂).
        # Isolated/noisy differentials (low coherence) collapse toward 0° (stable);
        # complex-wide coherent remodelling (high coherence) is kept. This is the
        # biology: a real dissociation/translocation moves interacting proteins
        # together, whereas noise moves single proteins independently.
        self.coherence_gate = bool(coherence_gate)
        self.coherence_gate_gamma = float(coherence_gate_gamma)
        # Spherical mode: renormalize z_treat onto the unit hypersphere (the
        # co-embedding is L2-normalized) so the movement is purely angular, and
        # use the cosine EPIC anchor (see compose_loss). Off by default → the
        # original z_treat = z + δ (Euclidean) behaviour is preserved.
        self.spherical = bool(spherical)
        # Factorize the movement into direction (from the combiner) × learned scalar
        # magnitude. Requires spherical (movement is tangential on the sphere).
        self.factorized = bool(factorized)
        # Remove the global treatment drift (mean tangential movement over the batch)
        # before computing direction / coherence / magnitude. ~50-65% of every
        # protein's movement is a shared drift that makes everything look "coherent"
        # and prevents a stable population. Centering isolates the complex-SPECIFIC
        # coordinated movement so the coherence denoiser works as intended.
        self.drift_remove = bool(drift_remove)
        # Unified movement: a SINGLE coherence-weighted, drift-removed vector where
        # magnitude AND direction are co-derived from neighbour agreement (no separate
        # magnitude head, no Bayesian combiner). δ_final = max(0, cos(δ_obs, δ̂))·δ_obs
        # (tangential, drift-removed): coherent complex-wide movement is kept, isolated
        # noise (disagrees with neighbours) → 0 in both magnitude and direction.
        self.unified = bool(unified)

        # Use the same modality order everywhere
        self.modality_order = list(cache.modality_dims.keys())
        n_mod = len(self.modality_order)
        d_z   = cache.d_z
        d_h   = cache.d_h
        K     = cache.K
        d_ep  = cache.modality_dims[cache.epic_name]

        # --- σ²_pred head initialization ---
        # If the head starts at σ²_pred=1.0 (log bias = 0) and σ²_raw is large
        # (e.g. mean σ²_EPIC × scale ≈ 4-5), the Bayesian combination starts
        # with w_raw = 0.18 / w_pred = 0.82, AND the Kendall (1/σ²·MSE + log σ²)
        # loss landscape favours driving σ²_pred down — the head collapses to
        # the floor across all proteins and stops discriminating.
        # Fix: initialize the head's bias near σ²_raw_mean so it starts
        # *balanced* (w_raw ≈ w_pred ≈ 0.5) and has room to move both ways.
        if sigma2_pred_init is None:
            sigma2_raw_mean = float(cache.sigma2_epic.float().mean().item()
                                     * self.sigma2_raw_scale)
            sigma2_pred_init = max(sigma2_raw_mean, float(sigma2_raw_floor))
        log_sigma2_init = math.log(max(float(sigma2_pred_init), 1e-6))
        print(f"[adapter] σ²_pred head init: σ² ≈ {sigma2_pred_init:.3f}  "
              f"(log_bias = {log_sigma2_init:.3f}); σ²_raw_mean ≈ "
              f"{float(cache.sigma2_epic.float().mean().item() * self.sigma2_raw_scale):.3f}")

        # --- sub-modules ---
        self.self_ctx     = SelfContext(d_z=d_z, d_h=d_h, n_modalities=n_mod,
                                        K=K, n_conds=len(cond_names),
                                        d_clust=d_clust, d_cond=d_cond,
                                        hidden=hidden, d_ctx=d_ctx, dropout=dropout)
        self.delta_enc    = DeltaEncoder(d_epic=d_ep, d_h=d_h, n_modalities=n_mod,
                                         hidden=hidden, d_e=d_e, dropout=dropout)
        self.attn         = NeighbourAttention(d_ctx=d_ctx, d_e=d_e, hidden=d_attn)
        self.head         = DeltaHead(d_ctx=d_ctx, d_agg=d_attn,
                                      hidden=hidden, d_z=d_z, dropout=dropout,
                                      log_sigma2_init=log_sigma2_init)
        # Factorized movement δ = m(p)·u(p): a learned non-negative scalar magnitude
        # head. Reads ONLY the neighbour aggregation `agg` (NOT ctx/cond_id/identity)
        # so it must predict magnitude from the coordinated neighbourhood context and
        # cannot shortcut to a per-condition constant ("which drug → high m"). With
        # cond_id available the head learned a flat per-condition offset (perfect
        # null detection but no within-condition discrimination / no stable proteins).
        # L1-sparsified so most proteins → 0 (stable) and only supported movers keep m.
        self.mag_head = nn.Sequential(
            nn.Linear(d_attn, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.residual_proj = ResidualMLP(d_in=d_ep, hidden=hidden, d_z=d_z, n_hidden=2)
        # Frozen-baseline → z-space projector. Small init so the residual MLP
        # carries most of the learning early on; the baseline contributes the
        # frozen-EPIC-encoder direction as a stable anchor.
        self.baseline_to_z = nn.Linear(d_h, d_z, bias=False)
        nn.init.normal_(self.baseline_to_z.weight, std=1e-3)

    # ---------- helpers ----------
    def _gather_h(self, idx: torch.Tensor) -> List[torch.Tensor]:
        """Per-modality h vectors for the requested protein indices, in fixed order."""
        return [self.cache.h_per_modality[m][idx].to(idx.device) for m in self.modality_order]

    def _gather_neighbour_h(self, n_idx: torch.Tensor) -> List[torch.Tensor]:
        """n_idx is (B, k). Returns list[m] of (B, k, d_h)."""
        return [self.cache.h_per_modality[m][n_idx].to(n_idx.device) for m in self.modality_order]

    # ---------- forward ----------
    def forward(self,
                idx: torch.Tensor,
                epic_ctrl: torch.Tensor,
                epic_treat: torch.Tensor,
                cond_id: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        """
        idx        : (B,)    protein indices
        epic_ctrl  : (B, d_epic)
        epic_treat : (B, d_epic)
        cond_id    : (B,)    int condition ids (0..len(cond_names)-1)
        """
        device = idx.device
        cache = self.cache

        # ---- per-protein static features from cache ----
        z      = cache.z[idx].to(device)
        h_self = self._gather_h(idx)
        cl     = cache.cluster_id[idx].to(device)

        # ---- self context (eq a) ----
        ctx = self.self_ctx(z, h_self, cl, cond_id)

        # ---- neighbour gather (eq b) ----
        n_idx = cache.knn_idx[idx].to(device)             # (B, k)
        n_w   = cache.knn_w[idx].to(device)               # (B, k)
        B, k  = n_idx.shape

        # Per-neighbour features. We need the neighbour's raw delta:
        # we don't pass the full delta matrix as input — instead we read from
        # the *batched* δ_raw cached by the training step. The training loop
        # owns this cache (see `forward_with_neighbour_delta` below).
        raise RuntimeError(
            "Use forward_with_neighbour_delta(...) — the bare forward needs "
            "neighbour deltas which are provided by the training loop, not the cache.")

    def forward_with_neighbour_delta(self,
                                     idx: torch.Tensor,
                                     epic_ctrl_i: torch.Tensor,
                                     epic_treat_i: torch.Tensor,
                                     delta_raw_i: torch.Tensor,
                                     delta_raw_neigh: torch.Tensor,
                                     cond_id: torch.Tensor,
                                     ) -> Dict[str, torch.Tensor]:
        """
        Forward pass when the training/inference loop has already prepared the
        condition-specific raw deltas for both centre nodes and their neighbours.

        idx              : (B,)             centre indices
        epic_ctrl_i      : (B, d_epic)
        epic_treat_i     : (B, d_epic)
        delta_raw_i      : (B, d_epic)      = epic_treat_i − epic_ctrl_i  (precomputed)
        delta_raw_neigh  : (B, k, d_epic)   = neighbours' deltas, already gathered
        cond_id          : (B,)
        """
        device = idx.device
        cache  = self.cache

        # --- per-protein static features ---
        z      = cache.z[idx].to(device)
        h_self = self._gather_h(idx)
        cl     = cache.cluster_id[idx].to(device)

        # --- (a) self context ---
        ctx = self.self_ctx(z, h_self, cl, cond_id)

        # --- (b) neighbour encoding (LEAVE-ONE-OUT) ---
        n_idx = cache.knn_idx[idx].to(device)               # (B, k)
        n_w   = cache.knn_w[idx].to(device)
        B, k, d_ep = delta_raw_neigh.shape
        # gather neighbour h_m
        h_neigh = self._gather_neighbour_h(n_idx)            # list of (B, k, d_h)
        # encode each (j) row → (B, k, d_e)
        # reshape to (B*k, ...) for the MLP, then back
        delta_flat = delta_raw_neigh.reshape(B * k, d_ep)
        h_flat = [h.reshape(B * k, -1) for h in h_neigh]
        e_flat = self.delta_enc(delta_flat, h_flat)          # (B*k, d_e)
        E = e_flat.reshape(B, k, -1)

        # --- (c) attention with edge-weight modulation ---
        agg, alpha = self.attn(ctx, E, n_w)

        # --- (d) head ---
        delta_hat, log_sigma2_pred = self.head(ctx, agg)
        sigma2_pred = torch.exp(log_sigma2_pred).clamp_min(self.sigma2_raw_floor)

        # --- (e) δ_raw projection: (b) frozen baseline + residual ---
        with torch.no_grad():
            base_treat = cache.E_EPIC(epic_treat_i)
            base_ctrl  = cache.E_EPIC(epic_ctrl_i)
            delta_baseline = base_treat - base_ctrl          # in h-space, NOT z-space
        # The frozen baseline lives in per-modality h-space (d_h), not z-space (d_z).
        # We map it to z-space via a fixed linear cast aligned with the cache geometry:
        # the simplest faithful choice is just zero-pad / project — we let the residual
        # MLP carry the burden of getting into z-space, and add the *direction* of the
        # frozen baseline as a regulariser. Concretely: project baseline to z via the
        # ResidualMLP's first layer reading the baseline as δ_raw — this couples them.
        delta_residual = self.residual_proj(delta_raw_i)     # (B, d_z), zero-init
        delta_baseline_z = self.baseline_to_z(delta_baseline)  # (B, d_z), small init
        delta_raw_proj = delta_baseline_z + delta_residual

        # --- (f) Bayesian combination ---
        sigma2_raw = (self.sigma2_raw_scale
                      * cache.sigma2_epic[idx].to(device)
                      ).clamp_min(self.sigma2_raw_floor)
        # posterior mean: δ_final = (σ²_pred · δ_raw_proj + σ²_raw · δ̂) / (σ²_raw + σ²_pred)
        w_raw  = sigma2_pred / (sigma2_raw + sigma2_pred)          # weight on observation
        w_pred = sigma2_raw  / (sigma2_raw + sigma2_pred)          # weight on neighbour prediction
        w_raw  = w_raw.unsqueeze(-1)
        w_pred = w_pred.unsqueeze(-1)
        delta_comb = w_raw * delta_raw_proj + w_pred * delta_hat

        # neighbourhood coherence (needed by the factorized magnitude target/gate)
        coherence = F.cosine_similarity(delta_raw_proj, delta_hat, dim=-1)  # (B,)

        learned_magnitude = None
        m_raw_target = None

        # --- UNIFIED movement: single coherence-weighted, drift-removed vector -----
        # magnitude and direction co-derived from one operation; supersedes the
        # combiner / factorized magnitude / gate when on.
        if self.unified:
            zhat = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            def _tang(v):
                return v - (v * zhat).sum(-1, keepdim=True) * zhat
            raw_t = _tang(delta_raw_proj)                     # tangential (drift RETAINED)
            hat_t = _tang(delta_hat)
            if self.drift_remove:
                raw_c = raw_t - raw_t.mean(0, keepdim=True)   # drift REMOVED (complex-specific)
                hat_c = hat_t - hat_t.mean(0, keepdim=True)
            else:
                raw_c, hat_c = raw_t, hat_t
            # coherence (signal-vs-noise gate) from the drift-removed (complex-specific)
            # deltas so the global drift doesn't make everything look coherent.
            coh = F.cosine_similarity(raw_c, hat_c, dim=-1)
            w = coh.clamp_min(0.0)                             # gate ∈ [0,1]
            coherence = coh
            # MOVEMENT (z_treat → direction-modules): coherence-gated, but on the
            # drift-RETAINED direction so the strong complex modules survive.
            delta_comb = w.unsqueeze(-1) * raw_t
            # MAGNITUDE (per-protein differential score → stability/ranking): on the
            # drift-REMOVED delta so it's sparse + noise-decoupled (stable population).
            learned_magnitude = w * raw_c.norm(dim=-1)
            z_treat = z + delta_comb
            if self.spherical:
                z_treat = z_treat / (z_treat.norm(dim=-1, keepdim=True) + 1e-8)
            delta_final = z_treat - z
            return {
                "z_treat": z_treat, "delta_final": delta_final, "delta_comb": delta_comb,
                "delta_hat": delta_hat, "delta_raw_proj": delta_raw_proj,
                "coherence": coherence, "learned_magnitude": learned_magnitude,
                "m_raw_target": m_raw_target, "delta_baseline_z": delta_baseline_z,
                "delta_residual": delta_residual, "log_sigma2_pred": log_sigma2_pred,
                "sigma2_pred": sigma2_pred, "sigma2_raw": sigma2_raw, "alpha": alpha,
            }

        # --- factorized δ = m(p)·u(p) ------------------------------------
        # Direction u from the combiner (validated pathway-coherent), magnitude m
        # from a learned, L1-sparsified head. The supervision target is the
        # COORDINATED magnitude = observed tangential magnitude × max(0, coherence)
        # — so isolated/noisy movement (low neighbour-agreement) drives m→0 while
        # coherent complex-wide movement keeps magnitude. Separates "how much"
        # (denoised scalar) from "which way" (the module direction).
        learned_magnitude = None
        m_raw_target = None
        if self.factorized:
            zhat = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            def _tangential(v):
                return v - (v * zhat).sum(-1, keepdim=True) * zhat
            t_comb = _tangential(delta_comb)
            t_raw  = _tangential(delta_raw_proj)
            t_hat  = _tangential(delta_hat)
            if self.drift_remove:
                # subtract the global drift (batch-mean tangential movement) so only
                # complex-specific coordinated movement remains.
                t_comb = t_comb - t_comb.mean(0, keepdim=True)
                t_raw  = t_raw  - t_raw.mean(0, keepdim=True)
                t_hat  = t_hat  - t_hat.mean(0, keepdim=True)
                coherence = F.cosine_similarity(t_raw, t_hat, dim=-1)  # drift-removed
            u = t_comb / (t_comb.norm(dim=-1, keepdim=True) + 1e-8)   # unit tangent direction
            m_raw_target = (t_raw.norm(dim=-1)
                            * coherence.clamp_min(0.0))               # COORDINATED magnitude
            m = torch.nn.functional.softplus(
                self.mag_head(agg)).squeeze(-1)  # (B,) ≥ 0, neighbourhood-derived
            learned_magnitude = m
            delta_comb = m.unsqueeze(-1) * u

        # --- coherence gate ----------------------------------------------
        # coherence = cos(observed δ_raw_proj, neighbour consensus δ̂) ∈ [-1, 1].
        # Gate ∈ [0, 1] suppresses movement where the observation disagrees with
        # the neighbourhood (isolated noise) and keeps it where they agree
        # (coherent complex-wide remodelling) → a stable majority + remodelled tail.
        if self.coherence_gate:
            gate = coherence.clamp_min(0.0)
            if self.coherence_gate_gamma != 1.0:
                gate = gate.pow(self.coherence_gate_gamma)
            delta_comb = gate.unsqueeze(-1) * delta_comb

        # --- treated z ON THE SPHERE -------------------------------------
        # The co-embedding is L2-normalized (lives on a unit hypersphere), so the
        # ONLY biologically meaningful movement is angular (it changes a protein's
        # cosine neighbourhood = complex (dis)association / translocation). The
        # radial component changes ‖z‖ but not the neighbourhood — a meaningless
        # degree of freedom that lets an adapter inflate magnitude without biology.
        # Renormalising removes it: delta_final is then the pure tangential (on-
        # sphere) movement.
        z_treat = z + delta_comb
        if self.spherical:
            z_treat = z_treat / (z_treat.norm(dim=-1, keepdim=True) + 1e-8)
        delta_final = z_treat - z

        return {
            "z_treat":         z_treat,
            "delta_final":     delta_final,
            "delta_comb":      delta_comb,
            "delta_hat":       delta_hat,
            "delta_raw_proj":  delta_raw_proj,
            "coherence":       coherence,
            "learned_magnitude": learned_magnitude,
            "m_raw_target":      m_raw_target,
            "delta_baseline_z": delta_baseline_z,
            "delta_residual":   delta_residual,
            "log_sigma2_pred": log_sigma2_pred,
            "sigma2_pred":     sigma2_pred,
            "sigma2_raw":      sigma2_raw,
            "alpha":           alpha,
        }
