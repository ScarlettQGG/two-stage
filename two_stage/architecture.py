"""cellmaps_coembedding.autoembed_sc.architecture

Neural modules for multimodal co-embedding with optional condition-specific
residual bottleneck adapters (used for treated SEC-MS).

This file is designed to be imported by autoembed_sc.__init__.

Key features
- Multi-modality encoders/decoders
- Optional modality aliasing (so treated SEC-MS replicate names can share weights)
- Optional condition adapters applied to a subset of modalities
"""
# architecture.py
from __future__ import annotations
from typing import Dict, Iterable, Optional, Set, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a frozen nn.Linear.
    Applies: y = W x + enabled*(alpha/r) * B(Ax)
    Only A and B are trainable.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 4,
                 alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear")
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        if self.r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.scale = float(alpha) / float(self.r)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.enabled = True  # <--- NEW

        in_f = self.base.in_features
        out_f = self.base.out_features
        self.A = nn.Parameter(torch.zeros(self.r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, self.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def set_enabled(self, flag: bool = True):
        self.enabled = bool(flag)

    def forward(self, x, return_parts: bool = False):
        base_out = self.base(x)
        if self.enabled:
            lora_out = (self.drop(x) @ self.A.t()) @ self.B.t()
            y = base_out + self.scale * lora_out
        else:
            lora_out = torch.zeros_like(base_out)
            y = base_out

        if return_parts:
            return y, base_out, (self.scale * lora_out)
        return y

class LowRankLatentAligner(nn.Module):
    """
    Latent-space alignment transform:
        z_aligned = z + scale * (z @ A.T) @ B.T

    Think of it as a low-rank update to identity in latent space.
    With strong near-identity regularization, this tends to learn a gentle
    rotation/shear-like correction rather than a wild warp.
    """
    def __init__(self, dim: int, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.r = int(r)
        self.scale = float(alpha) / float(max(1, r))

        self.enabled = True

        self.A = nn.Parameter(torch.zeros(self.r, self.dim))   # [r, D]
        self.B = nn.Parameter(torch.zeros(self.dim, self.r))   # [D, r]
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def set_enabled(self, flag: bool = True):
        self.enabled = bool(flag)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return z
        dz = (z @ self.A.t()) @ self.B.t()  # [B,D]
        return z + self.scale * dz

    def reg_identity(self) -> torch.Tensor:
        # encourages small transform (near identity)
        return (self.A.pow(2).mean() + self.B.pow(2).mean())

class ResidualBottleneckExpert(nn.Module):
    """One expert: bottleneck MLP producing a residual delta."""
    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        # Start near-identity: Δ≈0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MoEAdapter(nn.Module):
    """
    Mixture-of-Experts adapter:
      Δ(x) = Σ_e w_e(x) * Δ_e(x)
    Supports top-k routing for sparse, interpretable protein-specific changes.
    """
    def __init__(
        self,
        dim: int,
        bottleneck: int = 64,
        n_experts: int = 10,
        top_k: int = 2,
        router_hidden: int = 128,
        router_dropout: float = 0.0,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        assert n_experts >= 2
        assert 1 <= top_k <= n_experts

        self.dim = int(dim)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.router_temperature = float(router_temperature)

        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden),
            nn.ReLU(),
            nn.Dropout(router_dropout),
            nn.Linear(router_hidden, self.n_experts),
        )

        self.experts = nn.ModuleList([
            ResidualBottleneckExpert(dim=dim, bottleneck=bottleneck, dropout=router_dropout)
            for _ in range(self.n_experts)
        ])

        # Router init: start near-uniform so early training doesn't hard-commit randomly
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          dx:      [B, D]
          weights: [B, E] (after top-k masking + softmax)
        """
        temp = self.router_temperature if temperature is None else float(temperature)
        logits = self.router(x) / max(temp, 1e-8)  # [B,E]

        if self.top_k < self.n_experts:
            topv, topi = torch.topk(logits, k=self.top_k, dim=-1)
            masked = torch.full_like(logits, -1e9)
            masked.scatter_(1, topi, topv)
            weights = F.softmax(masked, dim=-1)  # [B,E]
        else:
            weights = F.softmax(logits, dim=-1)

        dx_all = torch.stack([ex(x) for ex in self.experts], dim=0)  # [E,B,D]
        w = weights.transpose(0, 1).unsqueeze(-1)                    # [E,B,1]
        dx = (w * dx_all).sum(dim=0)                                 # [B,D]
        return dx, weights


def _get_anchor_for_batch(
    anchors: Union[torch.Tensor, Dict[str, torch.Tensor]],
    m: str,
) -> torch.Tensor:
    """
    anchors can be either:
      - Tensor [B, D] aligned to the current batch rows (same for all modalities), OR
      - Dict[str, Tensor[B,D]] if you ever want modality-specific anchors.
    """
    if isinstance(anchors, dict):
        return anchors[m]
    return anchors

def _wrap_last_linear_in_module(mod: nn.Module, r: int,
                                alpha: int, dropout: float) -> bool:
    """
    Try to find and replace the last nn.Linear in a common container module.
    Returns True if replaced.
    Supports nn.Sequential, or modules with attributes like .net / .mlp (common patterns).
    """
    # Case 1: nn.Sequential
    if isinstance(mod, nn.Sequential):
        # find last Linear index
        last_idx = None
        for i, layer in enumerate(mod):
            if isinstance(layer, nn.Linear):
                last_idx = i
        if last_idx is None:
            return False
        mod[last_idx] = LoRALinear(mod[last_idx], r=r, alpha=alpha, dropout=dropout)
        return True
    else:
        # Case 2: common attribute containers
        for attr in ["net", "mlp", "model", "layers"]:
            if hasattr(mod, attr):
                sub = getattr(mod, attr)
                if isinstance(sub, nn.Module):
                    if _wrap_last_linear_in_module(sub, r=r, alpha=alpha, dropout=dropout):
                        return True
        # Case 3: direct children list scan (fallback): wrap last Linear found among immediate children
        last_name = None
        last_child = None
        for name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                last_name = name
                last_child = child
        if last_child is not None:
            setattr(mod, last_name, 
                    LoRALinear(last_child, r=r, alpha=alpha, dropout=dropout))
            return True
        return False
 
 
class UniEmbedNN(nn.Module):
    """
    Drop-in change: make Stage-2 adapter *anchor-aligned* by default when anchors are provided.

    When (condition is set) and (m is adapted) and anchors is not None:
        a = anchor[p] (static coembedding for the protein)
        z_raw = encoder(x)
        d0 = z_raw - a              # raw displacement from static
        Δd = adapter(d0)            # learned correction of displacement
        d  = d0 + Δd                # final displacement from static
        z_out = a + d               # final treated latent in static coordinate system

    Important:
      - If return_deltas=True, we return deltas[m] = d = (z_out - a)  (the biologically interpretable delta).
      - This makes your delta files match “distance moved away from static embedding”.
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        latent_dim: int = 128,
        hidden_size_1: int = 512,
        hidden_size_2: int = 256,
        dropout: float = 0.0,
        l2_norm: bool = False,
        modality_aliases: Optional[Dict[str, str]] = None,
        adapted_modalities: Optional[Iterable[str]] = None,
        conditions: Optional[Iterable[str]] = None,
        adapter_bottleneck: int = 64,
        # ---- Adapter behavior knobs (Stage-2) ----
        adapter_gate_init: float = 0.05,        # small => adapter starts near off
        adapter_center_input: bool = True,      # remove common-mode before adapter
        adapter_center_output: bool = True,     # remove common-mode after adapter
        # ---- MoE knobs ----
        adapter_type: str = "residual",  # "residual" or "moe"
        moe_n_experts: int = 4,
        moe_top_k: int = 2,
        moe_router_hidden: int = 128,
        moe_router_temperature: float = 1.0,
        latent_aligner_rank: int = 4,
        latent_aligner_alpha: float = 1.0,
    ):
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.hidden_size_1 = int(hidden_size_1)
        self.hidden_size_2 = int(hidden_size_2)
        self.dropout = float(dropout)
        self.l2_norm = bool(l2_norm)

        self.modality_aliases = modality_aliases or {}
        self._base_of: Dict[str, str] = {m: self.modality_aliases.get(m, m) for m in input_dims.keys()}
        self._unique_bases = sorted(set(self._base_of.values()))
        # record bases where LoRA has been enabled
        self._lora_enabled_bases: Set[str] = set()
        self.adapted_modalities: Set[str] = set(adapted_modalities or [])
        self._condition: Optional[str] = None

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        base_input_dims: Dict[str, int] = {}
        for m, base in self._base_of.items():
            base_input_dims.setdefault(base, input_dims[m])

        for base in self._unique_bases:
            in_dim = int(base_input_dims[base])
            self.encoders[base] = nn.Sequential(
                nn.LayerNorm(in_dim),          # <--- add
                nn.Dropout(self.dropout),
                nn.Linear(in_dim, self.hidden_size_1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size_1, self.hidden_size_2),
                nn.ReLU(),
                nn.Linear(self.hidden_size_2, self.latent_dim),
            )
            self.decoders[base] = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, self.hidden_size_2),
                nn.ReLU(),
                nn.Linear(self.hidden_size_2, self.hidden_size_1),
                nn.ReLU(),
                nn.Linear(self.hidden_size_1, in_dim),
            )
        # ---- Latent aligners (Phase-1 global alignment) ----
        # per-base so replicate streams sharing a base share the same aligner
        self.latent_aligners = nn.ModuleDict()
        for base in self._unique_bases:
            self.latent_aligners[base] = LowRankLatentAligner(
                dim=self.latent_dim,
                r=int(latent_aligner_rank),
                alpha=float(latent_aligner_alpha),
            )
            self.latent_aligners[base].set_enabled(False)
            
        self.adapters = nn.ModuleDict()
        cond_list = list(conditions or [])
        self.adapter_type = str(adapter_type).lower()

        for c in cond_list:
            if self.adapter_type == "moe":
                self.adapters[str(c)] = MoEAdapter(
                    dim=self.latent_dim,
                    bottleneck=int(adapter_bottleneck),
                    n_experts=int(moe_n_experts),
                    top_k=int(moe_top_k),
                    router_hidden=int(moe_router_hidden),
                    router_dropout=self.dropout,
                    router_temperature=float(moe_router_temperature),
                )
            else:
                self.adapters[str(c)] = ResidualBottleneckExpert(
                    dim=self.latent_dim,
                    bottleneck=int(adapter_bottleneck),
                    dropout=self.dropout,
                )
        # Adapter gate (one scalar shared across modalities for this condition)
        # Keeps Δ small unless gradients strongly need it.
        self.adapter_gate = nn.Parameter(torch.tensor(float(adapter_gate_init)))

        self.adapter_center_input = bool(adapter_center_input)
        self.adapter_center_output = bool(adapter_center_output)

    def set_condition(self, condition: Optional[str]) -> None:
        self._condition = condition
    
    def disable_all_aligners(self) -> None:
        for _, al in self.latent_aligners.items():
            al.set_enabled(False)
    
    def set_aligner_enabled(self, base: str, enabled: bool = True) -> None:
        if base not in self.latent_aligners:
            raise KeyError(f"Aligner base '{base}' not found in model.latent_aligners")
        self.latent_aligners[base].set_enabled(enabled)

    def set_aligner_trainable(self, base: str, flag: bool = True) -> None:
        if base not in self.latent_aligners:
            return
        for p in self.latent_aligners[base].parameters():
            p.requires_grad_(bool(flag))

    def freeze_all_aligners(self) -> None:
        for _, al in self.latent_aligners.items():
            for p in al.parameters():
                p.requires_grad_(False)

    def enable_only_these_aligners(self, bases: Iterable[str]) -> None:
        keep = set(bases)
        for b, al in self.latent_aligners.items():
            al.set_enabled(b in keep)

    def set_lora_enabled(self, enabled: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.set_enabled(enabled)
    
    def enable_lora_last_linear(self, base: str, r: int = 4, 
                                alpha: int = 16, dropout: float = 0.0) -> None:
        """
        Enable LoRA on ONLY the last nn.Linear inside the encoder for `base`.
        """
        if base not in self.encoders:
            raise KeyError(f"Encoder base '{base}' not found in model.encoders")
        ok = _wrap_last_linear_in_module(self.encoders[base], 
                                         r=int(r), alpha=int(alpha),
                                         dropout=float(dropout))
        if not ok:
            raise RuntimeError(f"Could not locate a Linear layer to wrap with LoRA in encoders['{base}']")
        self._lora_enabled_bases.add(base)
    # -------------------------------
    # Trainability helpers (NEW)
    # -------------------------------
    def freeze_all_encoders(self) -> None:
        """Freeze all encoder params including any LoRA params (A/B)."""
        for base, enc in self.encoders.items():
            for p in enc.parameters():
                p.requires_grad_(False)

        # also freeze LoRA A/B explicitly (in case someone set them trainable)
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.A.requires_grad_(False)
                m.B.requires_grad_(False)

    def set_lora_trainable(self, flag: bool = True) -> None:
        """Toggle trainability of LoRA params (A/B) only."""
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.A.requires_grad_(bool(flag))
                m.B.requires_grad_(bool(flag))

    def set_adapter_trainable(self, condition: str, flag: bool = True) -> None:
        """Toggle trainability of adapter params for a given condition."""
        c = str(condition)
        if c not in self.adapters:
            return
        for p in self.adapters[c].parameters():
            p.requires_grad_(bool(flag))

    def set_gate_trainable(self, flag: bool = True) -> None:
        """Toggle trainability of the scalar adapter gate."""
        self.adapter_gate.requires_grad_(bool(flag))

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_deltas: bool = False,
        return_router: bool = False,
        router_temperature: Optional[float] = None,
        anchors: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[Dict[str, torch.Tensor]] = None,   # <-- ADD THIS
        return_delta_components: bool = False,
        return_pre_latents: bool = False,
):
        """
        If anchors is provided (Stage-2), deltas[m] returned is:
            delta_anchor = z_out - anchor  (shape [B,D])

        If anchors is None (Stage-1 or fallback), deltas[m] returned is:
            dz = z_out - z_raw  (shape [B,D])
        """
        latents: Dict[str, torch.Tensor] = {}
        deltas: Dict[str, torch.Tensor] = {}
        router_w: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        pre_latents: Dict[str, torch.Tensor] = {}
        delta_components: Dict[str, Dict[str, torch.Tensor]] = {}
        
        use_adapter = (self._condition is not None) and (self._condition in self.adapters)
        # If anchor_mask is provided, it must be [B] boolean
        # If anchors provided but mask omitted, assume all rows anchored.
        for m, x in inputs.items():
            base = self._base_of.get(m, m)
            z_raw = self.encoders[base](x)

            # ---- Phase-1 global latent alignment (if enabled) ----
            if base in self.latent_aligners:
                z_raw = self.latent_aligners[base](z_raw)

            pre_latents[m] = z_raw
            if self.l2_norm:
                z_raw = F.normalize(z_raw, p=2, dim=1)

            z_out = z_raw

            if use_adapter and (m in self.adapted_modalities):
                adapter = self.adapters[self._condition]

                if anchors is not None:
                    # ----- Anchor-aligned displacement adaptation -----
                    a = _get_anchor_for_batch(anchors, m)  # [B,D]
                    if self.l2_norm:
                        a = F.normalize(a, p=2, dim=1)
                    d0 = z_raw - a  # [B,D]


                    if anchor_mask is None:
                        am = torch.ones(z_raw.size(0), dtype=torch.bool, device=z_raw.device)
                    else:
                        am = anchor_mask.to(device=z_raw.device, dtype=torch.bool)
                        if am.ndim != 1 or am.shape[0] != z_raw.shape[0]:
                            raise ValueError(f"anchor_mask must be shape [B]; got {tuple(am.shape)} vs B={z_raw.shape[0]}")

                    # NEW: also require that this modality is present for that protein in this batch
                    if input_mask is not None:
                        if m not in input_mask:
                            raise KeyError(f"input_mask provided but missing key '{m}'")
                        pm = input_mask[m].to(device=z_raw.device, dtype=torch.bool)
                        if pm.ndim != 1 or pm.shape[0] != z_raw.shape[0]:
                            raise ValueError(f"input_mask['{m}'] must be shape [B]; got {tuple(pm.shape)} vs B={z_raw.shape[0]}")
                        am = am & pm

                    msk = am.float().unsqueeze(1)  # [B,1]

                    

                    # -------- Center adapter input across proteins (within batch) --------
                    # This removes common-mode (global) drift so adapter can't become a batch corrector.
                    d_in = d0 * msk
                    if self.adapter_center_input:
                        mean_in = (d_in.sum(dim=0, keepdim=True) / (msk.sum() + 1e-8))
                        d_in = (d_in - mean_in) * msk


                    # -------- Apply adapter (MoE or residual bottleneck) --------
                    if isinstance(adapter, MoEAdapter):
                        dd, w = adapter(d_in, temperature=router_temperature)
                        if return_router:
                            router_w[m] = w
                    else:
                        dd = adapter(d_in)

                    # -------- Center adapter output (removes global component) --------
                    if self.adapter_center_output:
                        mean_out = (dd.sum(dim=0, keepdim=True) / (msk.sum() + 1e-8))
                        dd = (dd - mean_out) * msk

                    # -------- Gate the adapter residual --------
                    # Keeps Δ small unless needed; avoids “adapter fixes everything”.
                    dd = torch.tanh(self.adapter_gate) * dd

                    pre_d = d0 * msk              # LoRA/systematic displacement (masked)
                    adp_d = dd * msk              # adapter-only component (masked)
                    d = (d0 + dd) * msk           # total Δ to anchor (masked)

                    z_anchor = a + d
                    z_out = (msk * z_anchor) + ((1.0 - msk) * z_raw)

                    if return_deltas:
                        deltas[m] = d  # total delta to anchor

                    if return_delta_components:
                        delta_components.setdefault("pre", {})[m] = pre_d
                        delta_components.setdefault("adapter", {})[m] = adp_d
                        delta_components.setdefault("total", {})[m] = d

                else:
                    # ----- Fallback: adapt in latent space -----
                    if isinstance(adapter, MoEAdapter):
                        dz, w = adapter(z_raw, temperature=router_temperature)
                        if return_router:
                            router_w[m] = w
                    else:
                        dz = adapter(z_raw)

                    z_out = z_raw + dz
                    if return_deltas:
                        deltas[m] = dz

            latents[m] = z_out

        # Cross-decode every latent into every modality's decoder (same as before)
        out_modalities = list(inputs.keys())
        for in_m, z in latents.items():
            for out_m in out_modalities:
                out_base = self._base_of.get(out_m, out_m)
                outputs[f"{in_m}_{out_m}"] = self.decoders[out_base](z)

        # returns
        if return_pre_latents:
            if return_deltas and return_router and return_delta_components:
                return pre_latents, latents, outputs, deltas, router_w, delta_components
            if return_deltas and return_delta_components:
                return pre_latents, latents, outputs, deltas, delta_components
            if return_deltas and return_router:
                return pre_latents, latents, outputs, deltas, router_w
            if return_deltas:
                return pre_latents, latents, outputs, deltas
            return pre_latents, latents, outputs

        if return_deltas and return_router and return_delta_components:
            return latents, outputs, deltas, router_w, delta_components
        if return_deltas and return_delta_components:
            return latents, outputs, deltas, delta_components
        if return_deltas and return_router:
            return latents, outputs, deltas, router_w
        if return_deltas:
            return latents, outputs, deltas
        return latents, outputs

