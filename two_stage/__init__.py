# __init__.py  (cellmaps_coembedding.autoembed_sc)
"""
cellmaps_coembedding.autoembed_sc

- Stage 1: static co-embedding training (enc/dec only)
- Stage 2: treated SEC-MS adapter training (anchor-aligned)
- Saving treated embeddings and delta-to-anchor rankings

This version is adapted for the UPDATED architecture.py where:
  - If anchors are passed to model(..., anchors=anchors_batch),
    model returns deltas[m] = (z_out - anchor)  == Δ_anchor (biologically interpretable).
"""
from __future__ import annotations
import csv
import os
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Iterator, Union, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from .architecture import UniEmbedNN, LoRALinear



# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _read_tsv_embedding(path: str) -> Dict[str, np.ndarray]:
    emb: Dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty TSV embedding: {path}")
        for row in reader:
            if not row:
                continue
            pid = row[0]
            vec = np.fromstring("\t".join(row[1:]), sep="\t", dtype=float)
            emb[pid] = vec
    if len(emb) == 0:
        raise ValueError(f"No data rows in TSV embedding: {path}")
    return emb


def load_modalities_from_manifest(entries: List[dict]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    modalities_dict: Dict[str, Dict[str, np.ndarray]] = {}
    modality_aliases: Dict[str, str] = {}

    for e in entries:
        name = e["name"]
        mod = e["modality"]
        path = e["path"]
        modalities_dict[name] = _read_tsv_embedding(path)
        modality_aliases[name] = mod

    for n, d in modalities_dict.items():
        dims = {v.shape[0] for v in d.values()}
        if len(dims) != 1:
            raise ValueError(f"Entry '{n}' has inconsistent feature lengths: {sorted(dims)}")
    return modalities_dict, modality_aliases


def load_modalities_from_paths(paths: List[str], names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    if len(paths) != len(names):
        raise ValueError(f"paths and names must have same length. Got {len(paths)} vs {len(names)}")

    modalities_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for p, n in zip(paths, names):
        modalities_dict[n] = _read_tsv_embedding(p)

    for n, d in modalities_dict.items():
        dims = {v.shape[0] for v in d.values()}
        if len(dims) != 1:
            raise ValueError(f"Modality '{n}' has inconsistent feature lengths: {sorted(dims)}")
    return modalities_dict


def write_embedding_dictionary_to_file(path: str, d: Dict[str, np.ndarray], dim: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["protein"] + [str(i) for i in range(dim)])
        for k in sorted(d.keys()):
            vec = d[k]
            writer.writerow([k] + [f"{float(x):.8f}" for x in vec.tolist()])


def _load_anchor_tsv(path: str) -> Dict[str, np.ndarray]:
    return _read_tsv_embedding(path)

def _load_protein_list(path: str) -> Set[str]:
    s: Set[str] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            for p in parts:
                if p:
                    s.add(p)
    return s

# -----------------------------------------------------------------------------
# Dataset + batching
# -----------------------------------------------------------------------------

@dataclass
class TrainingDataWrapper:
    modalities_dict: Dict[str, Dict[str, np.ndarray]]
    modality_names: List[str]
    device: torch.device
    l2_norm: bool
    dropout: float
    latent_dim: int
    hidden_size_1: int
    hidden_size_2: int
    resultsdir: str


class Protein_Dataset(Dataset):
    """
    Union-of-proteins dataset.
    Missing modality returns zeros + mask=0.
    """
    def __init__(self, modalities_dict: Dict[str, Dict[str, np.ndarray]]):
        super().__init__()
        self.modalities_dict = modalities_dict
        self.modality_names = list(modalities_dict.keys())

        proteins = set()
        for m in self.modality_names:
            proteins |= set(modalities_dict[m].keys())
        self.protein_ids = sorted(proteins)

        self.dims: Dict[str, int] = {}
        for m in self.modality_names:
            any_vec = next(iter(modalities_dict[m].values()))
            self.dims[m] = int(any_vec.shape[0])

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int):
        pname = self.protein_ids[idx]
        protein: Dict[str, torch.Tensor] = {}
        mask: Dict[str, float] = {}

        for m in self.modality_names:
            if pname in self.modalities_dict[m]:
                vec = self.modalities_dict[m][pname]
                protein[m] = torch.tensor(vec, dtype=torch.float32)
                mask[m] = 1.0
            else:
                protein[m] = torch.zeros(self.dims[m], dtype=torch.float32)
                mask[m] = 0.0
        return protein, mask, idx


def _collate_protein_batch(batch):
    prot_dicts, mask_dicts, idxs = zip(*batch)
    modalities = prot_dicts[0].keys()
    batch_data = {m: torch.stack([p[m] for p in prot_dicts], dim=0) for m in modalities}
    batch_mask = {m: torch.tensor([md.get(m, 0.0) for md in mask_dicts], dtype=torch.float32) for m in modalities}
    return batch_data, batch_mask, list(idxs)

def compute_base_similarity_projected(
    modalities_dict: Dict[str, Dict[str, np.ndarray]],
    base_of: Dict[str, str],
    proj_dim: int = 128,
    max_proteins: int = 5000,
    min_overlap: int = 50,
    seed: int = 0,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, float]]:
    """
    Compute base similarity using mean cosine over overlapping proteins,
    after projecting each base's input vectors to a shared proj_dim.

    This makes similarity estimation less sensitive to differing input dims.
    """
    rng = np.random.RandomState(seed)

    # base -> list of modality streams
    base_to_streams = defaultdict(list)
    for m in modalities_dict.keys():
        base_to_streams[base_of.get(m, m)].append(m)

    # choose protein universe (cap for speed)
    proteins = set()
    for m, d in modalities_dict.items():
        proteins |= set(d.keys())
    proteins = sorted(proteins)
    if len(proteins) > max_proteins:
        proteins = rng.choice(proteins, size=max_proteins, replace=False).tolist()

    # base -> protein -> base-vector (avg across streams of that base)
    base_to_pvec = defaultdict(dict)
    base_dims = {}

    for base, streams in base_to_streams.items():
        # determine dim
        any_stream = streams[0]
        any_vec = next(iter(modalities_dict[any_stream].values()))
        D = int(any_vec.shape[0])
        base_dims[base] = D

        for p in proteins:
            vecs = []
            for s in streams:
                if p in modalities_dict[s]:
                    vecs.append(modalities_dict[s][p])
            if vecs:
                base_to_pvec[base][p] = np.mean(np.stack(vecs, 0), 0).astype(np.float32)

    bases = sorted(base_to_pvec.keys())

    # fixed random projection per base: R in R^{D x proj_dim}, scaled by 1/sqrt(D)
    proj = {}
    for b in bases:
        D = base_dims[b]
        R = rng.normal(0.0, 1.0, size=(D, proj_dim)).astype(np.float32) / np.sqrt(max(D, 1))
        proj[b] = R

    # helper: project + l2 normalize
    def _proj_norm(b, X):  # X [N,D]
        Y = X @ proj[b]     # [N,proj_dim]
        n = np.linalg.norm(Y, axis=1, keepdims=True)
        return Y / (np.maximum(n, eps))

    sim = {b: {} for b in bases}

    for bi in bases:
        for bj in bases:
            if bi == bj:
                sim[bi][bj] = 1.0
                continue
            overlap = sorted(set(base_to_pvec[bi].keys()) & set(base_to_pvec[bj].keys()))
            if len(overlap) < min_overlap:
                sim[bi][bj] = 0.0
                continue

            Xi = np.stack([base_to_pvec[bi][p] for p in overlap], 0)  # [N,Di]
            Xj = np.stack([base_to_pvec[bj][p] for p in overlap], 0)  # [N,Dj]

            Zi = _proj_norm(bi, Xi)
            Zj = _proj_norm(bj, Xj)

            sim[bi][bj] = float(np.mean(np.sum(Zi * Zj, axis=1)))  # mean cosine over proteins

    return sim

# -----------------------------------------------------------------------------
# Stage 1 aggregation helpers (replicate-to-base)
# -----------------------------------------------------------------------------

def _aggregate_latents_by_base(
    latents: Dict[str, torch.Tensor],
    batch_mask: Dict[str, torch.Tensor],
    base_of: Dict[str, str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    base_to_streams = defaultdict(list)
    for name in latents.keys():
        base_to_streams[base_of.get(name, name)].append(name)

    base_latents = {}
    base_present = {}

    for base, streams in base_to_streams.items():
        z_stack = torch.stack([latents[s] for s in streams], dim=0)               # [R,B,D]
        m_stack = torch.stack([batch_mask[s].bool() for s in streams], dim=0)     # [R,B]

        z_sum = torch.zeros_like(z_stack[0])                                      # [B,D]
        denom = torch.zeros(z_stack.shape[1], device=device)                      # [B]
        for r in range(z_stack.shape[0]):
            mr = m_stack[r]
            z_sum[mr] += z_stack[r][mr]
            denom[mr] += 1.0

        base_latents[base] = z_sum / (denom.unsqueeze(1) + 1e-8)
        base_present[base] = (denom > 0).float()

    return base_latents, base_present


def _aggregate_inputs_by_base(
    batch_data: Dict[str, torch.Tensor],
    batch_mask: Dict[str, torch.Tensor],
    base_of: Dict[str, str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    base_to_streams = defaultdict(list)
    for name in batch_data.keys():
        base_to_streams[base_of.get(name, name)].append(name)

    base_inputs = {}
    base_present = {}

    for base, streams in base_to_streams.items():
        x_stack = torch.stack([batch_data[s] for s in streams], dim=0)            # [R,B,Dx]
        m_stack = torch.stack([batch_mask[s].bool() for s in streams], dim=0)     # [R,B]

        x_sum = torch.zeros_like(x_stack[0])                                      # [B,Dx]
        denom = torch.zeros(x_stack.shape[1], device=device)                      # [B]
        for r in range(x_stack.shape[0]):
            mr = m_stack[r]
            x_sum[mr] += x_stack[r][mr]
            denom[mr] += 1.0

        base_inputs[base] = x_sum / (denom.unsqueeze(1) + 1e-8)
        base_present[base] = (denom > 0).float()

    return base_inputs, base_present


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------

def _replicate_consistency_loss(
    latents: Dict[str, torch.Tensor],
    batch_mask: Dict[str, torch.Tensor],
    replicate_groups: Optional[Dict[str, List[str]]],
    device: torch.device,
) -> torch.Tensor:
    """Stage-1 replicate agreement on latent directions (cosine)."""
    if not replicate_groups:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)
    n_terms = 0

    for _, mods in replicate_groups.items():
        mods = [m for m in mods if m in latents]
        if len(mods) < 2:
            continue

        present = [batch_mask[m].bool() for m in mods]
        present_count = torch.stack(present, dim=0).sum(dim=0)
        valid = present_count >= 2
        if valid.sum() == 0:
            continue

        z_stack = torch.stack([latents[m] for m in mods], dim=0)            # [R,B,D]
        m_stack = torch.stack([batch_mask[m].bool() for m in mods], dim=0)  # [R,B]

        z_sum = torch.zeros_like(z_stack[0])                                # [B,D]
        denom = torch.zeros(z_stack.shape[1], device=device)                # [B]
        for r in range(z_stack.shape[0]):
            mr = m_stack[r]
            z_sum[mr] += z_stack[r][mr]
            denom[mr] += 1
        z_mean = z_sum / (denom.unsqueeze(1) + 1e-8)

        for r in range(z_stack.shape[0]):
            mr = m_stack[r] & valid
            if mr.sum() == 0:
                continue
            total = total + torch.mean(1.0 - F.cosine_similarity(z_stack[r][mr], z_mean[mr], dim=1))
            n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device)
    return total / n_terms


def _replicate_delta_anchor_consistency_loss(
    deltas: Dict[str, torch.Tensor],
    batch_mask: Dict[str, torch.Tensor],
    replicate_groups: Optional[Dict[str, List[str]]],
    device: torch.device,
) -> torch.Tensor:
    """
    Stage-2 replicate agreement on Δ_anchor = (z_out - anchor).
    NOTE: with the updated architecture, model(..., anchors=...) returns deltas[m] = Δ_anchor directly.
    """
    if not replicate_groups:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)
    n_terms = 0

    for _, mods in replicate_groups.items():
        mods = [m for m in mods if (m in deltas and m in batch_mask)]
        if len(mods) < 2:
            continue

        present = [batch_mask[m].bool() for m in mods]
        present_count = torch.stack(present, dim=0).sum(dim=0)
        valid = present_count >= 2
        if valid.sum() == 0:
            continue

        d_stack = torch.stack([deltas[m] for m in mods], dim=0)            # [R,B,D]
        m_stack = torch.stack([batch_mask[m].bool() for m in mods], dim=0) # [R,B]

        d_sum = torch.zeros_like(d_stack[0])                               # [B,D]
        denom = torch.zeros(d_stack.shape[1], device=device)               # [B]
        for r in range(d_stack.shape[0]):
            mr = m_stack[r]
            d_sum[mr] += d_stack[r][mr]
            denom[mr] += 1
        d_mean = d_sum / (denom.unsqueeze(1) + 1e-8)

        for r in range(d_stack.shape[0]):
            mr = m_stack[r] & valid
            if mr.sum() == 0:
                continue
            total = total + torch.mean(1.0 - F.cosine_similarity(d_stack[r][mr], d_mean[mr], dim=1))
            n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=device)
    return total / n_terms


def _cosine_sim_matrix(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Xn = X / (X.norm(dim=1, keepdim=True) + eps)
    Yn = Y / (Y.norm(dim=1, keepdim=True) + eps)
    return Xn @ Yn.t()


def identity_infonce_loss(A: torch.Tensor, Z: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """
    Align identities without forcing equality:
      positives: (Z_i, A_i)
      negatives: (Z_i, A_j)  j!=i
    """
    if A.shape[0] < 2:
        return torch.tensor(0.0, device=A.device)
    logits = _cosine_sim_matrix(Z, A) / float(tau)
    labels = torch.arange(A.shape[0], device=A.device)
    return F.cross_entropy(logits, labels)


def _row_softmax_masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))
    masked_logits = masked_logits - masked_logits.max(dim=1, keepdim=True).values
    exp = torch.exp(masked_logits) * mask.to(logits.dtype)
    denom = exp.sum(dim=1, keepdim=True) + 1e-12
    return exp / denom


def _row_entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p + 1e-12)).sum(dim=1)


def _kl_rowwise(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=1)


def neighborhood_kl_loss_knn_masked(
    A: torch.Tensor,
    Z: torch.Tensor,
    k: int = 20,
    tau: float = 0.2,
    entropy_thresh: float = 2.5,
) -> torch.Tensor:
    """
    kNN-masked KL(P_anchor || P_treated) for stable proteins (low entropy in anchor neighborhood dist).
    """
    device = A.device
    N = A.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=device)

    simA = _cosine_sim_matrix(A, A) / float(tau)
    simZ = _cosine_sim_matrix(Z, Z) / float(tau)

    eye = torch.eye(N, device=device, dtype=torch.bool)
    simA = simA.masked_fill(eye, -1e9)
    simZ = simZ.masked_fill(eye, -1e9)

    k_eff = min(int(k), max(1, N - 1))
    knn_idx = torch.topk(simA, k=k_eff, dim=1).indices
    knn_mask = torch.zeros((N, N), device=device, dtype=torch.bool)
    knn_mask.scatter_(1, knn_idx, True)

    pA = _row_softmax_masked(simA, knn_mask)
    pZ = _row_softmax_masked(simZ, knn_mask)

    entA = _row_entropy(pA)
    stable = entA <= float(entropy_thresh)
    if stable.sum() == 0:
        return torch.tensor(0.0, device=device)

    kl = _kl_rowwise(pA, pZ)
    return kl[stable].mean()


def _huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    absx = torch.abs(x)
    d = torch.tensor(delta, device=x.device, dtype=x.dtype)
    quad = torch.minimum(absx, d)
    lin = absx - quad
    return 0.5 * quad * quad + d * lin


def delta_sparsity_huber(deltas: Dict[str, torch.Tensor], delta: float = 0.05) -> torch.Tensor:
    """
    Elementwise huber: encourages many coordinates to 0 but allows a few large moves.
    """
    if not deltas:
        return torch.tensor(0.0)
    terms = [ _huber(d, delta=float(delta)).mean() for d in deltas.values() ]
    return torch.stack(terms).mean()


def delta_norm_huber(deltas: Dict[str, torch.Tensor], delta: float = 0.1, eps: float = 1e-12) -> torch.Tensor:
    """
    Norm huber on ||Δ||: stabilizes overall magnitude without killing sparse dims.
    """
    if not deltas:
        return torch.tensor(0.0)
    terms = []
    for d in deltas.values():
        norm = torch.sqrt(torch.sum(d * d, dim=1) + eps)
        terms.append(_huber(norm, delta=float(delta)).mean())
    return torch.stack(terms).mean()


def compute_static_stable_set_from_anchor(
    anchor: dict,
    k: int = 50,
    tau: float = 0.2,
    stable_frac: float = 0.3,
    chunk: int = 2048,
    device: str = "cpu",
):
    """
    Stable set = lowest-entropy fraction under top-k neighborhood distribution in anchor space.
    Robust to small N by clamping k.
    """
    pnames = list(anchor.keys())
    if len(pnames) == 0:
        return set(), np.array([], dtype=np.float32)

    A = np.stack([np.asarray(anchor[p], dtype=np.float32) for p in pnames], axis=0)
    A = torch.tensor(A, device=device)
    A = F.normalize(A, dim=1)

    N = A.shape[0]
    # If too small for kNN entropy, just call everything stable.
    if N <= 2:
        ent_np = np.zeros(N, dtype=np.float32)
        return set(pnames), ent_np

    k_eff = min(int(k), N - 1)
    ent = torch.empty(N, device=device)

    for start in range(0, N, int(chunk)):
        end = min(N, start + int(chunk))
        sims = A[start:end] @ A.T
        row_idx = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), row_idx] = -1e9

        topk_vals, _ = torch.topk(sims, k=k_eff, dim=1)
        probs = F.softmax(topk_vals / float(tau), dim=1)
        ent[start:end] = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)

    ent_np = ent.detach().cpu().numpy()
    cutoff = np.quantile(ent_np, float(stable_frac))
    stable_mask = ent_np <= cutoff
    stable_pnames = set(np.array(pnames)[stable_mask].tolist())
    return stable_pnames, ent_np
# -----------------------------------------------------------------------------
# MoE regularizers (compatible with your architecture MoEAdapter weights output)
# -----------------------------------------------------------------------------

def moe_router_entropy_mean(router_w: Dict[str, torch.Tensor], treated_names: List[str], eps: float = 1e-12) -> torch.Tensor:
    terms = []
    dev = None
    for r in treated_names:
        if r not in router_w:
            continue
        w = router_w[r]
        dev = w.device
        ent = -(w * torch.log(w + eps)).sum(dim=1)
        terms.append(ent.mean())
    if not terms:
        return torch.tensor(0.0, device=dev if dev is not None else "cpu")
    return torch.stack(terms).mean()


def moe_target_entropy_loss(
    router_w: Dict[str, torch.Tensor],
    treated_names: List[str],
    n_experts: int,
    top_k: int,
    target: str = "log_uniform_topk",
) -> torch.Tensor:
    H = moe_router_entropy_mean(router_w, treated_names)
    if target == "log_uniform_all":
        H_t = math.log(max(int(n_experts), 2))
    else:
        H_t = math.log(max(int(top_k), 1))
    H_t = torch.tensor(float(H_t), device=H.device, dtype=H.dtype)
    return (H - H_t) ** 2


def moe_topk_load_balance_loss(
    router_w: Dict[str, torch.Tensor],
    treated_names: List[str],
    n_experts: int,
    top_k: int,
) -> torch.Tensor:
    terms = []
    dev = None
    n_experts = int(n_experts)
    top_k = int(top_k)

    for r in treated_names:
        if r not in router_w:
            continue
        w = router_w[r]
        dev = w.device
        sel = (w > 0).to(w.dtype)
        p = sel.mean(dim=0)
        target = torch.full_like(p, float(top_k) / float(n_experts))
        terms.append(((p - target) ** 2).mean())

    if not terms:
        return torch.tensor(0.0, device=dev if dev is not None else "cpu")
    return torch.stack(terms).mean()


# -----------------------------------------------------------------------------
# Scheduling helpers
# -----------------------------------------------------------------------------

def _lin_ramp(epoch: int, start: int, end: int) -> float:
    if end <= start:
        return 1.0 if epoch >= start else 0.0
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return (epoch - start) / float(end - start)


def _cosine_anneal(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return float(end)
    t = min(max(epoch / float(total_epochs - 1), 0.0), 1.0)
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))


def warmup_lambda(epoch: int, peak: float, warmup_start: int, warmup_end: int) -> float:
    return float(peak) * _lin_ramp(epoch, warmup_start, warmup_end)



# -----------------------------------------------------------------------------
# Saving utilities (Stage 1)
# -----------------------------------------------------------------------------

def _save_losses(path: str, losses: List[Dict[str, float]]) -> None:
    if not losses:
        return
    keys = sorted(losses[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(keys)
        for row in losses:
            writer.writerow([row.get(k, "") for k in keys])




def save_results(
    model: UniEmbedNN,
    protein_dataset: Protein_Dataset,
    data_wrapper: TrainingDataWrapper,
    results_suffix: str = "",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Save per-modality latent TSVs only for proteins that are PRESENT in that modality
    (batch_mask[m] == 1). Co-embedding is computed as mean across PRESENT bases only.
    """
    model.eval()
    device = data_wrapper.device

    per_protein_latents: Dict[str, Dict[str, np.ndarray]] = {}
    per_modality_latents: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in protein_dataset.modality_names}

    loader = DataLoader(
        protein_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=_collate_protein_batch
    )

    with torch.no_grad():
        for batch_data, batch_mask, batch_idxs in loader:
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device)
                batch_mask[k] = batch_mask[k].to(device)

            latents, _ = model(batch_data)

            for bi, idx in enumerate(batch_idxs):
                pname = protein_dataset.protein_ids[idx]
                per_protein_latents.setdefault(pname, {})

                # only record latents for modalities where that protein is present
                for m in latents.keys():
                    if m not in batch_mask:
                        continue
                    if batch_mask[m][bi].item() <= 0:
                        continue

                    z = latents[m][bi].detach().cpu().numpy()
                    per_protein_latents[pname][m] = z
                    per_modality_latents[m][pname] = z

    # write per-modality latent TSVs
    for m, d in per_modality_latents.items():
        out = f"{data_wrapper.resultsdir}_{m}{results_suffix}_latent.tsv"
        write_embedding_dictionary_to_file(out, d, data_wrapper.latent_dim)

    # compute co-embedding: mean across PRESENT bases only
    coembed: Dict[str, np.ndarray] = {}
    for pname, md in per_protein_latents.items():
        if not md:
            continue

        base_to_vecs = defaultdict(list)
        for name, z in md.items():
            base = model._base_of.get(name, name)
            base_to_vecs[base].append(z)

        # average within each base, then average across bases (bases present for this protein)
        base_means = [np.mean(vs, axis=0) for vs in base_to_vecs.values()]
        coembed[pname] = np.mean(base_means, axis=0)

    out_co = f"{data_wrapper.resultsdir}{results_suffix}_latent.tsv"
    write_embedding_dictionary_to_file(out_co, coembed, data_wrapper.latent_dim)

    torch.save(model.state_dict(), f"{data_wrapper.resultsdir}{results_suffix}_model.pth")
    return per_protein_latents


# -----------------------------------------------------------------------------
# Stage 1: static training
# -----------------------------------------------------------------------------

def fit_predict(
    resultsdir: str,
    modalities_dict: Dict,
    modality_aliases: Dict[str, str],
    batch_size: int = 16,
    latent_dim: int = 128,
    n_epochs: int = 250,
    dropout: float = 0.0,
    l2_norm: bool = False,
    hidden_size_1: int = 512,
    hidden_size_2: int = 256,
    learn_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    save_update_epochs: bool = False,
    save_epoch: int = 50,
    replicate_groups: Optional[Dict[str, List[str]]] = None,
    lambda_replicate: float = 0.0,
    lambda_recon: float = 1.0,
    lambda_triplet: float = 1.0,
    triplet_margin: float = 1.0,
    modality_balance: str = "on",  # "on" | "off"
) -> Iterator[List[Union[str, float]]]:

    modality_names = list(modalities_dict.keys())
    os.makedirs(os.path.dirname(resultsdir) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    input_dims = {m: next(iter(modalities_dict[m].values())).shape[0] for m in modality_names}

    dw = TrainingDataWrapper(
        modalities_dict=modalities_dict,
        modality_names=list(modality_names),
        device=device,
        l2_norm=bool(l2_norm),
        dropout=float(dropout),
        latent_dim=int(latent_dim),
        hidden_size_1=int(hidden_size_1),
        hidden_size_2=int(hidden_size_2),
        resultsdir=resultsdir,
    )

    model = UniEmbedNN(
        input_dims=input_dims,
        modality_aliases=modality_aliases,
        latent_dim=dw.latent_dim,
        hidden_size_1=dw.hidden_size_1,
        hidden_size_2=dw.hidden_size_2,
        dropout=dw.dropout,
        l2_norm=dw.l2_norm,
    ).to(device)
    
    base_sim = compute_base_similarity_projected(
        modalities_dict, model._base_of,
        proj_dim=128, max_proteins=10000, min_overlap=500, seed=0
    )

    def dissim(a, b, floor=0.05):
        # 1 - similarity, with a floor so we don't zero anything out
        return max(floor, (1.0 - float(base_sim.get(a, {}).get(b, 0.0)))**2)
    
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    protein_dataset = Protein_Dataset(modalities_dict)
    train_loader = DataLoader(protein_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=_collate_protein_batch)

    losses_log: List[Dict[str, float]] = []

    for epoch in range(int(n_epochs)):
        model.train()
        epoch_losses: List[float] = []
        epoch_recon: List[float] = []
        epoch_trip: List[float] = []
        epoch_rep: List[float] = []

        for batch_data, batch_mask, _ in train_loader:
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device)
                batch_mask[k] = batch_mask[k].to(device)

            latents, _ = model(batch_data)

            base_latents, base_latent_present = _aggregate_latents_by_base(latents, 
                                                                           batch_mask, 
                                                                           model._base_of, 
                                                                           device)
            base_inputs, base_input_present = _aggregate_inputs_by_base(batch_data, 
                                                                        batch_mask, 
                                                                        model._base_of, 
                                                                        device)
            base_keys = list(base_latents.keys())

            # (1) Cross-modality reconstruction in input space using base decoders
            pair_losses = []
            for in_base in base_keys:
                z = base_latents[in_base]
                for out_base in base_keys:
                    mask = base_latent_present[in_base].bool() & base_input_present[out_base].bool()
                    if mask.sum() == 0:
                        continue
                    y_hat = model.decoders[out_base](z[mask])
                    y = base_inputs[out_base][mask]
                    pair_losses.append((1.0 - F.cosine_similarity(y_hat, y, dim=1)).mean())

            recon_loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0, device=device)


            # (2) Triplet across bases
            trip_terms = []
            balance_mode = str(modality_balance).lower().strip()  # "on" or "off"
            for anchor_base in base_keys:
                other_bases = [b for b in base_keys if b != anchor_base]
                if not other_bases:
                    continue

                # --- choose positive base ---
                if balance_mode == "off":
                    # uniform random: no dissimilarity-based sampling
                    pos_base = np.random.choice(other_bases)
                    w_ab = 1.0  # no dissimilarity weighting
                else:
                    # current behavior: dissimilarity-weighted sampling + weighting
                    weights = np.array([dissim(anchor_base, b) for b in other_bases], dtype=np.float64)
                    weights = weights / (weights.sum() + 1e-12)
                    pos_base = np.random.choice(other_bases, p=weights)
                    w_ab = float(dissim(anchor_base, pos_base))

                mask = base_latent_present[anchor_base].bool() & base_latent_present[pos_base].bool()
                if mask.sum() < 2:
                    continue

                a = base_latents[anchor_base][mask]
                p = base_latents[pos_base][mask]

                pos_d = 1.0 - F.cosine_similarity(a, p, dim=1)

                perm = torch.randperm(p.shape[0], device=device)
                if torch.all(perm == torch.arange(p.shape[0], device=device)):
                    perm = torch.roll(perm, 1)
                n = p[perm]
                neg_d = 1.0 - F.cosine_similarity(a, n, dim=1)

                trip = torch.clamp(pos_d - neg_d + float(triplet_margin), min=0.0)
                trip_terms.append(w_ab * trip)

            trip_loss = torch.mean(torch.cat(trip_terms)) if trip_terms else torch.tensor(0.0, device=device)


            # (3) Replicate agreement (optional)
            rep_loss = _replicate_consistency_loss(latents, batch_mask, replicate_groups, device)

            loss = (float(lambda_recon) * recon_loss) + (float(lambda_triplet) * trip_loss) + (float(lambda_replicate) * rep_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_recon.append(recon_loss.item())
            epoch_trip.append(trip_loss.item())
            epoch_rep.append(rep_loss.item())

        losses_log.append({
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses) if epoch_losses else np.nan),
            "recon": float(np.mean(epoch_recon) if epoch_recon else np.nan),
            "triplet": float(np.mean(epoch_trip) if epoch_trip else np.nan),
            "replicate": float(np.mean(epoch_rep) if epoch_rep else np.nan),
        })

        if save_update_epochs and (epoch % int(save_epoch) == 0) and epoch > 0:
            save_results(model, protein_dataset, dw, results_suffix=f"_epoch{epoch}")

    _save_losses(f"{resultsdir}_loss.tsv", losses_log)
    per_protein_latents = save_results(model, protein_dataset, dw)

    for pname in sorted(per_protein_latents.keys()):
        md = per_protein_latents[pname]
        base_to_vecs = defaultdict(list)
        for name, z in md.items():
            base = model._base_of.get(name, name)
            base_to_vecs[base].append(z)
        base_means = [np.mean(vs, axis=0) for vs in base_to_vecs.values()]
        z = np.mean(base_means, axis=0)
        yield [pname] + list(map(float, z.tolist()))


# -----------------------------------------------------------------------------
# Stage 2: treated adapter training (ANCHOR-ALIGNED)
# -----------------------------------------------------------------------------

def fit_treated_adapter(
    resultsdir: str,
    modalities_dict: Dict,
    modality_aliases: Dict[str, str],
    static_model_path: str,
    static_anchor_tsv: str,
    condition_name: str,
    treated_secms_names: List[str],
    latent_dim: int = 128,
    hidden_size_1: int = 512,
    hidden_size_2: int = 256,
    dropout: float = 0.0,
    l2_norm: bool = False,
    adapter_bottleneck: int = 64,
    batch_size: int = 16,
    n_epochs: int = 100,
    learn_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    # ---- core losses ----
    lambda_recon: float = 1.0,
    lambda_rep: float = 1.0,
    lambda_id: float = 0.05,
    id_tau: float = 0.3,
    lambda_knn: float = 0.2,
    knn_k: int = 30,
    knn_tau: float = 0.15,
    knn_entropy_thresh: float = 2.5,
    # ---- sparsity / stabilization on Δ_anchor ----
    lambda_delta_sparse: float = 0.01,
    delta_sparse_huber: float = 0.05,
    lambda_delta_norm: float = 0.0,      # optional extra stabilizer
    delta_norm_huber_delta: float = 0.1,
    # ---- dead-zone denoise on stable proteins (optional) ----
    lambda_anchor_denoise: float = 0.03,
    anchor_denoise_tau: float = 0.15,
    stable_frac: float = 0.3,
    # ---- MoE knobs ----
    adapter_type: str = "moe",
    moe_n_experts: int = 4,
    moe_top_k: int = 2,
    moe_router_hidden: int = 128,
    moe_router_temperature: float = 1.0,
    moe_temp_start: float = 1.5,
    moe_temp_end: float = 0.7,
    # ---- MoE regularizers ----
    lambda_route_entropy: float = 0.01,
    lambda_load_balance: float = 0.01,
    # ---- LoRA on SEC-MS encoder (stage 2 only) ----
    lora_rank: int = 4,                 # 0 disables LoRA
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_learn_rate: Optional[float] = None,
    # ---- Optional stable protein list override ----
    stable_protein_list_path: Optional[str] = None,
    stable_list_mode: str = "override",  # "override" | "union" | "intersection"

    # ---- Two-phase knobs ----
    two_phase: bool = True,
    phase1_frac: float = 0.1,               # fraction of epochs for LoRA alignment
    phase1_lambda_anchor: float = 1.0,      # strong anchor pull on stable set
    phase1_lambda_recon: float = 0.2,       # optional light recon to prevent collapse
    phase1_steps_per_epoch: int = 50,
    phase1_batch_size: int = 128,  # stable set ~200; full-batch is fine
    phase1_only_stable:bool =True,
    # ---- Phase-1 alignment method ----
    phase1_method: str = "aligner",   # "aligner" | "lora" | "none"
    # ---- safety ----
    anchor_coverage_min: float = 0.7,
    fail_fast_on_low_anchor: bool = True,
) -> Tuple[UniEmbedNN, TrainingDataWrapper]:
    # -----------------------------
    # Helpers: freeze/unfreeze sets
    # -----------------------------
    def _set_trainable_lora(model_: UniEmbedNN, flag: bool) -> None:
        for mod in model_.modules():
            if isinstance(mod, LoRALinear):
                mod.A.requires_grad_(flag)
                mod.B.requires_grad_(flag)
    def _collect_lora_params(model_: UniEmbedNN) -> List[torch.nn.Parameter]:
        ps = []
        for mod in model_.modules():
            if isinstance(mod, LoRALinear):
                if mod.A.requires_grad: ps.append(mod.A)
                if mod.B.requires_grad: ps.append(mod.B)
        return ps
    
    def _set_trainable_aligner(model_: UniEmbedNN, base: str, flag: bool) -> None:
        model_.set_aligner_trainable(base, flag)
    def _set_enabled_aligner(model_: UniEmbedNN, base: str, flag: bool) -> None:
        model_.set_aligner_enabled(base, flag)
    def _collect_aligner_params(model_: UniEmbedNN, base: str) -> List[torch.nn.Parameter]:
        if base not in model_.latent_aligners:
            return []
        return [p for p in model_.latent_aligners[base].parameters() if p.requires_grad]
    
    def _set_trainable_adapter(model_: UniEmbedNN, cond: str, flag: bool) -> None:
        if str(cond) not in model_.adapters:
            return
        for p in model_.adapters[str(cond)].parameters():
            p.requires_grad_(flag)
    def _set_trainable_gate(model_: UniEmbedNN, flag: bool) -> None:
        model_.adapter_gate.requires_grad_(flag)


    def _build_optimizer_for_current_phase() -> optim.Optimizer:
        groups = []

        # adapter params
        if str(condition_name) in model.adapters:
            adp = [p for p in model.adapters[str(condition_name)].parameters() if p.requires_grad]
            if adp:
                groups.append({"params": adp, "lr": float(learn_rate), "weight_decay": float(weight_decay)})

        # adapter gate
        if model.adapter_gate.requires_grad:
            groups.append({"params": [model.adapter_gate], "lr": float(learn_rate), "weight_decay": 0.0})

        # aligner params
        aps = _collect_aligner_params(model, secms_base)
        if aps:
            groups.append({"params": aps, "lr": float(learn_rate), "weight_decay": float(weight_decay)})

        # LoRA params
        lps = _collect_lora_params(model)
        if lps:
            groups.append({
                "params": lps,
                "lr": float(lora_learn_rate if (lora_learn_rate is not None) else learn_rate),
                "weight_decay": 0.0
            })

        if not groups:
            raise RuntimeError("No trainable params found for optimizer (check phase settings).")
        return optim.Adam(groups)




    os.makedirs(resultsdir, exist_ok=True)
    modality_names = list(modalities_dict.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dims = {m: next(iter(modalities_dict[m].values())).shape[0] for m in modality_names}

    dw = TrainingDataWrapper(
        modalities_dict=modalities_dict,
        modality_names=list(modality_names),
        device=device,
        l2_norm=bool(l2_norm),
        dropout=float(dropout),
        latent_dim=int(latent_dim),
        hidden_size_1=int(hidden_size_1),
        hidden_size_2=int(hidden_size_2),
        resultsdir=os.path.join(resultsdir, "adapter"),
    )

    model = UniEmbedNN(
        input_dims=input_dims,
        latent_dim=dw.latent_dim,
        hidden_size_1=dw.hidden_size_1,
        hidden_size_2=dw.hidden_size_2,
        dropout=dw.dropout,
        l2_norm=dw.l2_norm,
        modality_aliases=modality_aliases,
        adapted_modalities=treated_secms_names,
        conditions=[condition_name],
        adapter_bottleneck=int(adapter_bottleneck),
        adapter_type=adapter_type,
        moe_n_experts=moe_n_experts,
        moe_top_k=moe_top_k,
        moe_router_hidden=moe_router_hidden,
        moe_router_temperature=moe_router_temperature,
    ).to(device)

    
    # -----------------------------
    # Determine SEC-MS base + phase1 mode EARLY
    # -----------------------------
    secms_base = modality_aliases[treated_secms_names[0]]
    phase1_base = secms_base  # LoRA wraps encoder for this base

    method = str(phase1_method).lower().strip()
    if method not in {"aligner", "lora", "none"}:
        raise ValueError("phase1_method must be aligner|lora|none")

    use_lora_in_phase1 = (method == "lora") and (int(lora_rank) > 0)
    use_aligner_in_phase1 = (method == "aligner")

    if (method == "lora") and int(lora_rank) <= 0:
        raise ValueError("phase1_method='lora' requires lora_rank > 0")

    # -----------------------------
    # Load static weights (now secms_base exists)
    # -----------------------------
    state = torch.load(static_model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    print("missing_keys:", missing)
    print("unexpected_keys:", unexpected)

    loaded_keys = set(state.keys())
    if not any(k.startswith(f"encoders.{secms_base}.") for k in loaded_keys):
        raise RuntimeError(f"Static weights missing for encoders base='{secms_base}'. Check modality_aliases consistency.")
    if not any(k.startswith(f"decoders.{secms_base}.") for k in loaded_keys):
        raise RuntimeError(f"Static weights missing for decoders base='{secms_base}'. Check modality_aliases consistency.")


    # -----------------------------
    # Enable LoRA module wrapping once (if phase1 wants LoRA)
    # Important: wrapping changes modules; do it before optimizers.
    # -----------------------------
    if use_lora_in_phase1:
        model.enable_lora_last_linear(
            phase1_base,
            r=int(lora_rank),
            alpha=int(lora_alpha),
            dropout=float(lora_dropout),
        )
        
    # -----------------------------
    # Two-phase schedule
    # -----------------------------
    E = int(n_epochs)
    if method == "none":
        two_phase = False

    E1 = int(round(float(phase1_frac) * E)) if bool(two_phase) else 0
    E1 = max(0, min(E1, E))
    phase = "phase2" if (E1 == 0) else "phase1"

    # -----------------------------
    # Phase enter/exit: uses flags computed above
    # -----------------------------
    def _enter_phase(phase_name: str) -> None:
        nonlocal phase
        phase = phase_name

        if phase_name == "phase1":
            # no adapter in phase1
            model.set_condition(None)
            _set_trainable_adapter(model, condition_name, False)
            _set_trainable_gate(model, False)

            # default: disable/freeze both
            model.enable_only_these_aligners([])
            _set_trainable_aligner(model, secms_base, False)

            model.set_lora_enabled(False)
            _set_trainable_lora(model, False)

            # enable+train only the chosen mechanism
            if use_aligner_in_phase1:
                model.enable_only_these_aligners([secms_base])
                _set_trainable_aligner(model, secms_base, True)

            if use_lora_in_phase1:
                model.set_lora_enabled(True)
                _set_trainable_lora(model, True)

        else:  # phase2
            # adapter ON in phase2
            model.set_condition(str(condition_name))
            _set_trainable_adapter(model, condition_name, True)
            _set_trainable_gate(model, True)

            # keep phase1 transforms ACTIVE but FROZEN
            if use_aligner_in_phase1:
                model.enable_only_these_aligners([secms_base])
                _set_trainable_aligner(model, secms_base, False)
            else:
                model.enable_only_these_aligners([])

            if use_lora_in_phase1:
                model.set_lora_enabled(True)   # active in forward
                _set_trainable_lora(model, False)  # frozen
            else:
                model.set_lora_enabled(False)



    _enter_phase(phase)
    optimizer = _build_optimizer_for_current_phase()
    trainable = [(n, p.requires_grad) for n,p in model.named_parameters() if p.requires_grad]
    print("N trainable:", len(trainable))
    print(trainable[:20])

    # -----------------------------
    # Dataset + TWO loaders (phase1 can be >=128)
    # -----------------------------
    
    # -----------------------------
    # Dataset
    # -----------------------------
    protein_dataset = Protein_Dataset(modalities_dict)

    # -----------------------------
    # Load anchor + compute stable set BEFORE loaders
    # -----------------------------
    anchor = _load_anchor_tsv(static_anchor_tsv)

    stable_set, ent = compute_static_stable_set_from_anchor(
        anchor, k=int(knn_k), tau=float(knn_tau),
        stable_frac=float(stable_frac), chunk=2048, device=str(device)
    )

    dataset_proteins = set(protein_dataset.protein_ids)
    anchor_proteins = set(anchor.keys())
    available = dataset_proteins & anchor_proteins
    stable_pnames = set(stable_set) & available

    if stable_protein_list_path is not None:
        user_list = _load_protein_list(stable_protein_list_path)
        user_stable = set(user_list) & available
        mode = str(stable_list_mode).lower()
        if mode == "override":
            stable_pnames = user_stable
        elif mode == "union":
            stable_pnames = stable_pnames | user_stable
        elif mode == "intersection":
            stable_pnames = stable_pnames & user_stable
        else:
            raise ValueError(f"stable_list_mode must be override/union/intersection, got: {stable_list_mode}")

    print(f"[Stable set] size={len(stable_pnames)} (available={len(available)}) "
          f"(entropy={len(stable_set)}, user={'yes' if stable_protein_list_path else 'no'})")

    stable_indices = [i for i, pname in enumerate(protein_dataset.protein_ids) if pname in stable_pnames]
    if len(stable_indices) == 0:
        raise RuntimeError("Stable set is empty after intersection with dataset/anchor. Cannot run phase1.")

    # -----------------------------
    # TWO loaders
    # -----------------------------
    bs_phase2 = int(batch_size)
    bs_phase1 = min(int(phase1_batch_size), len(stable_indices))

    sampler_phase1 = SubsetRandomSampler(stable_indices)
    loader_phase1 = DataLoader(
        protein_dataset,
        batch_size=bs_phase1,
        sampler=sampler_phase1,
        shuffle=False,
        collate_fn=_collate_protein_batch,
        drop_last=False,
    )

    loader_phase2 = DataLoader(
        protein_dataset,
        batch_size=bs_phase2,
        shuffle=True,
        collate_fn=_collate_protein_batch,
    )


    rep_groups = {"treated_secms": list(treated_secms_names)} if len(treated_secms_names) >= 2 else None

    losses_log: List[Dict[str, float]] = []
   

    for epoch in range(E):
        model.train()
        # phase switch
        if bool(two_phase) and (epoch == E1) and (phase != "phase2"):
            _enter_phase("phase2")
            optimizer = _build_optimizer_for_current_phase()
        # phase2-relative epoch for warmups/anneals
        E2 = max(1, E - E1)
        p2_epoch = max(0, epoch - E1)
        p2_total = E2
        epoch_losses = []
        log_recon, log_rep, log_id, log_knn = [], [], [], []
        log_sparse, log_dnorm, log_denoise = [], [], []
        log_route, log_balance = [], []
        log_anchor_align = []
        temp_log = []

        # anchor coverage counters
        anchor_ctr = {"denom": 0, "anchored": 0}

        # schedules
        if phase == "phase2":
            warm_recon_end   = int(0.10 * p2_total)
            warm_anchor_end  = int(0.40 * p2_total)
            warm_sparse_s    = int(0.30 * p2_total)
            warm_sparse_e    = int(0.70 * p2_total)
            sched_epoch = p2_epoch
            sched_total = p2_total
        else:
            # phase1 uses its own direct objective; schedules below won't be used anyway
            warm_recon_end = warm_anchor_end = warm_sparse_s = warm_sparse_e = 0
            sched_epoch = 0
            sched_total = 1

        temp_eff = _cosine_anneal(sched_epoch, 
                                  sched_total, 
                                  start=float(moe_temp_start), 
                                  end=float(moe_temp_end))


        lambda_recon_eff = float(lambda_recon)
        lambda_rep_eff   = float(lambda_rep)

        lambda_id_eff  = warmup_lambda(sched_epoch, 
                                       peak=float(lambda_id),
                                       warmup_start=warm_recon_end,
                                       warmup_end=warm_anchor_end)
        lambda_knn_eff = warmup_lambda(sched_epoch, peak=float(lambda_knn),
                                       warmup_start=warm_recon_end,
                                       warmup_end=warm_anchor_end)
 

        lambda_sparse_eff = warmup_lambda(sched_epoch, 
                                          peak=float(lambda_delta_sparse), 
                                          warmup_start=warm_sparse_s, 
                                          warmup_end=warm_sparse_e)
        lambda_dnorm_eff  = warmup_lambda(sched_epoch, 
                                          peak=float(lambda_delta_norm),   
                                          warmup_start=warm_sparse_s, 
                                          warmup_end=warm_sparse_e)
 
        lambda_anchor_denoise_eff = warmup_lambda(sched_epoch, 
                                                  peak=float(lambda_anchor_denoise), 
                                                  warmup_start=warm_sparse_s, 
                                                  warmup_end=warm_sparse_e)
 
        lambda_load_balance_eff  = warmup_lambda(sched_epoch, 
                                                 peak=float(lambda_load_balance),  
                                                 warmup_start=warm_recon_end, 
                                                 warmup_end=warm_anchor_end)
        lambda_route_entropy_eff = warmup_lambda(sched_epoch, 
                                                 peak=float(lambda_route_entropy), 
                                                 warmup_start=warm_recon_end, 
                                                 warmup_end=warm_anchor_end)

        # -----------------------------
        # Choose iterator for this phase
        # -----------------------------
        if phase == "phase1":
            it = iter(loader_phase1)
            n_steps = int(phase1_steps_per_epoch)
        else:
            it = iter(loader_phase2)
            n_steps = None  # loop through full epoch

        def _run_one_step(batch_data, batch_mask, batch_idxs):
            

            # move to device
            for k in batch_data:
                batch_data[k] = batch_data[k].to(device)
                batch_mask[k] = batch_mask[k].to(device)

            # anchors aligned to batch rows
            anchors_batch = torch.zeros((len(batch_idxs), int(latent_dim)), device=device, dtype=torch.float32)
            anchor_mask_batch = torch.zeros((len(batch_idxs),), device=device, dtype=torch.bool)
            for bi, idx in enumerate(batch_idxs):
                pname = protein_dataset.protein_ids[idx]
                if pname in anchor:
                    anchors_batch[bi] = torch.tensor(anchor[pname], device=device, dtype=torch.float32)
                    anchor_mask_batch[bi] = True

            # forward
            if phase == "phase1":
                pre_latents, latents, outputs = model(
                    batch_data,
                    return_pre_latents=True,
                    return_deltas=False,
                    return_router=False,
                    anchors=anchors_batch,
                    anchor_mask=anchor_mask_batch,
                    input_mask=batch_mask,   
                )
                deltas = {}
                router_w = {}
            else:
                pre_latents, latents, outputs, deltas, router_w = model(
                    batch_data,
                    return_pre_latents=True,
                    return_deltas=True,
                    return_router=True,
                    router_temperature=temp_eff,
                    anchors=anchors_batch,
                    anchor_mask=anchor_mask_batch,
                    input_mask=batch_mask,  
                )

            # -------------------------
            # Anchor coverage counters
            # -------------------------
            with torch.no_grad():
                denom_any = torch.zeros_like(batch_mask[treated_secms_names[0]])
                for r in treated_secms_names:
                    denom_any += batch_mask[r].float()
                present_any = denom_any > 0
                anchor_ctr["denom"] += int(present_any.sum().item())
                anchor_ctr["anchored"] += int((present_any & anchor_mask_batch).sum().item())

            # (1) Recon on treated SEC-MS (self)
            recon_terms = []
            for r in treated_secms_names:
                m = batch_mask[r].bool()
                if m.sum() == 0:
                    continue
                y_hat = outputs[f"{r}_{r}"][m]
                y = batch_data[r][m]
                recon_terms.append(1.0 - F.cosine_similarity(y_hat, y, dim=1))
            recon_loss = torch.mean(torch.cat(recon_terms)) if recon_terms else torch.tensor(0.0, device=device)

            # (2) Replicate Δ_anchor consistency (phase2 only)
            rep_loss = torch.tensor(0.0, device=device) if (phase == "phase1") else \
                _replicate_delta_anchor_consistency_loss(deltas, batch_mask, rep_groups, device)

            # optional aligner reg (phase1 only)
            aligner_reg = torch.tensor(0.0, device=device)
            lambda_aligner_id = 1e-3
            if phase == "phase1" and use_aligner_in_phase1:
                aligner_reg = model.latent_aligners[secms_base].reg_identity()

            # Build treated mean latent Z_mean + anchors A
            zs, ms = [], []
            for r in treated_secms_names:
                zs.append(latents[r])
                ms.append(batch_mask[r].bool())
            z_stack = torch.stack(zs, dim=0)
            m_stack = torch.stack(ms, dim=0)

            z_sum = torch.zeros_like(z_stack[0])
            denom = torch.zeros(z_stack.shape[1], device=device)
            for rr in range(z_stack.shape[0]):
                mr = m_stack[rr]
                z_sum[mr] += z_stack[rr][mr]
                denom[mr] += 1
            z_mean = z_sum / (denom.unsqueeze(1) + 1e-8)

            A_list, Z_list, stable_mask_list = [], [], []
            for bi, idx in enumerate(batch_idxs):
                if denom[bi] <= 0:
                    continue
                if not anchor_mask_batch[bi]:
                    continue
                pname = protein_dataset.protein_ids[idx]
                A_list.append(anchors_batch[bi])
                Z_list.append(z_mean[bi])
                stable_mask_list.append(pname in stable_pnames)

            if len(A_list) < 2:
                anchor_align_loss = torch.tensor(0.0, device=device)
                id_loss = torch.tensor(0.0, device=device)
                knn_loss = torch.tensor(0.0, device=device)
                anchor_denoise_loss = torch.tensor(0.0, device=device)
            else:
                A = torch.stack(A_list, dim=0)
                Z = torch.stack(Z_list, dim=0)

                stable_mask = torch.tensor(stable_mask_list, device=device, dtype=torch.bool)

                # anchor align: phase1 stable-only if enabled
                if bool(phase1_only_stable) and stable_mask.any():
                    A_use, Z_use = A[stable_mask], Z[stable_mask]
                else:
                    A_use, Z_use = A, Z

                anchor_align_loss = (1.0 - F.cosine_similarity(F.normalize(Z_use, dim=1),
                                                              F.normalize(A_use, dim=1),
                                                              dim=1)).mean() if A_use.shape[0] > 0 \
                                   else torch.tensor(0.0, device=device)

                if phase == "phase1":
                    id_loss = torch.tensor(0.0, device=device)
                    knn_loss = torch.tensor(0.0, device=device)
                else:
                    id_loss = identity_infonce_loss(A, Z, tau=float(id_tau))
                    knn_loss = neighborhood_kl_loss_knn_masked(
                        A, Z, k=int(knn_k), tau=float(knn_tau),
                        entropy_thresh=float(knn_entropy_thresh)
                    )

                # dead-zone denoise on stable proteins
                if stable_mask.any():
                    delta_anchor = (Z - A)[stable_mask]
                    mag = torch.norm(delta_anchor, dim=1)
                    anchor_denoise_loss = torch.relu(mag - float(anchor_denoise_tau)).mean()
                else:
                    anchor_denoise_loss = torch.tensor(0.0, device=device)

            # phase2-only regs
            if phase == "phase1":
                reg_sparse = torch.tensor(0.0, device=device)
                reg_dnorm  = torch.tensor(0.0, device=device)
                load_bal = torch.tensor(0.0, device=device)
                route_ent_loss = torch.tensor(0.0, device=device)
            else:
                reg_sparse = delta_sparsity_huber(deltas, delta=float(delta_sparse_huber)).to(device)
                reg_dnorm  = delta_norm_huber(deltas, delta=float(delta_norm_huber_delta)).to(device) \
                             if float(lambda_delta_norm) > 0 else torch.tensor(0.0, device=device)

                if router_w:
                    load_bal = moe_topk_load_balance_loss(router_w, treated_secms_names,
                                                          n_experts=int(moe_n_experts), top_k=int(moe_top_k))
                    route_ent_loss = moe_target_entropy_loss(router_w, treated_secms_names,
                                                             n_experts=int(moe_n_experts), top_k=int(moe_top_k),
                                                             target="log_uniform_topk")
                else:
                    load_bal = torch.tensor(0.0, device=device)
                    route_ent_loss = torch.tensor(0.0, device=device)

            # total loss
            if phase == "phase1":
                loss = phase1_lambda_anchor * anchor_align_loss + phase1_lambda_recon * recon_loss
                if use_aligner_in_phase1:
                    loss = loss + lambda_aligner_id * aligner_reg
            else:
                loss = (
                    lambda_recon_eff * recon_loss +
                    lambda_rep_eff   * rep_loss +
                    lambda_id_eff    * id_loss +
                    lambda_knn_eff   * knn_loss +
                    lambda_sparse_eff * reg_sparse +
                    lambda_dnorm_eff  * reg_dnorm +
                    lambda_anchor_denoise_eff * anchor_denoise_loss +
                    lambda_load_balance_eff * load_bal +
                    lambda_route_entropy_eff * route_ent_loss
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            epoch_losses.append(loss.item())
            log_recon.append(recon_loss.item())
            log_rep.append(rep_loss.item())
            log_id.append(id_loss.item())
            log_knn.append(knn_loss.item())
            log_sparse.append(reg_sparse.item())
            log_dnorm.append(reg_dnorm.item())
            log_denoise.append(anchor_denoise_loss.item())
            log_route.append(route_ent_loss.item())
            log_balance.append(load_bal.item())
            log_anchor_align.append(anchor_align_loss.item())
            temp_log.append(temp_eff)

        # -----------------------------
        # Iterate
        # -----------------------------
        if phase == "phase1":
            for _ in range(int(phase1_steps_per_epoch)):
                try:
                    batch_data, batch_mask, batch_idxs = next(it)
                except StopIteration:
                    it = iter(loader_phase1)
                    batch_data, batch_mask, batch_idxs = next(it)
                _run_one_step(batch_data, batch_mask, batch_idxs)
        else:
            for batch_data, batch_mask, batch_idxs in loader_phase2:
                _run_one_step(batch_data, batch_mask, batch_idxs)

        epoch_denom_count = int(anchor_ctr["denom"])
        epoch_anchored_count = int(anchor_ctr["anchored"])
        anchor_coverage = (float(epoch_anchored_count) / float(epoch_denom_count)) if epoch_denom_count > 0 else float("nan")

        if bool(fail_fast_on_low_anchor) and (epoch_denom_count > 0) and (anchor_coverage < float(anchor_coverage_min)):
            raise RuntimeError(
                f"[Anchor coverage check] epoch={epoch} anchor_coverage={anchor_coverage:.3f} "
                f"(anchored={epoch_anchored_count}, denom>0={epoch_denom_count}) < min={float(anchor_coverage_min):.2f}. "
                "Likely protein ID mismatch between treated data and static anchor TSV."
            )

        losses_log.append({
            "epoch": epoch,
            "phase": phase,
            "loss": float(np.mean(epoch_losses) if epoch_losses else np.nan),
            "recon": float(np.mean(log_recon) if log_recon else np.nan),
            "replicate": float(np.mean(log_rep) if log_rep else np.nan),
            "id": float(np.mean(log_id) if log_id else np.nan),
            "knn": float(np.mean(log_knn) if log_knn else np.nan),
            "reg_sparse": float(np.mean(log_sparse) if log_sparse else np.nan),
            "reg_dnorm": float(np.mean(log_dnorm) if log_dnorm else np.nan),
            "anchor_denoise": float(np.mean(log_denoise) if log_denoise else np.nan),
            "anchor_align": float(np.mean(log_anchor_align) if log_anchor_align else np.nan),
            "route_ent": float(np.mean(log_route) if log_route else np.nan),
            "load_bal": float(np.mean(log_balance) if log_balance else np.nan),
            "moe_temp": float(np.mean(temp_log) if temp_log else temp_eff),
            "anchor_coverage": float(anchor_coverage),
            "anchor_anchored": float(epoch_anchored_count),
            "anchor_denom_gt0": float(epoch_denom_count),
        })


    _save_losses(os.path.join(resultsdir, f"loss_{condition_name}.tsv"), losses_log)
    
    torch.save(model.state_dict(), os.path.join(resultsdir, f"treated_{condition_name}_model.pth"))
    return model, dw


# -----------------------------------------------------------------------------
# Saving treated embeddings (Stage 2 outputs) - aligned with new deltas
# -----------------------------------------------------------------------------

def save_treated_embeddings(
    resultsdir: str,
    model: UniEmbedNN,
    protein_dataset: Protein_Dataset,
    data_wrapper: TrainingDataWrapper,
    condition_name: str,
    treated_secms_names: List[str],
    static_anchor_tsv: str,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> None:
    """
    Writes:
      treated_<cond>_<rep>_latent.tsv
      treated_<cond>_secms_mean_latent.tsv
      treated_<cond>_coembedding.tsv            (treated if present else static)
      treated_<cond>_delta_mean.tsv             (mean Δ_anchor across reps)
      treated_<cond>_delta_rank.tsv             (protein, ||Δ_anchor||, n_reps)
    """
    os.makedirs(resultsdir, exist_ok=True)
    device = data_wrapper.device
    latent_dim = data_wrapper.latent_dim

    model.set_condition(str(condition_name))
    model.eval()

    static_anchor = _load_anchor_tsv(static_anchor_tsv)

    def _write_header(writer):
        writer.writerow(["protein"] + [str(i) for i in range(1, latent_dim + 1)])

    def _write_vec(writer, pname: str, vec: np.ndarray):
        writer.writerow([pname] + [f"{x:.6g}" for x in vec.tolist()])

    outbase = os.path.join(resultsdir, f"treated_{condition_name}")

    rep_files, rep_writers = {}, {}
    for r in treated_secms_names:
        fp = open(f"{outbase}_{r}_latent.tsv", "w", newline="")
        rep_files[r] = fp
        w = csv.writer(fp, delimiter="\t")
        rep_writers[r] = w
        _write_header(w)
    # split delta into pre (LoRA/global) and adapter-only components if architecture supports it
    delta_pre_fp = open(f"{outbase}_delta_pre_mean.tsv", "w", newline="")
    delta_pre_writer = csv.writer(delta_pre_fp, delimiter="\t")
    _write_header(delta_pre_writer)

    delta_adp_fp = open(f"{outbase}_delta_adapter_mean.tsv", "w", newline="")
    delta_adp_writer = csv.writer(delta_adp_fp, delimiter="\t")
    _write_header(delta_adp_writer)
    
    mean_fp = open(f"{outbase}_secms_mean_latent.tsv", "w", newline="")
    mean_writer = csv.writer(mean_fp, delimiter="\t")
    _write_header(mean_writer)

    co_fp = open(f"{outbase}_coembedding.tsv", "w", newline="")
    co_writer = csv.writer(co_fp, delimiter="\t")
    _write_header(co_writer)

    delta_fp = open(f"{outbase}_delta_mean.tsv", "w", newline="")
    delta_writer = csv.writer(delta_fp, delimiter="\t")
    _write_header(delta_writer)

    delta_rank_fp = open(f"{outbase}_delta_rank.tsv", "w", newline="")
    delta_rank_writer = csv.writer(delta_rank_fp, delimiter="\t")
    delta_rank_writer.writerow(["protein", "delta_l2", "n_reps"])

    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        protein_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_protein_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    try:
        with torch.no_grad():
            for batch_data, batch_mask, batch_idxs in loader:
                for k in batch_data:
                    batch_data[k] = batch_data[k].to(device, non_blocking=True)
                    batch_mask[k] = batch_mask[k].to(device, non_blocking=True)

                # anchors aligned to batch rows
                anchors_batch = torch.zeros((len(batch_idxs), int(latent_dim)), device=device, dtype=torch.float32)
                anchor_mask_batch = torch.zeros((len(batch_idxs),), device=device, dtype=torch.bool)
                for bi, idx in enumerate(batch_idxs):
                    pname = protein_dataset.protein_ids[idx]
                    if pname in static_anchor:
                        anchors_batch[bi] = torch.tensor(static_anchor[pname], device=device, dtype=torch.float32)
                        anchor_mask_batch[bi] = True


                # with anchors, deltas returned are Δ_anchor = z_out - anchor
                # try to also request delta components if available
                try:
                    latents, _, deltas, dcomp = model(
                        batch_data,
                        return_deltas=True,
                        return_delta_components=True,
                        anchors=anchors_batch,
                        anchor_mask=anchor_mask_batch,
                        input_mask=batch_mask,   
                    )

                except TypeError:
                    latents, _, deltas = model(
                        batch_data,
                        return_deltas=True,
                        anchors=anchors_batch,
                        anchor_mask=anchor_mask_batch,
                        input_mask=batch_mask,   # <-- ADD
                    )

                    dcomp = {}

                for bi, idx in enumerate(batch_idxs):
                    pname = protein_dataset.protein_ids[idx]

                    zs = []
                    ds = []
                    pre_ds = []
                    adp_ds = []
                    for r in treated_secms_names:
                        if batch_mask[r][bi].item() > 0:
                            z = latents[r][bi].detach().cpu().numpy()
                            _write_vec(rep_writers[r], pname, z)
                            zs.append(z)

                            if r in deltas:
                                d = deltas[r][bi].detach().cpu().numpy()
                                ds.append(d)
                            if isinstance(dcomp, dict):
                                if "pre" in dcomp and r in dcomp["pre"]:
                                    pre_ds.append(dcomp["pre"][r][bi].detach().cpu().numpy())
                                if "adapter" in dcomp and r in dcomp["adapter"]:
                                    adp_ds.append(dcomp["adapter"][r][bi].detach().cpu().numpy())

                    z_treat_mean = None
                    if zs:
                        z_treat_mean = np.mean(zs, axis=0)
                        _write_vec(mean_writer, pname, z_treat_mean)

                    if ds:
                        d_mean = np.mean(ds, axis=0)
                        _write_vec(delta_writer, pname, d_mean)
                        delta_l2 = float(np.linalg.norm(d_mean))
                        delta_rank_writer.writerow([pname, f"{delta_l2:.6g}", str(len(ds))])
                    if pre_ds:
                        _write_vec(delta_pre_writer, pname, np.mean(pre_ds, axis=0))
                    if adp_ds:
                        _write_vec(delta_adp_writer, pname, np.mean(adp_ds, axis=0))

                    z_static = static_anchor.get(pname, None)
                    # coembedding for treatment: prefer anchor + mean_delta (interpretable)
                    if (z_static is not None) and ds:
                        d_mean = np.mean(ds, axis=0)
                        z_co = z_static + d_mean
                        _write_vec(co_writer, pname, z_co)
                    '''
                    elif z_static is not None:
                        # fallback if no treated data at all
                        _write_vec(co_writer, pname, z_static)
                    # coembedding: treated if present else static
                    if z_treat_mean is not None:
                        _write_vec(co_writer, pname, z_treat_mean)
                    elif z_static is not None:
                        _write_vec(co_writer, pname, z_static)
                    '''
    finally:
        for fp in rep_files.values():
            fp.close()
        mean_fp.close()
        co_fp.close()
        delta_fp.close()
        delta_pre_fp.close()
        delta_adp_fp.close()
        delta_rank_fp.close()
