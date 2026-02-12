#! /usr/bin/env python
"""
Two-stage runner for CellMaps co-embedding with SEC-MS condition adapters.

This runner matches the UPDATED:
- architecture.py (anchor-centered adaptation)
- __init__.py (fit_treated_adapter returns Δ_anchor via anchors mechanism)

Main changes vs older runner:
- removed lambda_nodrift (no longer supported)
- added lambda_anchor_denoise / anchor_denoise_tau / stable_frac
- optional lambda_delta_norm / delta_norm_huber_delta
- optional anchor coverage fail-fast knobs
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


from exceptions import CellmapsCoEmbeddingError
import two_stage as twostage


# -----------------------------------------------------------------------------
# Manifest parsing
# -----------------------------------------------------------------------------

def _load_manifest(path: str) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise CellmapsCoEmbeddingError("Manifest must be a JSON list of entries.")

    names = set()
    for e in data:
        for k in ("name", "modality", "condition", "replicate", "path"):
            if k not in e:
                raise CellmapsCoEmbeddingError(f"Manifest entry missing '{k}': {e}")

        if e["name"] in names:
            raise CellmapsCoEmbeddingError(f"Duplicate entry name in manifest: {e['name']}")
        names.add(e["name"])

        if "include_in_static" not in e:
            e["include_in_static"] = False

    return data


def _split_manifest(entries: List[dict]) -> Tuple[List[dict], Dict[str, List[dict]]]:
    static_entries = [e for e in entries if bool(e.get("include_in_static", False))]

    treated_groups: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        if bool(e.get("include_in_static", False)):
            continue
        cond = e.get("condition")
        if cond is None or cond == "untreated":
            continue
        treated_groups[cond].append(e)

    for cond in treated_groups:
        treated_groups[cond] = sorted(treated_groups[cond], key=lambda x: int(x.get("replicate", 0)))

    return static_entries, dict(treated_groups)


# -----------------------------------------------------------------------------
# Two-stage runner
# -----------------------------------------------------------------------------

def run_two_stage(
    manifest_path: str,
    outdir: str,
    latent_dim: int = 128,
    epochs_static: int = 250,
    epochs_adapter: int = 100,
    batch_size: int = 16,
    dropout: float = 0.0,
    l2_norm: bool = False,
    hidden_size_1: int = 512,
    hidden_size_2: int = 256,
    learn_rate_adapter: float = 1e-4,
    weight_decay_adapter: float = 1e-4,

    stable_list_mode: str = "override",  # override|union|intersection

    two_phase: bool = True,
    phase1_lambda_anchor: float = 1.0,
    phase1_lambda_recon: float = 0.2,
    phase1_steps_per_epoch: int = 50,
    phase1_batch_size: int = 128,
    phase1_only_stable: bool = True,

    # ---- Stage 1 ----
    lambda_replicate_static: float = 0.0,
    lambda_recon_static: float = 1.0,
    lambda_triplet: float = 1.0,
    modality_balance: str = "on",
    quality_weight_mode: str = "auto",
    quality_floor: float = 0.1,
    quality_exponent: float = 1.0,
    quality_rank_weight: float = 0.3,
    # ---- Stage 2 ----
    adapter_bottleneck: int = 64,
    lambda_recon: float = 1.0,
    lambda_rep: float = 1.0,
    lambda_id: float = 0.1,
    id_tau: float = 0.3,
    lambda_knn: float = 0.5,
    knn_k: int = 20,
    knn_tau: float = 0.2,
    knn_entropy_thresh: float = 2.5,
    lambda_delta_sparse: float = 0.05,
    delta_sparse_huber: float = 0.05,
    # optional extra stabilizer on ||Δ||
    lambda_delta_norm: float = 0.0,
    delta_norm_huber_delta: float = 0.1,
    # stable-only dead-zone denoise
    lambda_anchor_denoise: float = 0.03,
    anchor_denoise_tau: float = 0.2,
    stable_frac: float = 0.1,
    # ---- MoE knobs ----
    adapter_type: str = "moe",
    moe_n_experts: int = 4,
    moe_top_k: int = 2,
    moe_router_hidden: int = 128,
    moe_router_temperature: float = 1.0,
    lambda_route_entropy: float = 0.01,
    lambda_load_balance: float = 0.01,
    moe_temp_start: float = 1.5,
    moe_temp_end: float = 0.7,
    # ---- LoRA on SEC-MS encoder (stage 2 only) ----
    lora_rank: int = 4,                 # 0 disables LoRA
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_learn_rate: Optional[float] = None,
    # ---- Phase-1 alignment method ----
    phase1_method: str = "aligner",   # "aligner" | "lora" | "none"
    stable_protein_list_path=None,
    phase1_frac=0.2,
    # ---- anchor coverage safety ----
    anchor_coverage_min: float = 0.7,
    fail_fast_on_low_anchor: bool = True,
) -> None:

    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    entries = _load_manifest(manifest_path)
    static_entries, treated_groups = _split_manifest(entries)

    if len(static_entries) < 2:
        raise CellmapsCoEmbeddingError("Stage 1 needs at least two static inputs (include_in_static=true).")

    # -------------------------
    # Stage 1: static training
    # -------------------------
    static_modalities, static_aliases = twostage.load_modalities_from_manifest(static_entries)
    resultsdir_static = os.path.join(outdir, "auto_static")

    # replicate groups by base modality
    rep_groups = defaultdict(list)
    for e in static_entries:
        rep_groups[e["modality"]].append(e["name"])
    rep_groups = {k: v for k, v in rep_groups.items() if len(v) >= 2}

    for _ in twostage.fit_predict(
        resultsdir=resultsdir_static,
        modalities_dict=static_modalities,
        modality_aliases=static_aliases,
        modality_balance=modality_balance,
        latent_dim=latent_dim,
        n_epochs=epochs_static,
        batch_size=batch_size,
        lambda_triplet=lambda_triplet,
        lambda_recon=lambda_recon_static,
        dropout=dropout,
        l2_norm=l2_norm,
        save_update_epochs=False,
        replicate_groups=rep_groups if rep_groups else None,
        lambda_replicate=lambda_replicate_static,
        quality_weight_mode=quality_weight_mode,
        quality_floor=quality_floor,
        quality_exponent=quality_exponent,
        quality_rank_weight=quality_rank_weight,
    ):
        pass

    static_model_path = f"{resultsdir_static}_model.pth"
    static_anchor_tsv = f"{resultsdir_static}_latent.tsv"

    if not os.path.exists(static_model_path):
        raise CellmapsCoEmbeddingError(f"Static model not found: {static_model_path}")
    if not os.path.exists(static_anchor_tsv):
        raise CellmapsCoEmbeddingError(f"Static co-embedding TSV not found: {static_anchor_tsv}")

    # -------------------------
    # Stage 2: per-condition adapter
    # -------------------------
    for cond, secms_entries in treated_groups.items():
        treated_names = [e["name"] for e in secms_entries]

        treated_modalities, treated_aliases = twostage.load_modalities_from_manifest(secms_entries)
        resultsdir_cond = os.path.join(outdir, f"auto_adapter_{cond}")

        model2, dw2 = twostage.fit_treated_adapter(
            resultsdir=resultsdir_cond,
            modalities_dict=treated_modalities,
            modality_aliases=treated_aliases,
            static_model_path=static_model_path,
            static_anchor_tsv=static_anchor_tsv,
            condition_name=cond,
            treated_secms_names=treated_names,

            latent_dim=latent_dim,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            dropout=dropout,
            l2_norm=l2_norm,
            adapter_bottleneck=adapter_bottleneck,

            batch_size=batch_size,
            n_epochs=epochs_adapter,
            learn_rate=learn_rate_adapter,
            weight_decay=weight_decay_adapter,

            # --- core losses ---
            lambda_recon=lambda_recon,
            lambda_rep=lambda_rep,
            lambda_id=lambda_id,
            id_tau=id_tau,
            lambda_knn=lambda_knn,
            knn_k=knn_k,
            knn_tau=knn_tau,
            knn_entropy_thresh=knn_entropy_thresh,

            # --- Δ_anchor regularization ---
            lambda_delta_sparse=lambda_delta_sparse,
            delta_sparse_huber=delta_sparse_huber,
            lambda_delta_norm=lambda_delta_norm,
            delta_norm_huber_delta=delta_norm_huber_delta,

            # --- stable denoise ---
            lambda_anchor_denoise=lambda_anchor_denoise,
            anchor_denoise_tau=anchor_denoise_tau,
            stable_frac=stable_frac,

            # --- adapter / MoE ---
            adapter_type=adapter_type,
            moe_n_experts=moe_n_experts,
            moe_top_k=moe_top_k,
            moe_router_hidden=moe_router_hidden,
            moe_router_temperature=moe_router_temperature,
            moe_temp_start=moe_temp_start,
            moe_temp_end=moe_temp_end,
            lambda_route_entropy=lambda_route_entropy,
            lambda_load_balance=lambda_load_balance,

            # --- LoRA ---
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_learn_rate=lora_learn_rate,

            # --- stable list override ---
            stable_protein_list_path=stable_protein_list_path,
            stable_list_mode=stable_list_mode,
            
            # --- two-phase ---
            two_phase=two_phase,
            phase1_frac=phase1_frac,
            phase1_lambda_anchor=phase1_lambda_anchor,
            phase1_lambda_recon=phase1_lambda_recon,
            phase1_steps_per_epoch=phase1_steps_per_epoch,
            phase1_batch_size=phase1_batch_size,
            phase1_method=phase1_method,
            phase1_only_stable=phase1_only_stable,

            # --- safety ---
            anchor_coverage_min=anchor_coverage_min,
            fail_fast_on_low_anchor=fail_fast_on_low_anchor,
        )


        protein_dataset = twostage.Protein_Dataset(dw2.modalities_dict)

        twostage.save_treated_embeddings(
            resultsdir=resultsdir_cond,
            model=model2,
            protein_dataset=protein_dataset,
            data_wrapper=dw2,
            condition_name=cond,
            treated_secms_names=treated_names,
            static_anchor_tsv=static_anchor_tsv,
        )

    print(f"[OK] Two-stage run complete. Outputs written to: {outdir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-stage SEC-MS adapter runner (manifest-driven).")
    p.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--epochs_static", type=int, default=500)
    p.add_argument("--epochs_adapter", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l2_norm", action="store_true")
    # model sizes
    p.add_argument("--hidden_size_1", type=int, default=512)
    p.add_argument("--hidden_size_2", type=int, default=256)

    # optimizer
    p.add_argument("--learn_rate_adapter", type=float, default=1e-4)
    p.add_argument("--weight_decay_adapter", type=float, default=1e-4)

    # Stage 1
    p.add_argument("--lambda_replicate_static", type=float, default=0.75)
    p.add_argument("--lambda_recon_static", type=float, default=0.9)
    p.add_argument("--lambda_triplet", type=float, default=0.8)
    p.add_argument("--modality_balance", default="on", choices=["on", "off"],
               help="Stage1: if off, sample triplet pairs uniformly and do not weight by modality dissimilarity.")
    p.add_argument("--quality_weight_mode", default="auto", choices=["auto", "off"],
               help="Stage1: if auto, down-weight low-discriminability modalities (e.g. SEC-MS with high pairwise similarity).")
    p.add_argument("--quality_floor", type=float, default=0.1,
               help="Stage1: minimum quality weight for any modality (prevents total silencing).")
    p.add_argument("--quality_exponent", type=float, default=1.0,
               help="Stage1: exponent applied to (1 - mean_cosine_sim) when computing quality weights.")
    p.add_argument("--quality_rank_weight", type=float, default=0.3,
               help="Stage1: weight of effective-rank signal in quality score (0=pure discriminability, 1=pure rank).")


    # Stage 2
    p.add_argument("--adapter_bottleneck", type=int, default=64)
    p.add_argument("--lambda_recon", type=float, default=0.6)
    p.add_argument("--lambda_rep", type=float, default=0.5)
    p.add_argument("--lambda_id", type=float, default=0.02)
    p.add_argument("--id_tau", type=float, default=0.3)

    p.add_argument("--lambda_delta_sparse", type=float, default=0.01)
    p.add_argument("--delta_sparse_huber", type=float, default=0.05)

    p.add_argument("--lambda_delta_norm", type=float, default=0.08)
    p.add_argument("--delta_norm_huber_delta", type=float, default=0.1)

    p.add_argument("--lambda_knn", type=float, default=0.02)
    p.add_argument("--knn_k", type=int, default=30)
    p.add_argument("--knn_tau", type=float, default=0.2)
    p.add_argument("--knn_entropy_thresh", type=float, default=2.3)

    p.add_argument("--lambda_anchor_denoise", type=float, default=0.05)
    p.add_argument("--anchor_denoise_tau", type=float, default=0.3)
    p.add_argument("--stable_frac", type=float, default=0.1)

    # MoE
    p.add_argument("--adapter_type", type=str, default="moe", choices=["residual", "moe"])
    p.add_argument("--moe_n_experts", type=int, default=4)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--moe_router_hidden", type=int, default=128)
    p.add_argument("--moe_router_temperature", type=float, default=1.0)
    p.add_argument("--lambda_route_entropy", type=float, default=0.005)
    p.add_argument("--lambda_load_balance", type=float, default=0.005)
    p.add_argument("--moe_temp_start", type=float, default=1.5)
    p.add_argument("--moe_temp_end", type=float, default=0.7)
    # --- stable list override behavior ---
    p.add_argument("--stable_protein_list_path", default=None)
    p.add_argument("--stable_list_mode", default="override",
                   choices=["override", "union", "intersection"])

    # --- two-phase knobs ---
    p.add_argument("--two_phase", action="store_true")
    p.add_argument("--no_two_phase", dest="two_phase", action="store_false")
    p.set_defaults(two_phase=True)

    p.add_argument("--phase1_frac", type=float, default=0.1)
    p.add_argument("--phase1_lambda_anchor", type=float, default=1.0)
    p.add_argument("--phase1_lambda_recon", type=float, default=0.2)
    p.add_argument("--phase1_steps_per_epoch", type=int, default=50)
    p.add_argument("--phase1_batch_size", type=int, default=128)

    p.add_argument("--phase1_method", default="aligner",
                   choices=["aligner", "lora", "none"])
    p.add_argument("--phase1_only_stable", action="store_true")
    p.add_argument("--no_phase1_only_stable", dest="phase1_only_stable", action="store_false")
    p.set_defaults(phase1_only_stable=True)
    
    # ---- LoRA on SEC-MS encoder (stage 2 only) ----
    p.add_argument("--lora_rank", type=int, default=4) # 0 disables LoRA
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_learn_rate", type=float, default=1e-5)
 

    # anchor coverage safety
    p.add_argument("--anchor_coverage_min", type=float, default=0.7)
    p.add_argument("--no_fail_fast_on_low_anchor", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    run_two_stage(
        manifest_path=args.manifest,
        outdir=args.outdir,
        latent_dim=args.latent_dim,
        epochs_static=args.epochs_static,
        epochs_adapter=args.epochs_adapter,
        batch_size=args.batch_size,
        dropout=args.dropout,
        l2_norm=args.l2_norm,
        hidden_size_1=args.hidden_size_1,
        hidden_size_2=args.hidden_size_2,
        learn_rate_adapter=args.learn_rate_adapter,
        weight_decay_adapter=args.weight_decay_adapter,
        modality_balance=args.modality_balance if hasattr(args, "modality_balance") else "on",
        quality_weight_mode=args.quality_weight_mode,
        quality_floor=args.quality_floor,
        quality_exponent=args.quality_exponent,
        quality_rank_weight=args.quality_rank_weight,
        stable_list_mode=args.stable_list_mode,
        two_phase=args.two_phase,
        phase1_lambda_anchor=args.phase1_lambda_anchor,
        phase1_lambda_recon=args.phase1_lambda_recon,
        phase1_steps_per_epoch=args.phase1_steps_per_epoch,
        phase1_batch_size=args.phase1_batch_size,
        phase1_only_stable=args.phase1_only_stable,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_learn_rate=args.lora_learn_rate,

        lambda_replicate_static=args.lambda_replicate_static,
        lambda_recon_static=args.lambda_recon_static,
        lambda_triplet=args.lambda_triplet,
        adapter_bottleneck=args.adapter_bottleneck,
        lambda_recon=args.lambda_recon,
        lambda_rep=args.lambda_rep,
        lambda_id=args.lambda_id,
        id_tau=args.id_tau,
        lambda_knn=args.lambda_knn,
        knn_k=args.knn_k,
        knn_tau=args.knn_tau,
        knn_entropy_thresh=args.knn_entropy_thresh,
        lambda_delta_sparse=args.lambda_delta_sparse,
        delta_sparse_huber=args.delta_sparse_huber,
        lambda_delta_norm=args.lambda_delta_norm,
        delta_norm_huber_delta=args.delta_norm_huber_delta,
        lambda_anchor_denoise=args.lambda_anchor_denoise,
        anchor_denoise_tau=args.anchor_denoise_tau,
        stable_frac=args.stable_frac,
        adapter_type=args.adapter_type,
        moe_n_experts=args.moe_n_experts,
        moe_top_k=args.moe_top_k,
        moe_router_hidden=args.moe_router_hidden,
        moe_router_temperature=args.moe_router_temperature,
        
        lambda_route_entropy=args.lambda_route_entropy,
        lambda_load_balance=args.lambda_load_balance,
        moe_temp_start=args.moe_temp_start,
        moe_temp_end=args.moe_temp_end,
        anchor_coverage_min=args.anchor_coverage_min,
        phase1_method=args.phase1_method,
        phase1_frac=args.phase1_frac,
        stable_protein_list_path=args.stable_protein_list_path,
        fail_fast_on_low_anchor=(not args.no_fail_fast_on_low_anchor),
    )


if __name__ == "__main__":
    main()
