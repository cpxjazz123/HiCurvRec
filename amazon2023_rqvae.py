#!/usr/bin/env python3
"""Train the three Amazon-2023 RQ-VAE variants and export TIGER SIDs.

This runner deliberately imports the implementations from the three source
projects instead of reimplementing their quantizers:

* ``hg``: HG-Rec hyperbolic residual VQ;
* ``curvature``: curvature_base's M2/M3/cyclic-curvature RQ-VAE;
* ``letter``: LETTER's collaborative-signal and diversity-regularized RQ-VAE.

The input order is the Amazon2023 item-id order used by TIGER.  Every output
is a fresh raw SID JSON (the TIGER loader adds one to each component).  A
fourth collision-extension component is appended in the same spirit as the
existing RecBole export, so every item has a unique complete SID tuple.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
HG_ROOT = ROOT / "HG-Rec"
CURVATURE_ROOT = ROOT / "curvature_base"
LETTER_ROOT = ROOT / "LETTER" / "RQ-VAE"
DEFAULT_EMB = ROOT / "TIGER" / "dataset" / "Amazon_2023_Instruments" / "item_emb.parquet"
DEFAULT_INTERACTIONS = ROOT / "TIGER" / "dataset" / "Amazon_2023_Instruments" / "train_recbole.parquet"
SID_ROOT = ROOT / "SIDS"
RUN_ROOT = SID_ROOT / "rqvae_runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        # Amazon2023 RQ-VAE is dominated by Linear/GEMM kernels.  A100 has
        # dedicated TF32/BF16 paths for these; there are no convolutions in
        # this runner, but keep the cuDNN switch enabled for completeness.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device, enabled: bool):
    """Use the A100 BF16 path while keeping model parameters in FP32."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_adamw(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
    use_fused: bool,
) -> torch.optim.Optimizer:
    """Create the fastest available AdamW without making CPU runs fail."""
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if use_fused and device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError) as exc:
            print(f"[runner] fused AdamW unavailable ({exc}); using standard AdamW", flush=True)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def maybe_compile_parts(
    model: torch.nn.Module,
    device: torch.device,
    enabled: bool,
    mode: str,
) -> torch.nn.Module:
    """Compile only stable MLP blocks.

    The three source quantizers contain Python-side codebook loops and, for
    LETTER, changing per-epoch cluster-label containers.  Compiling the whole
    model would trigger graph breaks/recompiles.  Encoder/decoder blocks are
    stable across variants and give Inductor a useful fusion target without
    changing the source quantizer semantics.
    """
    if not enabled or device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    # This environment has a standalone functorch package whose import list
    # still expects draw_joint_graph, while the bundled Torch 2.13 partitioner
    # no longer exports that legacy name.  Inductor does not call the legacy
    # helper for these modules; aliasing it to the bundled graph drawer lets
    # torch.compile initialize without changing any model math.
    try:
        import torch._functorch.partitioners as partitioners

        if not hasattr(partitioners, "draw_joint_graph") and hasattr(
            partitioners, "draw_graph"
        ):
            partitioners.draw_joint_graph = partitioners.draw_graph
    except Exception as exc:
        print(f"[runner] functorch compatibility shim skipped: {exc}", flush=True)
    for name in ("encoder", "decoder"):
        block = getattr(model, name, None)
        if block is None:
            continue
        try:
            compiled = torch.compile(block, mode=mode, dynamic=True)
            setattr(model, name, compiled)
            print(f"[runner] torch.compile enabled for {name} (mode={mode})", flush=True)
        except Exception as exc:
            # A source model should remain runnable if a particular Torch/
            # Inductor combination cannot compile one of its blocks.
            print(f"[runner] torch.compile skipped for {name}: {type(exc).__name__}: {exc}", flush=True)
    return model


def restore_uncompiled_parts(model: torch.nn.Module) -> torch.nn.Module:
    """Restore source MLP modules before FP32 SID export/checkpoint reload."""
    for name in ("encoder", "decoder"):
        block = getattr(model, name, None)
        original = getattr(block, "_orig_mod", None)
        if original is not None:
            setattr(model, name, original)
            print(f"[runner] restored uncompiled {name} for exact FP32 export", flush=True)
    return model


def clip_gradients(model: torch.nn.Module) -> None:
    """Use foreach clipping on CUDA to avoid one kernel per parameter tensor."""
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    except (TypeError, RuntimeError):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


def load_item_embeddings(path: Path) -> torch.Tensor:
    frame = pd.read_parquet(path)
    if "embedding" not in frame.columns:
        raise ValueError(f"{path} does not contain an embedding column")
    values = np.stack(frame["embedding"].to_numpy()).astype(np.float32, copy=False)
    if values.ndim != 2 or values.shape[1] != 768:
        raise ValueError(f"expected [n_items, 768] embeddings, got {values.shape}")
    if "ItemID" in frame.columns:
        item_ids = frame["ItemID"].astype(int).to_numpy()
        expected = np.arange(len(frame), dtype=np.int64)
        if not np.array_equal(item_ids, expected):
            raise ValueError("item_emb.parquet is not in zero-based item-id order")
    return torch.from_numpy(values)


def batches_on_device(
    x: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    *,
    drop_last: bool = False,
) -> Iterable[torch.Tensor]:
    n = x.shape[0]
    if shuffle:
        order = torch.randperm(n, device=device)
    else:
        order = torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        if drop_last and stop - start < batch_size:
            break
        yield x.index_select(0, order[start:stop])


def raw_codes_to_unique(codes: np.ndarray, codebook_sizes: Sequence[int]) -> np.ndarray:
    """Append the same collision ordinal used by the source exporters.

    The extra component is zero for raw-unique tuples.  For a duplicate raw
    tuple, items receive deterministic ordinals 0, 1, ... in item order.  It
    is deliberately not offset by the sum of the semantic codebook sizes: the
    source exporters use one common token namespace and do the same.
    """
    if codes.ndim != 2 or codes.shape[1] != len(codebook_sizes):
        raise ValueError(f"unexpected code shape {codes.shape} for {codebook_sizes}")
    del codebook_sizes  # Retained for call-site/config compatibility.
    normalized = codes.astype(np.int64, copy=False)
    _, inverse, counts = np.unique(
        normalized, axis=0, return_inverse=True, return_counts=True
    )
    out = np.concatenate(
        [normalized, np.zeros((normalized.shape[0], 1), dtype=np.int64)], axis=1
    )
    for group_id, count in enumerate(counts.tolist()):
        if count > 1:
            positions = np.flatnonzero(inverse == group_id)
            out[positions, -1] = np.arange(count, dtype=np.int64)
    if len(np.unique(out, axis=0)) != len(out):
        raise RuntimeError("collision extension failed to make SIDs unique")
    return out


def fresh_run_dir(variant: str, run_tag: str) -> Path:
    """Create a never-before-used run directory; never load an old run."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = RUN_ROOT / f"{variant}_{run_tag}_{stamp}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def sid_output_path(variant: str, suffix: str) -> Path:
    names = {
        "hg": "amazon2023_hg_rec_sids",
        "curvature": "amazon2023_curvature_sids",
        "letter": "amazon2023_letter_sids",
    }
    return SID_ROOT / f"{names[variant]}_{suffix}.json"


def write_sid_json(path: Path, codes: np.ndarray, variant: str, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(i): [int(v) for v in row] for i, row in enumerate(codes)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    meta_path = path.with_suffix(".meta.json")
    meta = dict(metadata)
    meta.update(
        {
            "variant": variant,
            "num_items": int(codes.shape[0]),
            "sid_width": int(codes.shape[1]),
            "unique_sids": int(np.unique(codes, axis=0).shape[0]),
            "max_raw_token": int(codes.max()),
            "sid_json": str(path),
        }
    )
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


@torch.no_grad()
def collision_rate(codes: np.ndarray) -> float:
    return 1.0 - (float(np.unique(codes, axis=0).shape[0]) / float(codes.shape[0]))


def save_state(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, path, pickle_protocol=4)


@torch.no_grad()
def collect_hg_raw_codes(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    use_sk: bool,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for batch in batches_on_device(x, batch_size, False, device):
        chunks.append(model.get_indices(batch, use_sk=use_sk).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)


def collision_groups(codes: np.ndarray) -> List[np.ndarray]:
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for item_id, row in enumerate(codes.astype(np.int64, copy=False)):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    return [np.asarray(items, dtype=np.int64) for items in groups.values() if len(items) > 1]


@torch.no_grad()
def resolve_hg_collisions(
    model: torch.nn.Module,
    x: torch.Tensor,
    codes: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_rounds: int = 30,
) -> Tuple[np.ndarray, int]:
    """Reproduce HG-Rec's collision re-assignment stage before the ordinal."""
    resolved = codes.copy()
    rounds = 0
    model.eval()
    # HG-Rec's Amazon configuration uses sk_eps=[0, 0, 0]. In that
    # configuration use_sk=True and use_sk=False are mathematically identical,
    # so the source collision-reassignment loop is a no-op. Avoid launching
    # thousands of identical tiny GPU kernels for every collision group.
    layers = getattr(getattr(model, "hrq", None), "vq_layers", ())
    if not any(float(getattr(layer, "sk_eps", 0.0)) > 0.0 for layer in layers):
        return resolved, rounds
    for round_id in range(max_rounds):
        groups = collision_groups(resolved)
        if not groups:
            break
        before = resolved.copy()
        for group in groups:
            item_ids = torch.as_tensor(group, dtype=torch.long, device=device)
            replacement = model.get_indices(
                x.index_select(0, item_ids), use_sk=True
            ).cpu().numpy()
            resolved[group] = replacement
        rounds = round_id + 1
        if np.array_equal(before, resolved):
            break
    return resolved, rounds


def train_hg(args: argparse.Namespace, x_cpu: torch.Tensor, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    sys.path.insert(0, str(HG_ROOT))
    try:
        from model.hrqvae import HRQVAE
    finally:
        # Keep the source path available while the imported package is alive,
        # but do not let it affect the other variants in a later invocation.
        sys.path.pop(0)

    model = HRQVAE(
        in_dim=768,
        num_emb_list=[64, 128, 256],
        e_dim=32,
        layers=[512, 256, 128, 64],
        dropout_prob=0.0,
        bn=False,
        loss_type="poincare",
        quant_loss_weight=1.0,
        beta=1.0,
        kmeans_init=True,
        kmeans_iters=args.kmeans_iters,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=50,
    ).to(device)

    x = x_cpu.to(device, non_blocking=True)
    # The source initializes each codebook on the first training mini-batch.
    # Do not replace that with a full-data KMeans pass: it changes the source
    # initialization and also changes the residual distribution seen by L1/L2.
    model = maybe_compile_parts(model, device, args.use_compile, args.compile_mode)
    optimizer = make_adamw(
        model,
        lr=args.lr,
        weight_decay=args.hg_weight_decay,
        device=device,
        use_fused=args.use_fused_optimizer,
    )
    steps_per_epoch = max(1, x.shape[0] // args.batch_size)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.hg_warmup_epochs * steps_per_epoch,
        num_training_steps=args.epochs * steps_per_epoch,
    )
    best_loss = float("inf")
    best_collision = float("inf")
    best_loss_path = args.run_dir / "best_loss_model.pth"
    best_collision_path = args.run_dir / "best_collision_model.pth"
    log_path = args.run_dir / "train.jsonl"

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        total_tensor = torch.zeros((), device=device)
        n_batches = 0
        for batch in batches_on_device(
            x, args.batch_size, True, device, drop_last=True
        ):
            optimizer.zero_grad(set_to_none=True)
            # HG's source KMeans initializer converts the first latent batch
            # to a NumPy array.  Keep only that first call in FP32; all later
            # updates use the A100 BF16 path safely.
            hg_layers = getattr(getattr(model, "hrq", None), "vq_layers", ())
            amp_now = args.use_amp and all(
                bool(getattr(layer, "initted", True)) for layer in hg_layers
            )
            with autocast_context(device, amp_now):
                out, rq_loss, _ = model(batch, use_sk=True)
                loss, recon = model.compute_loss(out, rq_loss, xs=batch)
            loss.backward()
            clip_gradients(model)
            optimizer.step()
            scheduler.step()
            total_tensor = total_tensor + loss.detach()
            n_batches += 1
        loss_sum = float(total_tensor.item())
        mean_loss = loss_sum / max(n_batches, 1)
        improved = loss_sum < best_loss
        if improved:
            best_loss = loss_sum
            save_state(best_loss_path, model)

        collision = None
        if epoch % args.hg_eval_interval == 0 or epoch == args.epochs:
            collision_codes = collect_hg_raw_codes(
                model, x, args.export_batch_size, device, use_sk=True
            )
            collision = collision_rate(collision_codes)
            if collision < best_collision:
                best_collision = collision
                save_state(best_collision_path, model)
        record = {
            "epoch": epoch,
            "loss": mean_loss,
            "loss_sum": loss_sum,
            "best_loss": best_loss,
            "collision": collision,
            "best_collision": best_collision,
            "seconds": time.time() - started,
            "improved": improved,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if epoch == 1 or epoch % args.log_interval == 0 or improved or collision is not None:
            print(
                f"[hg] epoch={epoch}/{args.epochs} loss={mean_loss:.6f} "
                f"best={best_loss:.6f} collision={collision} "
                f"best_collision={best_collision:.6f} time={record['seconds']:.2f}s",
                flush=True,
            )

    selected_path = best_collision_path if best_collision_path.exists() else best_loss_path
    model.load_state_dict(torch.load(selected_path, map_location=device, weights_only=False))
    model = restore_uncompiled_parts(model)
    raw = collect_hg_raw_codes(
        model, x, args.export_batch_size, device, use_sk=False
    )
    raw, collision_rounds = resolve_hg_collisions(
        model, x, raw, args.export_batch_size, device
    )
    full = raw_codes_to_unique(raw, [64, 128, 256])
    metadata = {
        "source": str(HG_ROOT),
        "input_embeddings": str(args.embedding_path),
        "training_epochs": int(args.epochs),
        "best_train_loss_sum": best_loss,
        "best_collision_rate": best_collision,
        "raw_collision_rate": collision_rate(raw),
        "collision_resolution_rounds": int(collision_rounds),
        "best_loss_model": str(best_loss_path),
        "best_collision_model": str(selected_path),
        "codebook_sizes": [64, 128, 256, 1],
    }
    output_path = sid_output_path("hg", args.output_suffix)
    write_sid_json(output_path, full, "hg", metadata)
    print(f"[hg] raw collision={metadata['raw_collision_rate']:.6f}; exported {full.shape}", flush=True)
    return model, metadata


def build_curvature_model():
    sys.path.insert(0, str(CURVATURE_ROOT))
    from modules.rqvae import RqVae
    from modules.quantize import QuantizeDistance, QuantizeForwardMode

    model = RqVae(
        input_dim=768,
        embed_dim=32,
        hidden_dims=[512, 256, 128],
        codebook_size=256,
        codebook_kmeans_init=True,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=3,
        commitment_weight=1.0,
        n_cat_features=0,
        gate_M2_intrinsic=True,
        gate_M3_transport=True,
        hyperbolic_distance=True,
        sk_eps=0.05,
        sk_iters=3,
        distance_mode=QuantizeDistance.L2,
        rbf_bandwidth=1.0,
        mahalanobis_init_var=1.0,
        prefix_router_layers=None,
        margin_reg_weight=0.0,
        spread_loss_weight=0.0,
        anisotropy_loss_weight=0.0,
        use_anisotropy_reg=False,
        use_tcu=False,
        use_mcdq=False,
        use_scs=False,
        use_fixed_curvature=True,
        c_fixed=1.0,
        use_curriculum_curvature=False,
        use_cyclic_curvature=True,
        c_cyclic_min=0.3,
        c_cyclic_max=1.0,
        c_cyclic_period=50_000,
        use_geodesic_midpoint_commit=True,
        midpoint_layer_mask=[True, False, False],
        use_mobius_gyrovector=False,
    )
    return model


def curvature_loss(model: torch.nn.Module, batch: torch.Tensor, temperature: float):
    """Exact reconstruction + quantization objective without O(B^2) diagnostics."""
    quantized = model.get_semantic_ids(batch, gumbel_t=temperature)
    x_hat = model.decode(quantized.embeddings.sum(axis=-1))
    reconstruction = model.reconstruction_loss(x_hat, batch).mean()
    loss = reconstruction + quantized.quantize_loss.mean()
    return loss, reconstruction


def train_curvature(args: argparse.Namespace, x_cpu: torch.Tensor, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    model = build_curvature_model().to(device)
    x = x_cpu.to(device, non_blocking=True)
    model.train()
    # Match curvature_base initialization: initialize from a shuffled training
    # mini-batch, not from a full-data KMeans pass.
    with torch.no_grad():
        init_order = torch.randperm(x.shape[0], device=device)
        init_batch = x.index_select(
            0, init_order[: min(args.batch_size, x.shape[0])]
        )
        model.get_semantic_ids(init_batch, gumbel_t=args.temperature)

    model = maybe_compile_parts(model, device, args.use_compile, args.compile_mode)
    optimizer = make_adamw(
        model,
        lr=args.lr,
        weight_decay=args.curvature_weight_decay,
        device=device,
        use_fused=args.use_fused_optimizer,
    )
    best_loss = float("inf")
    best_step = 0
    best_path = args.run_dir / "best_loss_model.pth"
    final_path = args.run_dir / "final_model.pth"
    log_path = args.run_dir / "train.jsonl"

    # curvature_base increments its ``global_step`` once per rank-local batch
    # and then all-reduces it across the four source ranks.  On one GPU, using
    # 100,000 local updates would therefore overrun the source sample budget.
    # Count source-equivalent samples instead and derive the schedule step from
    # that count.  This keeps the 100k global-step protocol meaningful when the
    # available world size and batch size change.
    source_batch_size = int(args.curvature_source_batch_size)
    target_samples = (
        args.curvature_max_steps * source_batch_size // args.batch_size
    ) * args.batch_size
    if target_samples <= 0:
        raise ValueError("curvature target sample budget is smaller than one batch")
    samples_seen = 0
    optimizer_steps = 0
    effective_global_step = 0
    epoch = 0
    while samples_seen < target_samples:
        epoch += 1
        started = time.time()
        model.train()
        total_tensor = torch.zeros((), device=device)
        n_batches = 0
        for batch in batches_on_device(
            x, args.batch_size, True, device, drop_last=True
        ):
            if samples_seen >= target_samples:
                break
            optimizer.zero_grad(set_to_none=True)
            model.set_curriculum_step(effective_global_step)
            with autocast_context(device, args.use_amp):
                loss, reconstruction = curvature_loss(model, batch, args.temperature)
            loss.backward()
            clip_gradients(model)
            optimizer.step()
            optimizer_steps += 1
            samples_seen += int(batch.shape[0])
            effective_global_step = min(
                args.curvature_max_steps,
                int(round(samples_seen / source_batch_size)),
            )
            total_tensor = total_tensor + loss.detach()
            n_batches += 1
        mean_loss = float(total_tensor.item()) / max(n_batches, 1)
        improved = mean_loss < best_loss
        if improved:
            best_loss = mean_loss
            best_step = effective_global_step
            save_state(best_path, model)
        record = {
            "epoch": epoch,
            "global_step": effective_global_step,
            "optimizer_steps": optimizer_steps,
            "samples_seen": samples_seen,
            "loss": mean_loss,
            "best_loss": best_loss,
            "target_steps": int(args.curvature_max_steps),
            "target_samples": int(target_samples),
            "seconds": time.time() - started,
            "improved": improved,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if epoch == 1 or epoch % args.log_interval == 0 or improved:
            print(
                f"[curvature] epoch={epoch} step={effective_global_step}/{args.curvature_max_steps} "
                f"loss={mean_loss:.6f} best={best_loss:.6f} "
                f"time={record['seconds']:.2f}s",
                flush=True,
            )

    # curvature_base exports the final fixed-step state. Keep the best-loss
    # checkpoint for auditability, but do not substitute it for the source's
    # final global-step state.
    save_state(final_path, model)
    model.load_state_dict(torch.load(final_path, map_location=device, weights_only=False))
    model.set_curriculum_step(effective_global_step)
    model = restore_uncompiled_parts(model)
    model.eval()
    code_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in batches_on_device(x, args.export_batch_size, False, device):
            output = model.get_semantic_ids(batch, gumbel_t=0.001)
            # ``RqVae.get_semantic_ids`` already returns ``sem_ids`` as
            # [batch, n_layers]; do not transpose it a second time.
            code_chunks.append(output.sem_ids.cpu().numpy())
    raw = np.concatenate(code_chunks, axis=0).astype(np.int64, copy=False)
    full = raw_codes_to_unique(raw, [256, 256, 256])
    metadata = {
        "source": str(CURVATURE_ROOT),
        "mechanism": "v318 M2 intrinsic + M3 transport + cyclic curvature + L0 geodesic midpoint",
        "input_embeddings": str(args.embedding_path),
        "training_epochs": int(epoch),
        "global_steps": int(effective_global_step),
        "optimizer_steps": int(optimizer_steps),
        "samples_seen": int(samples_seen),
        "target_global_steps": int(args.curvature_max_steps),
        "target_samples": int(target_samples),
        "best_train_loss": best_loss,
        "raw_collision_rate": collision_rate(raw),
        "best_loss_step": int(best_step),
        "best_loss_model": str(best_path),
        "final_model": str(final_path),
        "codebook_sizes": [256, 256, 256, 1],
    }
    output_path = sid_output_path("curvature", args.output_suffix)
    write_sid_json(output_path, full, "curvature", metadata)
    print(f"[curvature] raw collision={metadata['raw_collision_rate']:.6f}; exported {full.shape}", flush=True)
    return model, metadata


def build_cf_embedding(interactions_path: Path, n_items: int, output_path: Path, seed: int) -> np.ndarray:
    """Build a fresh 32-D collaborative embedding from train-only sequences."""
    if output_path.exists():
        loaded = torch.load(output_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, torch.Tensor):
            cf = loaded.detach().cpu().numpy()
        else:
            cf = np.asarray(loaded)
        if cf.shape == (n_items, 32):
            return cf.astype(np.float32, copy=False)

    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD

    frame = pd.read_parquet(interactions_path)
    rows: List[int] = []
    cols: List[int] = []
    for row in frame.itertuples(index=False):
        target = int(row.target)
        history = row.history.tolist() if isinstance(row.history, np.ndarray) else list(row.history)
        for item in history:
            item = int(item)
            if 0 <= item < n_items and 0 <= target < n_items:
                rows.extend((item, target))
                cols.extend((target, item))
    values = np.ones(len(rows), dtype=np.float32)
    cooc = coo_matrix((values, (rows, cols)), shape=(n_items, n_items)).tocsr()
    cooc.sum_duplicates()
    # Log weighting prevents very frequent items from dominating the CF loss.
    cooc.data = np.log1p(cooc.data)
    svd = TruncatedSVD(n_components=32, n_iter=7, random_state=seed)
    cf = svd.fit_transform(cooc).astype(np.float32, copy=False)
    norms = np.linalg.norm(cf, axis=1, keepdims=True)
    cf = cf / np.maximum(norms, 1e-6)
    cf *= 0.8
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(cf), output_path, pickle_protocol=4)
    print(f"[letter] built CF embedding {cf.shape} from {len(rows)//2} sequence pairs", flush=True)
    return cf


def letter_labels(
    model: torch.nn.Module, n_clusters: int
) -> Dict[str, List[int]]:
    from sklearn.cluster import KMeans

    labels: Dict[str, List[int]] = {}
    for idx, layer in enumerate(model.rq.vq_layers):
        weights = layer.embedding.weight.detach().cpu().numpy()
        cluster = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=10,
            random_state=42,
        ).fit(weights)
        labels[str(idx)] = cluster.labels_.tolist()
    return labels


@torch.no_grad()
def collect_letter_raw_codes(model: torch.nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    """Run LETTER's eval-time residual nearest-neighbor path without loss work."""
    chunks: List[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        residual = model.encoder(x[start : start + batch_size])
        level_ids = []
        for layer in model.rq.vq_layers:
            weights = layer.embedding.weight
            distances = (
                (residual ** 2).sum(dim=1, keepdim=True)
                + (weights ** 2).sum(dim=1, keepdim=True).t()
                - 2 * residual @ weights.t()
            )
            ids = distances.argmin(dim=-1)
            level_ids.append(ids)
            residual = residual - weights.index_select(0, ids)
        chunks.append(torch.stack(level_ids, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.int64, copy=False)


def fast_letter_diversity_loss_main_entry(
    quantizer: torch.nn.Module,
    x: torch.Tensor,
    x_q: torch.Tensor,
    indices: torch.Tensor,
    labels: List[int],
) -> torch.Tensor:
    """Vectorize LETTER's random same-cluster positive sampling.

    The source samples uniformly from the current codeword's cluster and
    retries when it samples the codeword itself.  A per-codeword candidate
    table gives the same distribution without a Python loop over the batch;
    singleton clusters retain the source self fallback.
    """
    label_key = tuple(int(v) for v in labels)
    cached_key = getattr(quantizer, "_fast_diversity_label_key", None)
    if (
        cached_key != label_key
        or getattr(quantizer, "_fast_diversity_candidates", None) is None
    ):
        members: Dict[int, List[int]] = {}
        for codeword, cluster in enumerate(label_key):
            members.setdefault(cluster, []).append(codeword)
        candidate_lists = []
        for codeword, cluster in enumerate(label_key):
            candidates = [
                member for member in members[cluster] if member != codeword
            ]
            if not candidates:
                candidates = [codeword]
            candidate_lists.append(candidates)
        max_candidates = max(len(row) for row in candidate_lists)
        candidate_table = torch.empty(
            (len(candidate_lists), max_candidates), dtype=torch.long
        )
        candidate_counts = torch.empty(len(candidate_lists), dtype=torch.long)
        candidate_valid = torch.empty(len(candidate_lists), dtype=torch.bool)
        for codeword, candidates in enumerate(candidate_lists):
            candidate_counts[codeword] = len(candidates)
            candidate_valid[codeword] = len(members[label_key[codeword]]) > 1
            candidate_table[codeword, : len(candidates)] = torch.as_tensor(
                candidates, dtype=torch.long
            )
            if len(candidates) < max_candidates:
                candidate_table[codeword, len(candidates) :] = candidates[0]
        quantizer._fast_diversity_label_key = label_key
        quantizer._fast_diversity_candidates = candidate_table
        quantizer._fast_diversity_counts = candidate_counts
        quantizer._fast_diversity_valid = candidate_valid
        quantizer._fast_diversity_has_valid = bool(candidate_valid.any())
        quantizer._fast_diversity_candidates_device = None
        quantizer._fast_diversity_counts_device = None
        quantizer._fast_diversity_valid_device = None

    indices = indices.long().reshape(-1)
    # Move the tiny lookup tables once per label refresh, rather than once per
    # minibatch.  The previous .to(device) calls were cheap in bytes but
    # introduced an avoidable allocator/synchronization path on every layer.
    if quantizer._fast_diversity_candidates_device is None:
        quantizer._fast_diversity_candidates_device = quantizer._fast_diversity_candidates.to(
            indices.device, non_blocking=True
        )
        quantizer._fast_diversity_counts_device = quantizer._fast_diversity_counts.to(
            indices.device, non_blocking=True
        )
        quantizer._fast_diversity_valid_device = quantizer._fast_diversity_valid.to(
            indices.device, non_blocking=True
        )
    candidates = quantizer._fast_diversity_candidates_device
    counts = quantizer._fast_diversity_counts_device.index_select(0, indices)
    valid = quantizer._fast_diversity_valid_device.index_select(0, indices)
    offsets = (torch.rand(indices.shape, device=indices.device) * counts).long()
    positives = candidates.index_select(0, indices).gather(1, offsets[:, None]).squeeze(1)
    if not quantizer._fast_diversity_has_valid:
        # A singleton cluster has no valid same-cluster positive. Returning a
        # graph-connected zero is the mathematically defined objective and
        # avoids the source implementation's self-target/1e12-logit blow-up.
        return x_q.sum() * 0.0
    logits = x_q @ quantizer.embedding.weight.t()
    logits = logits.scatter(1, indices[:, None], -1e12)
    return F.cross_entropy(logits[valid], positives[valid])


def train_letter(args: argparse.Namespace, x_cpu: torch.Tensor, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    sys.path.insert(0, str(LETTER_ROOT))
    try:
        from models.rqvae import RQVAE
        from models.vq import VectorQuantizer
    finally:
        sys.path.pop(0)

    # Keep LETTER's diversity regularizer while using a vectorized sampler
    # with the same per-codeword distribution as the source implementation.
    VectorQuantizer.diversity_loss_main_entry = fast_letter_diversity_loss_main_entry

    cf_path = args.run_dir / "letter_amazon2023_cf_embedding.pt"
    cf = build_cf_embedding(Path(args.interactions_path), x_cpu.shape[0], cf_path, args.seed)
    model = RQVAE(
        in_dim=768,
        num_emb_list=[256, 256, 256],
        e_dim=32,
        layers=[512, 256, 128],
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        kmeans_init=True,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=[0.0, 0.0, 0.0],
        sk_iters=50,
        alpha=args.letter_alpha,
        beta=args.letter_beta,
        n_clusters=args.letter_n_clusters,
        sample_strategy="all",
        cf_embedding=cf,
    ).to(device)
    x = x_cpu.to(device, non_blocking=True)

    # Keep LETTER's full-data residual KMeans initialization.
    model.eval()
    with torch.no_grad():
        model.vq_initialization(x)

    model = maybe_compile_parts(model, device, args.use_compile, args.compile_mode)
    optimizer = make_adamw(
        model,
        lr=args.lr,
        weight_decay=args.letter_weight_decay,
        device=device,
        use_fused=args.use_fused_optimizer,
    )
    best_loss = float("inf")
    best_path = args.run_dir / "best_loss_model.pth"
    best_sid_path = args.run_dir / "best_collision_model.pth"
    best_collision = float("inf")
    log_path = args.run_dir / "train.jsonl"
    cf_device = torch.from_numpy(cf).to(device=device)

    labels = None
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        # Recomputing sklearn KMeans is a CPU synchronization point.  Refresh
        # it periodically; the diversity objective continues to use the
        # current cached partition between refreshes.
        if labels is None or (epoch - 1) % args.label_refresh == 0:
            labels = letter_labels(model, args.letter_n_clusters)
        total_tensor = torch.zeros((), device=device)
        n_batches = 0
        order = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], args.batch_size):
            if start + args.batch_size > x.shape[0]:
                break  # source LETTER DataLoader uses drop_last=True
            emb_idx = order[start : start + args.batch_size]
            batch = x.index_select(0, emb_idx)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.use_amp):
                out, rq_loss, _, dense_out = model(batch, labels)
                recon = F.mse_loss(out, batch, reduction="mean")
                cf_batch = cf_device.index_select(0, emb_idx).to(dtype=dense_out.dtype)
                cf_loss = model.CF_loss(dense_out, cf_batch)
                loss = recon + rq_loss + args.letter_alpha * cf_loss
            loss.backward()
            clip_gradients(model)
            optimizer.step()
            total_tensor = total_tensor + loss.detach()
            n_batches += 1
        loss_sum = float(total_tensor.item())
        mean_loss = loss_sum / max(n_batches, 1)
        improved = loss_sum < best_loss
        if improved:
            best_loss = loss_sum
            save_state(best_path, model)
        sid_collision = None
        if epoch % args.sid_eval_interval == 0 or epoch == args.epochs:
            model.eval()
            sid_probe = collect_letter_raw_codes(model, x, args.export_batch_size)
            sid_collision = collision_rate(sid_probe)
            if sid_collision < best_collision:
                best_collision = sid_collision
                save_state(best_sid_path, model)
            model.train()
        record = {
            "epoch": epoch,
            "loss": mean_loss,
            "loss_sum": loss_sum,
            "best_loss": best_loss,
            "raw_collision": sid_collision,
            "best_raw_collision": best_collision,
            "seconds": time.time() - started,
            "improved": improved,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if epoch == 1 or epoch % args.log_interval == 0 or improved or sid_collision is not None:
            print(
                f"[letter] epoch={epoch}/{args.epochs} loss={mean_loss:.6f} "
                f"best={best_loss:.6f} collision={sid_collision} "
                f"best_collision={best_collision:.6f} time={record['seconds']:.2f}s",
                flush=True,
            )

    # SID quality is governed by complete-code collisions, so use the best
    # collision checkpoint for export while retaining the independent loss
    # checkpoint for auditability.
    model.load_state_dict(torch.load(best_sid_path, map_location=device, weights_only=False))
    model = restore_uncompiled_parts(model)
    model.eval()
    labels = letter_labels(model, args.letter_n_clusters)
    code_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.export_batch_size):
            batch = x[start : start + args.export_batch_size]
            output = model.get_indices(batch, labels, use_sk=False)
            code_chunks.append(output.cpu().numpy())
    raw = np.concatenate(code_chunks, axis=0).astype(np.int64, copy=False)
    full = raw_codes_to_unique(raw, [256, 256, 256])
    metadata = {
        "source": str(LETTER_ROOT),
        "mechanism": "LETTER collaborative-signal contrastive alignment + codebook diversity loss",
        "input_embeddings": str(args.embedding_path),
        "cf_embedding": str(cf_path),
        "training_epochs": int(args.epochs),
        "best_train_loss_sum": best_loss,
        "best_collision_rate": best_collision,
        "raw_collision_rate": collision_rate(raw),
        "best_loss_model": str(best_path),
        "best_collision_model": str(best_sid_path),
        "codebook_sizes": [256, 256, 256, 1],
    }
    output_path = sid_output_path("letter", args.output_suffix)
    write_sid_json(output_path, full, "letter", metadata)
    print(f"[letter] raw collision={metadata['raw_collision_rate']:.6f}; exported {full.shape}", flush=True)
    return model, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["hg", "curvature", "letter"], required=True)
    parser.add_argument("--embedding_path", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--interactions_path", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Source default: HG/curvature=1000 epochs, LETTER=5000 epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Fast default: 16384; pass an explicit value to override it.",
    )
    parser.add_argument("--export_batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hg_weight_decay", type=float, default=0.0)
    parser.add_argument("--curvature_weight_decay", type=float, default=1e-4)
    parser.add_argument("--letter_weight_decay", type=float, default=1e-4)
    parser.add_argument("--kmeans_iters", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--sid_eval_interval", type=int, default=None)
    parser.add_argument("--hg_eval_interval", type=int, default=5)
    parser.add_argument("--hg_warmup_epochs", type=int, default=20)
    parser.add_argument("--curvature_max_steps", type=int, default=100_000)
    parser.add_argument(
        "--curvature_source_batch_size",
        type=int,
        default=640,
        help="Per-rank batch used to translate curvature_base's global-step budget.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--letter_alpha", type=float, default=0.1)
    parser.add_argument("--letter_beta", type=float, default=0.1)
    parser.add_argument("--letter_n_clusters", type=int, default=10)
    parser.add_argument(
        "--label_refresh",
        type=int,
        default=4,
        help="Refresh LETTER's CPU KMeans labels every N epochs (fast default: 4).",
    )
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BF16 autocast on CUDA (parameters remain FP32).",
    )
    parser.add_argument(
        "--use_compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile stable encoder/decoder MLP blocks with TorchInductor.",
    )
    parser.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="max-autotune",
        help="TorchInductor mode for compiled MLP blocks.",
    )
    parser.add_argument(
        "--use_fused_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW on CUDA when supported.",
    )
    parser.add_argument("--run_tag", type=str, default="fidelity_v2")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="fidelity_v2",
        help="Suffix for the newly exported SID JSON; old outputs are preserved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs is None:
        args.epochs = 5000 if args.variant == "letter" else 1000
    if args.batch_size is None:
        # A100-safe fast defaults.  LETTER's in-batch CF loss is O(B^2), so
        # 16384 is the measured throughput optimum; a full 24587-item batch
        # is allocatable but slower because the CF GEMM grows quadratically.
        args.batch_size = 16384
    if args.kmeans_iters is None:
        args.kmeans_iters = 1000 if args.variant == "hg" else 100
    if args.sid_eval_interval is None:
        args.sid_eval_interval = 500 if args.variant == "letter" else 5
    if args.weight_decay is not None:
        # Backward-compatible override for old launch commands.
        if args.variant == "hg":
            args.hg_weight_decay = args.weight_decay
        elif args.variant == "curvature":
            args.curvature_weight_decay = args.weight_decay
        else:
            args.letter_weight_decay = args.weight_decay
    args.hg_eval_interval = int(args.hg_eval_interval)
    if args.variant == "hg":
        args.hg_eval_interval = int(args.sid_eval_interval)
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.export_batch_size <= 0
        or args.curvature_max_steps <= 0
        or args.curvature_source_batch_size <= 0
    ):
        raise ValueError("epochs and batch sizes must be positive")
    if (
        args.label_refresh <= 0
        or args.sid_eval_interval <= 0
        or args.hg_eval_interval <= 0
        or args.hg_warmup_epochs < 0
        or args.letter_n_clusters <= 0
    ):
        raise ValueError("evaluation intervals and cluster settings must be positive")
    set_seed(args.seed)
    SID_ROOT.mkdir(parents=True, exist_ok=True)
    args.run_dir = fresh_run_dir(args.variant, args.run_tag)
    x_cpu = load_item_embeddings(args.embedding_path).contiguous()
    args.batch_size = min(args.batch_size, int(x_cpu.shape[0]))
    args.export_batch_size = min(args.export_batch_size, int(x_cpu.shape[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[runner] variant={args.variant} items={x_cpu.shape[0]} dim={x_cpu.shape[1]} "
        f"device={device} batch={args.batch_size} epochs={args.epochs} "
        f"run_dir={args.run_dir}",
        flush=True,
    )
    if args.variant == "hg":
        train_hg(args, x_cpu, device)
    elif args.variant == "curvature":
        train_curvature(args, x_cpu, device)
    else:
        train_letter(args, x_cpu, device)


if __name__ == "__main__":
    main()
