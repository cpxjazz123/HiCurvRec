#!/usr/bin/env python3
"""Task #23 + #24 — 从 PM-RQ Phase 2/3 模型提取 SID 张量 (适配 TIGER 输入).

PM-RQ 模型只保存了 model.pt (含 codebook_state / cascade_state)，未保存 item SID。
本脚本 forward 全部 11924 items, 生成 (N, 3) [Phase 2] 或 (N, 9) [Phase 3 cascade] SID。
为兼容 TIGER semantic_id_path (需 num_hierarchies 列), pad 末尾加 0 列: (4, N) / (10, N)。

输出:
    products/task22_pm_rq/sid_phase2.pt        (4, 11924) int64
    products/task22_pm_rq/sid_phase3_cascade.pt (10, 11924) int64

启动:
    cd /home/wlia0047/ar57/wenyu/GeneRec
    python3 scripts/task23_extract_pmrq_sid.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from task22_pm_rq_phase1b_toy import (  # noqa: E402
    ProductManifoldCodebook,
    SimpleDecoder,
)
from task22_pm_rq_phase3_cascade import ProductManifoldCascade  # noqa: E402

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DEFAULT_EMB = REPO / "products/task99_mckg_rebuild/entity_embedding.pt"
DEFAULT_P2_MODEL = REPO / "products/task22_pm_rq/phase2_full/phase2_model.pt"
DEFAULT_P3_MODEL = REPO / "products/task22_pm_rq/phase3_cascade/phase3_model.pt"
OUT_DIR = REPO / "products/task22_pm_rq"


def normalize_subitem(sub_item: torch.Tensor) -> torch.Tensor:
    """per-subspace z-score normalize + hyperbolic clamp (与 Phase 2/3 training 一致)."""
    sub = sub_item.clone()
    for m in range(3):
        sub[m] = (sub[m] - sub[m].mean(0)) / sub[m].std(0).clamp(min=1e-5)
    sub[2] = sub[2] * 0.5
    norms = sub[2].norm(dim=-1, keepdim=True)
    sub[2] = sub[2] / torch.clamp(norms / 0.85, min=1.0)
    return sub


def extract_phase2_sid(
    emb_path: Path, model_path: Path, out_path: Path,
    K: int = 256, dim: int = 64, batch_size: int = 1024,
) -> torch.Tensor:
    """Phase 2 single-layer PM-RQ: 1 layer × 3 codes/item → pad to (4, N)."""
    print(f"[phase2] loading MCKG embedding from {emb_path}")
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    sub_item = normalize_subitem(emb["subspace_item"][:, :, :])  # (3, 11924, 64)
    M, N, D = sub_item.shape
    print(f"[phase2] sub_item shape={tuple(sub_item.shape)}, K={K}, dim={dim}")

    print(f"[phase2] loading codebook from {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    codebook = ProductManifoldCodebook(K=K, dim=dim)
    codebook.load_state_dict(ckpt["codebook_state"])
    codebook.eval()

    # Forward 全部 N items (按 batch 切分, 拼成 (N, 3))
    all_idx_s = torch.empty(N, dtype=torch.long)
    all_idx_e = torch.empty(N, dtype=torch.long)
    all_idx_h = torch.empty(N, dtype=torch.long)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = sub_item[:, start:end, :]  # (3, B, dim)
            idx_s, idx_e, idx_h, _, _, _ = codebook(batch)
            all_idx_s[start:end] = idx_s
            all_idx_e[start:end] = idx_e
            all_idx_h[start:end] = idx_h
    print(f"[phase2] forward done in {time.time() - t0:.2f}s")

    # Stack to (3, N) → pad to (4, N) with zeros dedup column
    sid = torch.stack([all_idx_s, all_idx_e, all_idx_h], dim=0)  # (3, N)
    dedup_pad = torch.zeros(1, N, dtype=torch.long)  # (1, N)
    sid_padded = torch.cat([sid, dedup_pad], dim=0)  # (4, N)
    assert sid_padded.shape == (4, N), f"unexpected shape {sid_padded.shape}"

    print(f"[phase2] SID shape={tuple(sid_padded.shape)}, dtype={sid_padded.dtype}, "
          f"min={sid_padded.min().item()}, max={sid_padded.max().item()}, "
          f"util: s={len(torch.unique(all_idx_s))}/{K} "
          f"e={len(torch.unique(all_idx_e))}/{K} "
          f"h={len(torch.unique(all_idx_h))}/{K}")
    print(f"[phase2] sid_unique_3tuple: "
          f"{len(torch.unique(sid[:, :].T, dim=0))}/{N}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sid_padded, out_path)
    print(f"[phase2] saved → {out_path}")
    return sid_padded


def extract_phase3_sid(
    emb_path: Path, model_path: Path, out_path: Path,
    K: int = 256, dim: int = 64, num_layers: int = 3,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Phase 3 cascade: 3 layers × 3 codes/item = 9 codes/item → pad to (10, N)."""
    print(f"[phase3] loading MCKG embedding from {emb_path}")
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    sub_item = normalize_subitem(emb["subspace_item"][:, :, :])  # (3, 11924, 64)
    M, N, D = sub_item.shape
    print(f"[phase3] sub_item shape={tuple(sub_item.shape)}, K={K}, dim={dim}, layers={num_layers}")

    print(f"[phase3] loading cascade from {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cascade = ProductManifoldCascade(num_layers=num_layers, K=K, dim=dim)
    cascade.load_state_dict(ckpt["cascade_state"])
    cascade.eval()

    # Forward 全部 N items batch-by-batch → 收集每层 idx_s/e/h → (3 layers × 3 codes, N)
    layer_indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for layer in range(num_layers):
        layer_indices.append((
            torch.empty(N, dtype=torch.long),
            torch.empty(N, dtype=torch.long),
            torch.empty(N, dtype=torch.long),
        ))

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = sub_item[:, start:end, :]  # (3, B, dim)
            all_indices, _, all_z = cascade(batch)
            for layer, (idx_s, idx_e, idx_h) in enumerate(all_indices):
                layer_indices[layer][0][start:end] = idx_s
                layer_indices[layer][1][start:end] = idx_e
                layer_indices[layer][2][start:end] = idx_h
    print(f"[phase3] forward done in {time.time() - t0:.2f}s")

    # Stack: 9 codes → (9, N), pad to (10, N)
    cols = []
    for layer in range(num_layers):
        idx_s, idx_e, idx_h = layer_indices[layer]
        cols.extend([idx_s, idx_e, idx_h])
    sid = torch.stack(cols, dim=0)  # (9, N)
    dedup_pad = torch.zeros(1, N, dtype=torch.long)
    sid_padded = torch.cat([sid, dedup_pad], dim=0)  # (10, N)
    assert sid_padded.shape == (10, N), f"unexpected shape {sid_padded.shape}"

    # 利用率 per layer
    util_lines = []
    for layer in range(num_layers):
        idx_s, idx_e, idx_h = layer_indices[layer]
        util_lines.append(
            f"L{layer}: s={len(torch.unique(idx_s))}/{K} "
            f"e={len(torch.unique(idx_e))}/{K} "
            f"h={len(torch.unique(idx_h))}/{K}"
        )
    print(f"[phase3] SID shape={tuple(sid_padded.shape)}, dtype={sid_padded.dtype}, "
          f"min={sid_padded.min().item()}, max={sid_padded.max().item()}")
    print("[phase3] util per layer: " + " | ".join(util_lines))
    print(f"[phase3] 9-tuple unique: "
          f"{len(torch.unique(sid[:9].T, dim=0))}/{N}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sid_padded, out_path)
    print(f"[phase3] saved → {out_path}")
    return sid_padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-path", type=str, default=str(DEFAULT_EMB))
    parser.add_argument("--p2-model", type=str, default=str(DEFAULT_P2_MODEL))
    parser.add_argument("--p3-model", type=str, default=str(DEFAULT_P3_MODEL))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    out_dir = Path(args.out_dir)

    if not args.skip_phase2:
        extract_phase2_sid(
            emb_path=Path(args.emb_path),
            model_path=Path(args.p2_model),
            out_path=out_dir / "sid_phase2.pt",
            K=args.K, dim=args.dim, batch_size=args.batch_size,
        )

    if not args.skip_phase3:
        extract_phase3_sid(
            emb_path=Path(args.emb_path),
            model_path=Path(args.p3_model),
            out_path=out_dir / "sid_phase3_cascade.pt",
            K=args.K, dim=args.dim, num_layers=args.num_layers,
            batch_size=args.batch_size,
        )

    print("\n[done] PM-RQ SID extraction complete.")


if __name__ == "__main__":
    main()
