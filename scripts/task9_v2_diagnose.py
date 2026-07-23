"""
task1_v2_diagnose.py — Task 15 v2：新层内部训练早期漂移曲线

关键改动（vs v1）：
  - 用 task11_v2 训练产出的密集 ckpt（每 200 步一个）
  - 测量对象：新层自己从 step 1 到 step 2000 的内部漂移曲线
  - 输出：完整的漂移曲线（不只是 max shift 一个数字）

输入：task11_v2 训练的密集 ckpt + LLM embedding
输出：
  - 新层自己的 ||r||_mean 曲线
  - 相邻 ckpt 间的 Wasserstein 距离曲线
  - "稳定点"检测：第一次 5% 以下的 step
"""

import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# 修复 GRID import
GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
from src.utils.custom_hydra_resolvers import (  # noqa: F401
    remove_chars_from_string,
    conditional_expression,
    extract_fields_from_list_of_dicts,
    create_map_from_list_of_dicts,
    math_eval,
    remove_item_from_list,
)


def parse_step_from_ckpt(name: str) -> int:
    """从 ckpt 文件名提取 step number（支持 'checkpoint_000_003000.ckpt' 和 'checkpoint_epoch=000_step=000700.ckpt'）"""
    m = re.search(r"step=(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"checkpoint_\d+_(\d+)\.ckpt", name)
    if m:
        return int(m.group(1))
    return -1


def load_rkmeans_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hp = ckpt["hyper_parameters"]
    codebooks = []
    i = 0
    while f"quantization_layer_list.{i}.centroids" in state:
        codebooks.append(state[f"quantization_layer_list.{i}.centroids"])
        i += 1
    return codebooks, hp, ckpt.get("current_layer", -1)


def quantize_residuals(embedding: torch.Tensor, codebooks: list, normalize: bool) -> List[np.ndarray]:
    residuals = []
    r = embedding.float().clone()
    for cb in codebooks:
        if normalize:
            r = torch.nn.functional.normalize(r, dim=-1)
        dist = torch.cdist(r.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(dim=1)
        q = cb[idx]
        r = r - q
        residuals.append(r.detach().cpu().numpy())
    return residuals


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    if len(a_sorted) != len(b_sorted):
        n = max(len(a_sorted), len(b_sorted))
        a_sorted = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
        b_sorted = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
    return float(np.mean(np.abs(a_sorted - b_sorted)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="目录包含多个 ckpt")
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--max_items", type=int, default=2000)
    parser.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/task11_v2_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding
    embedding = torch.load(args.embedding, weights_only=True).float()
    if args.max_items and embedding.shape[0] > args.max_items:
        torch.manual_seed(42)
        idx = torch.randperm(embedding.shape[0])[:args.max_items]
        embedding = embedding[idx]
    N, D = embedding.shape
    print(f"[load] embedding subsample shape={tuple(embedding.shape)}")

    # 找到所有 ckpt 并按 step 排序
    ckpt_paths = sorted(Path(args.ckpt_dir).glob("checkpoint_*.ckpt"), key=lambda p: parse_step_from_ckpt(p.name))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint_*.ckpt in {args.ckpt_dir}")
    print(f"[load] found {len(ckpt_paths)} ckpts, step range: {parse_step_from_ckpt(ckpt_paths[0].name)} to {parse_step_from_ckpt(ckpt_paths[-1].name)}")

    # 加载每个 ckpt 算残差
    per_step_data = []  # list of dict
    for cp in ckpt_paths:
        step = parse_step_from_ckpt(cp.name)
        codebooks, hp, current_layer = load_rkmeans_ckpt(str(cp))
        normalize = hp.get("normalize_residuals", False)
        residuals = quantize_residuals(embedding, codebooks, normalize=normalize)
        per_step_data.append({
            "step": step,
            "current_layer": int(current_layer),
            "n_layers": len(codebooks),
            "normalize": normalize,
            "residuals": residuals,
        })
        print(f"  step {step}: current_layer={current_layer}, n_layers={len(codebooks)}")

    # 核心分析：每个新层（current_layer=idx 第一次出现的 step）从它出现的 step 开始的内部漂移曲线
    # 关注"刚启动训练的新层"：第一次 current_layer 跳到该 idx 的 step
    print("\n[analyze] new-layer internal drift curves ...")
    layer_first_step = {}  # layer_idx -> 第一个 current_layer == layer_idx 的 step
    for d in per_step_data:
        if d["current_layer"] not in layer_first_step:
            layer_first_step[d["current_layer"]] = d["step"]

    new_layer_curves = {}  # layer_idx -> list of {step, norm_mean, drift_to_prev}
    for layer_idx, first_step in sorted(layer_first_step.items()):
        curve = []
        prev_r = None
        for d in per_step_data:
            if d["step"] < first_step:
                continue
            r = d["residuals"][layer_idx]  # 当前 ckpt 在该层的残差
            norm = np.linalg.norm(r, axis=1)
            entry = {
                "step": d["step"],
                "step_offset_from_first": d["step"] - first_step,
                "norm_mean": float(norm.mean()),
                "norm_std": float(norm.std()),
                "abs_mean": float(np.abs(r).mean()),
            }
            if prev_r is not None:
                # 漂移：相对前一个 ckpt
                prev_norm = np.linalg.norm(prev_r, axis=1)
                w_norm = wasserstein_1d(norm, prev_norm)
                mean_shift = abs(norm.mean() - prev_norm.mean()) / (prev_norm.mean() + 1e-12)
                entry["wasserstein_norm"] = w_norm
                entry["mean_norm_shift"] = mean_shift
            curve.append(entry)
            prev_r = r
        new_layer_curves[layer_idx] = curve
        print(f"  layer {layer_idx} curve: {len(curve)} ckpts, step {first_step} -> {curve[-1]['step']}")

    # 对每条曲线找"稳定点"（连续 3 个 ckpt 的 mean_norm_shift < 5%）
    summary = {}
    for layer_idx, curve in new_layer_curves.items():
        if len(curve) < 2:
            summary[f"layer_{layer_idx}"] = {"n_points": len(curve), "stable_step": None, "convergence_steps": None}
            continue
        stable_step = None
        convergence_steps = None
        for i in range(len(curve) - 2):
            shifts = [curve[j].get("mean_norm_shift", 0) for j in range(i + 1, min(i + 4, len(curve)))]
            # 取每个相对其前一个的 shift（第一个是相对 step 0）
            if all(s < 0.05 for s in shifts if s is not None):
                stable_step = curve[i]["step"]
                convergence_steps = curve[i]["step_offset_from_first"]
                break
        # 整体漂移：从 first 到 last
        first_norm = curve[0]["norm_mean"]
        last_norm = curve[-1]["norm_mean"]
        total_shift = abs(last_norm - first_norm) / (first_norm + 1e-12)
        summary[f"layer_{layer_idx}"] = {
            "n_points": len(curve),
            "first_step": curve[0]["step"],
            "last_step": curve[-1]["step"],
            "first_norm_mean": first_norm,
            "last_norm_mean": last_norm,
            "total_rel_shift": float(total_shift),
            "stable_step": stable_step,
            "convergence_steps": convergence_steps,
        }
        print(f"  layer {layer_idx}: norm {first_norm:.4f} -> {last_norm:.4f} (shift={total_shift*100:.2f}%), "
              f"stable at step {stable_step} (converged in {convergence_steps} steps)")

    # Save
    report = {
        "ckpt_dir": str(args.ckpt_dir),
        "embedding_path": args.embedding,
        "embedding_shape": [N, D],
        "n_ckpts": len(ckpt_paths),
        "step_range": [parse_step_from_ckpt(ckpt_paths[0].name), parse_step_from_ckpt(ckpt_paths[-1].name)],
        "layer_first_step": layer_first_step,
        "new_layer_curves": new_layer_curves,
        "summary": summary,
        "judgment": "",
    }

    # 判断
    judgments = []
    for layer_idx, s in summary.items():
        if s.get("stable_step") is None:
            judgments.append(f"  {layer_idx}: 训练 {s['n_points']} 步内未稳定（仍在漂移）")
        else:
            judgments.append(
                f"  {layer_idx}: 训练 {s['convergence_steps']} 步后稳定（mean_shift < 5% 持续 ≥3 步）"
            )
    report["judgment"] = "\n".join(judgments)

    with open(out_dir / "task1_v2_drift_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[report] saved {out_dir}/task1_v2_drift_report.json")
    print(f"\n=== Summary ===")
    for j in judgments:
        print(j)


if __name__ == "__main__":
    main()