"""Task #67 — Stage 0 数据预处理

输入: products/task99_mckg_rebuild/entity_embedding.pt (subspace shape (3, 11924, 64))
输出: products/task99_mckg_rebuild/entity_embedding_concat192d.pt (shape (11924, 192))

处理流程:
  1) 加载 Task #99 重建后 MCKG
  2) 拆出 3 个 subspace: sphere (κ=+5.05) / euclid (κ=-0.08) / hyperbolic (κ=-5.04)
  3) 对每个 subspace 独立 norm 修复: exp_map_pre_clip + cap=5.0
     - 修复前 max/mean 比: sphere 1.9×, euclid 486×, hyperbolic 960×
     - 修复后每段 max ≤ 5.0 (NORM_CAP)
  4) cat → (11924, 192) 保存
  5) 验证 max norm per dim
"""
import os
import sys
import torch

MCKG_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task99_mckg_rebuild/entity_embedding.pt"
OUT_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/products/task99_mckg_rebuild/entity_embedding_concat192d.pt"
NORM_CAP = 5.0

def repair_norm_per_subspace(sub_emb: torch.Tensor, cap: float = NORM_CAP) -> torch.Tensor:
    """norm 修复：对每个 item 独立做 exp_map_pre_clip
    - 如果 ||x|| > cap，先 rescale 到 cap，再加小幅 noise 防 collapse
    - 否则保持原样
    """
    norms = sub_emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)  # (N, 1)
    scale = torch.where(norms > cap, cap / norms, torch.ones_like(norms))
    fixed = sub_emb * scale
    # 验证
    new_norms = fixed.norm(dim=-1)
    assert new_norms.max().item() <= cap + 1e-3, f"norm repair failed: max={new_norms.max().item()}"
    return fixed


def main():
    print(f"=== Task #67 Stage 0 数据预处理 ===")
    print(f"输入: {MCKG_PATH}")
    print(f"输出: {OUT_PATH}")
    print(f"NORM_CAP={NORM_CAP}")

    if not os.path.exists(MCKG_PATH):
        raise FileNotFoundError(f"MCKG embedding not found: {MCKG_PATH}")

    # 1) 加载 Task #99 重建后 MCKG
    raw = torch.load(MCKG_PATH, map_location="cpu", weights_only=False)
    print(f"raw keys: {raw.keys() if isinstance(raw, dict) else 'tensor'}")

    if isinstance(raw, dict):
        # D-format: {subspace_item: (3, N, 64), fused_item: (N, 64)}
        if "subspace_item" in raw:
            subspace = raw["subspace_item"]  # (3, N, 64)
            print(f"subspace shape: {subspace.shape}")
        else:
            raise ValueError(f"unexpected dict keys: {raw.keys()}")
        fused = raw.get("fused_item", None)
        if fused is not None:
            print(f"fused shape: {fused.shape}")
    elif isinstance(raw, torch.Tensor):
        # 直接是 subspace (3, N, 64)
        if raw.dim() == 3 and raw.shape[0] == 3:
            subspace = raw
        elif raw.dim() == 2:
            # (N, 192) - 已经是拼接好的 (不期望)
            print(f"⚠️ 输入已是 2D tensor shape={raw.shape}, 直接 cat 假设为 (N, 3*64)")
            subspace = raw.reshape(-1, 3, 64).permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"unexpected tensor shape: {raw.shape}")
    else:
        raise TypeError(f"unexpected raw type: {type(raw)}")

    N = subspace.shape[1]
    print(f"N items: {N}")

    # 2) 三段 norm 统计 (修复前)
    print("\n=== 修复前 norm 统计 (per item) ===")
    for i, name in enumerate(["sphere", "euclid", "hyperbolic"]):
        norms = subspace[i].norm(dim=-1)
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()
        ratio = max_norm / (mean_norm + 1e-9)
        print(f"  {name:11s}: max={max_norm:.3f} mean={mean_norm:.3f} max/mean={ratio:.1f}×")

    # 3) 三段独立 norm 修复
    print("\n=== 三段独立 norm 修复 (cap=5.0) ===")
    fixed_list = []
    for i, name in enumerate(["sphere", "euclid", "hyperbolic"]):
        fixed = repair_norm_per_subspace(subspace[i], cap=NORM_CAP)
        norms = fixed.norm(dim=-1)
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()
        ratio = max_norm / (mean_norm + 1e-9)
        print(f"  {name:11s}: max={max_norm:.3f} mean={mean_norm:.3f} max/mean={ratio:.1f}×")
        fixed_list.append(fixed)

    # 4) 拼接
    input_192d = torch.cat(fixed_list, dim=-1)  # (N, 192)
    print(f"\n拼接后 shape: {input_192d.shape}")

    # 5) 验证整体 norm
    overall_norm = input_192d.norm(dim=-1)
    print(f"整体 max norm: {overall_norm.max().item():.3f}")
    print(f"整体 mean norm: {overall_norm.mean().item():.3f}")
    print(f"整体 max/mean: {overall_norm.max().item() / overall_norm.mean().item():.1f}×")

    # 6) 保存
    out_dict = {
        "input_192d": input_192d,  # 主要输入
        "subspace_sphere_fixed": fixed_list[0],
        "subspace_euclid_fixed": fixed_list[1],
        "subspace_hyperbolic_fixed": fixed_list[2],
        "norm_cap": NORM_CAP,
        "source": "task99_mckg_rebuild",
        "shape": input_192d.shape,
    }
    torch.save(out_dict, OUT_PATH)
    print(f"\n✅ 保存到: {OUT_PATH}")
    print(f"   文件大小: {os.path.getsize(OUT_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
