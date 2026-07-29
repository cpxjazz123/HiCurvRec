"""
Task #263 / Issue #17 Gate 2 — task253 Stage 1 ckpt direct measurement.

绕过 train_hrqvae.py (因 fit 函数体 `import glob, os` 行 322 把 module-level `os` 覆盖成本地变量,
                     step2 monitor 整段 try/except 全场失败, hrqvae.log 没一行 utilization 真打过).
用 task222 verifier 同样的算法: load ckpt -> eval mode -> 累积 indices -> per-layer unique count.

不重训 (Issue #17 §Gate 2 通过条件, 不申请 Stage 3 预算).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
HG_REC = REPO / "HG-Rec"
sys.path.insert(0, str(HG_REC))

from model.hrqvae import HRQVAE  # noqa: E402
from model.utils import EmbDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Stage 1 best_collision_model.pth")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_json", required=True)
    return parser.parse_args()


def load_model_from_ckpt(ckpt_path, device):
    """Load ckpt 跟 hrqvae.py namespace 一致."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"] if "args" in ckpt else ckpt["config"]
    # Namespace vs dict
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)

    # 关键字段 (跟 train_hrqvae.py 行 367-383 一致)
    # assignment_mode_list 可能是 csv 字符串 ("shared,shared,shared") 或 list.
    _aml = ckpt_args.get("assignment_mode_list", ["shared", "shared", "shared"])
    if isinstance(_aml, str):
        _aml = [s.strip() for s in _aml.split(",")]
    n_emb = len(ckpt_args["num_emb_list"])
    # 截断或补齐, 跟 HRQVAE.assert 一致.
    if len(_aml) != n_emb:
        if _aml and all(_aml[0] == x for x in _aml):
            _aml = [_aml[0]] * n_emb
        else:
            _aml = (_aml + ["shared"] * n_emb)[:n_emb]
    model = HRQVAE(
        in_dim=ckpt_args.get("in_dim", 768),
        num_emb_list=ckpt_args["num_emb_list"],
        e_dim=ckpt_args["e_dim"],
        layers=ckpt_args["layers"],
        loss_type=ckpt_args.get("loss_type", "poincare"),
        product_manifold=ckpt_args.get("product_manifold", False),
        angular_dim=ckpt_args.get("angular_dim", 4),
        radial_dim=ckpt_args.get("radial_dim", 32),
        beta=ckpt_args.get("beta", 0.5),
        # FIX 2026-07-29 (Task #273): bn 不能默认 False, task178 ckpt args.bn=True
        # 会让 verifier HRQVAE 重建 encoder/decoder 缺 BN layer → shape 不匹配 crash
        bn=ckpt_args.get("bn", False),
        assignment_mode_list=_aml,
        sk_eps=ckpt_args.get("sk_epsilons", ckpt_args.get("sk_eps", [0.0, 0.0, 0.0])),
    ).to(device)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # 已知 missing keys: radius/κ params (用户 2026-07-27 Phase 0B 后新增, 没在 task222 ep29 ckpt 里).
        known_new_param_keys = ["log_r", "hyp_mean", "hyp_scale"]
        known_new_missing = [k for k in missing if any(s in k for s in known_new_param_keys)]
        other_missing = [k for k in missing if k not in known_new_missing]
        if other_missing:
            raise RuntimeError(
                f"unexpected missing keys (not radius/κ params): {other_missing}"
            )
        print(f"[Task #263] ckpt 缺 {len(known_new_missing)} 个 radius/κ params, "
              f"用 model 默认初始化 (注意: utilization 测量对这部分参数不敏感).")

    model.eval()
    return model, ckpt_args


def measure_utilization(model, data_loader, device):
    """Pre-revive per-layer utilization, 跟 hrqvae_trainer.py 行 250-263 算法一致."""
    all_indices_per_layer = []
    indices_set = set()
    num_sample = 0

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                batch, _rho_target = batch
            batch = batch.to(device)
            num_sample += len(batch)
            indices = model.get_indices(batch)
            indices_cpu = indices.detach().cpu()
            all_indices_per_layer.append(indices_cpu)
            for index in indices_cpu.view(-1, indices_cpu.shape[-1]).numpy():
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

    collision_rate = (num_sample - len(indices_set)) / num_sample

    # Per-layer
    all_idx = torch.cat(all_indices_per_layer, dim=0)  # (N, L)
    num_hier = all_idx.shape[-1]
    per_layer = {}
    utilization_total = 0.0
    for li in range(num_hier):
        layer_idx = all_idx[:, li]
        unique = int(layer_idx.unique().numel())
        # K 来自 vq_layer
        vq = model.hrq.vq_layers[li]
        K = vq.embeddings.weight.shape[0]
        usage = unique / K
        per_layer[f"layer_{li}"] = {
            "unique": unique,
            "total_capacity": K,
            "fraction": usage,
        }
        utilization_total += usage

    return per_layer, collision_rate, num_sample, len(indices_set), utilization_total


def main():
    args = parse_args()
    print(f"[Task #263] Loading ckpt: {args.ckpt}")
    model, ckpt_args = load_model_from_ckpt(args.ckpt, args.device)

    # Build data loader (跟 train_hrqvae.py 行 358 一致)
    print(f"[Task #263] Loading data: {args.data_path}")
    dataset = EmbDataset(args.data_path, depth_npy_path=getattr(args, "depth_npy_path", None))
    from torch.utils.data import DataLoader as _DL
    data_loader = _DL(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[Task #263] Measuring per-layer utilization (pre-revive, direct measurement)...")
    per_layer, collision_rate, num_sample, n_unique_sid, utilization_total = measure_utilization(
        model, data_loader, args.device
    )

    out = {
        "task": "task263_issue17_gate2_direct_measurement",
        "ckpt_path": args.ckpt,
        "n_items": num_sample,
        "n_unique_sid": n_unique_sid,
        "collision_rate_pre_resolve": collision_rate,
        "per_layer_utilization": per_layer,
        "utilization_avg_over_layers": utilization_total / len(per_layer),
        "ckpt_args_summary": {
            "num_emb_list": ckpt_args.get("num_emb_list"),
            "e_dim": ckpt_args.get("e_dim"),
            "loss_type": ckpt_args.get("loss_type"),
            "product_manifold": ckpt_args.get("product_manifold"),
            "beta": ckpt_args.get("beta"),
            "assignment_mode_list": ckpt_args.get("assignment_mode_list"),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Task #263] Wrote {args.out_json}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_layer_utilization"}, indent=2))
    print("per_layer_utilization:")
    for li, val in per_layer.items():
        flag = " ⚠️ below 90% (L0 stop-loss (i))" if li == "layer_0" and val["fraction"] < 0.9 else ""
        print(f"  {li}: {val['unique']}/{val['total_capacity']} = {100 * val['fraction']:.2f}%{flag}")


if __name__ == "__main__":
    main()
