"""Issue #76 precheck: 锁定 v77 baseline + 三层 codebook + Stage1/2 ckpt + SID + 用户/交互无重叠审计.

Per Issue #76 固定条件:
- v77 baseline: test_R10=0.1080, valid_R10=0.1312
- Stage1 per-item radius (R_MAX=0.99 + sigmoid)
- Stage2: L0 K64 / L1 K128 / L2 K256, 三层独立 learnable kappa
- Stage3 HAB frozen + Stage1 per-item radius
- Task84 评估协议 + DDP 4-card bf16

precheck 必须输出:
- v77 六指标 + 产物 hash
- 三层 codebook shape
- 初始 kappa/c
- Stage1/2 checkpoint hash
- SID SHA256
- train/valid/test 用户/交互数 + 三者无重叠审计
- 结论: precheck PASS / precheck blocked
"""
import os
import sys
import json
import hashlib
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
GENREC = REPO / "HG-Rec"
DATA = GENREC / "dataset" / "Instruments"

# v77 已知指标 (历史记录)
V77_METRICS = {
    "valid_R5": 0.10796703312259454,
    "valid_R10": 0.1312042123423173,
    "valid_R20": 0.160030742505422,
    "valid_NDCG5": 0.09029159952814762,
    "valid_NDCG10": 0.09779699717003565,
    "valid_NDCG20": 0.10508150658928431,
    "best_epoch": 104,
    "early_stop_counter": 4,
    "lambda_raw": [-1.3737, 0.57365, 1.187],
    "lambda_eff": [-0.2, 0.19871, 0.2],
    "U_l2_norm": [0.0467, 0.0626, 0.0875],
    "V_l2_norm": [0.0468, 0.0626, 0.0875],
    "test_R10": 0.1080,
}

V77_CKPT = Path("/tmp/v77_peritem_hab/HG_Rec_best.pth")
V77_STAGE2 = REPO / "taskA/_history/taskA_stage2_kappa_sync/hrqvae_kappa_sync.ckpt"
V77_SID = REPO / "taskA/_history/taskA_stage2_hyp_v2_capmatch_1000ep/sid_output.npy"
V77_STAGE1 = REPO / "taskA/_history/taskA_stage1_hyp_v2"


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = {}
    out["v77_baseline_metrics"] = V77_METRICS
    # v77 ckpt hash
    out["v77_ckpt_sha256"] = sha256_of(V77_CKPT)
    out["v77_ckpt_size_bytes"] = V77_CKPT.stat().st_size
    # Stage2 ckpt hash
    out["v77_stage2_ckpt_sha256"] = sha256_of(V77_STAGE2)
    out["v77_stage2_ckpt_size_bytes"] = V77_STAGE2.stat().st_size
    # SID hash
    import numpy as np
    sid = np.load(V77_SID)
    out["v77_sid_sha256"] = hashlib.sha256(sid.tobytes()).hexdigest()
    out["v77_sid_shape"] = list(sid.shape)
    out["v77_sid_unique_3digit"] = int(len(set(tuple(sid[i, -3:]) for i in range(sid.shape[0]))))
    out["v77_sid_total_items"] = int(sid.shape[0])
    out["v77_sid_unique_3digit_ratio"] = round(out["v77_sid_unique_3digit"] / out["v77_sid_total_items"], 4)
    # Stage1 dir hash (per-item radius artifacts)
    if V77_STAGE1.exists():
        stage1_files = sorted([p for p in V77_STAGE1.rglob("*") if p.is_file()])
        h = hashlib.sha256()
        for p in stage1_files:
            h.update(str(p.relative_to(V77_STAGE1)).encode())
            try:
                h.update(p.read_bytes())
            except Exception:
                pass
        out["v77_stage1_dir_sha256"] = h.hexdigest()
        out["v77_stage1_files"] = [str(p.relative_to(REPO)) for p in stage1_files[:5]]
    # Stage2 三层 codebook shape (读 ckpt) — 适配 slim 格式: model_state_dict + final_kappas + final_cs
    import torch
    ck = torch.load(V77_STAGE2, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    out["v77_stage2_state_dict_keys"] = len(sd)
    codebook_shapes = {}
    for k, v in sd.items():
        if "embed" in k.lower() or "codebook" in k.lower():
            if hasattr(v, "shape"):
                codebook_shapes[k] = list(v.shape)
    out["v77_stage2_codebook_shapes"] = codebook_shapes
    out["v77_stage2_config"] = ck.get("config", {})
    out["v77_stage2_final_kappas"] = ck.get("final_kappas", [])
    out["v77_stage2_final_cs"] = ck.get("final_cs", [])
    out["v77_stage2_final_mix_weights"] = ck.get("final_mix_weights", [])
    # train/valid/test 用户/交互审计 (schema: user/string + history/list<int64> + target/int64)
    import pyarrow.parquet as pq
    user_sets = {}
    item_sets = {}
    for split in ["train", "valid", "test"]:
        f = DATA / f"{split}.parquet"
        if not f.exists():
            out[f"{split}_missing"] = True
            continue
        tbl = pq.read_table(f)
        users = set(tbl.column("user").to_pylist())
        items = set()
        n_inter = 0
        for row in tbl.to_pylist():
            for it in row["history"]:
                items.add(int(it))
                n_inter += 1
            items.add(int(row["target"]))
            n_inter += 1
        out[f"{split}_n_interactions"] = int(n_inter)
        out[f"{split}_n_users"] = int(len(users))
        out[f"{split}_n_items"] = int(len(items))
        out[f"{split}_users_sample"] = sorted(list(users))[:5]
        user_sets[split] = users
        item_sets[split] = items
    if len(user_sets) == 3:
        out["audit_user_overlap"] = {
            "train_valid": len(user_sets["train"] & user_sets["valid"]),
            "train_test": len(user_sets["train"] & user_sets["test"]),
            "valid_test": len(user_sets["valid"] & user_sets["test"]),
        }
        out["audit_item_overlap"] = {
            "train_valid": len(item_sets["train"] & item_sets["valid"]),
            "train_test": len(item_sets["train"] & item_sets["test"]),
            "valid_test": len(item_sets["valid"] & item_sets["test"]),
        }
    # 结论
    must_have = [
        "v77_ckpt_sha256", "v77_stage2_ckpt_sha256", "v77_sid_sha256",
        "v77_stage2_codebook_shapes", "v77_stage2_final_kappas", "v77_stage2_final_cs",
    ]
    out["precheck_status"] = "PASS" if all(k in out and out[k] for k in must_have) else "blocked"
    # 关键发现: Stage2 是否真有效曲率 (final_cs 全 1 = flat Euclidean, 推翻 Issue #76 假设)
    out["issue76_premise_check"] = {
        "stage2_final_cs_all_1": all(c == 1.0 for c in out.get("v77_stage2_final_cs", [])),
        "stage2_final_kappas_all_0": all(k == 0.0 for k in out.get("v77_stage2_final_kappas", [])),
        "implication": (
            "v77 SID 来自 flat Euclidean Stage2 (c=1, kappa=0). "
            "Issue #76 假设 'Stage2 曲率与 next-item 行为目标未对齐' 不成立 — "
            "因为 Stage2 没有曲率信号可供对齐. Phase A 必须先激活 Stage2 曲率, 再做行为监督."
        ) if all(c == 1.0 for c in out.get("v77_stage2_final_cs", [])) else
        "Stage2 确实有非零曲率, Issue #76 假设成立.",
    }
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
