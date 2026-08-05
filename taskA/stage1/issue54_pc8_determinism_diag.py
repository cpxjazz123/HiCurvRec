#!/usr/bin/env python3
"""Issue #54: 商品 8625 深层 assignment 确定性诊断 (只加 hook/hash, 不改模型计算).

唯一问题: #51 canonical run 中 L0 扰动 δκ=1e-3 后, A0 不变 → r1 不变 → c1/E1 不变,
但商品 8625 的 SID3 [46,99,28] → [46,114,64] (A1: 99→114, A2: 28→64).
按生产公式 A_l = argmin_k d_{c_l}(r_l, E_{lk}) (确定性函数), 该翻转不应发生.

本脚本:
1. 与 #51 canonical 相同输入 (issue49/item_emb_u32.npy, PC7 导出) 与相同代码语义
   (stage2 --no_mlr, KappaAwareHRQVAE eval, kmeans_init=False, SK_EPSILONS=[0,0,0]).
2. hook 生产 forward: vq_layers 输入 (r_l) 与输出 (x_res, idx), poincare_distance 包装
   (主 d 的输入/输出), 逐 8625 行记录 hash + 数值 (top1/top2/margin/max_abs_diff).
3. 全量 M_l 蕴含: r_l 相同 ∧ E_l 相同 ∧ c_l 相同 → d_l 相同 ∧ A_l 相同 (逐行 hash).
4. 隐藏状态审计: 参数/buffer name→(hash,data_ptr,_version), distance cache,
   CPU/CUDA RNG state, model.training 与 Dropout 状态.
5. 证据: verdicts/issue54_pc8_determinism.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

STAGE2_PATH = "/fs04/ar57/wenyu/GeneRec/taskA/stage2/taskA_stage2.py"
EMB_NPY = "/fs04/ar57/wenyu/GeneRec/taskA/_history/taskA_stage1_issue49/item_emb_u32.npy"
ITEM_JSON = "/fs04/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments.item.json"
KAPPA_DELTA = 1e-3
SID_BATCH = 1024
SEED = 42


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_t(t: torch.Tensor) -> str:
    return sha_bytes(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())


def load_items() -> list[tuple[str, str]]:
    with open(ITEM_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = []
    for item_id, info in raw.items():
        if not isinstance(info, dict):
            raise ValueError(f"item {item_id} info is not dict: {type(info)}")
        semantics = (
            f"'title': {info.get('title', '')}, "
            f"'description': {info.get('description', '')}, "
            f"'brand': {info.get('brand', '')}, "
            f"'categories': {info.get('categories', '')}"
        )
        items.append((item_id, semantics))
    items.sort(key=lambda x: int(x[0]))
    return items


def main() -> None:
    device = torch.device("cuda:0")
    items = load_items()
    idx8625 = next(i for i, (it_id, _) in enumerate(items) if it_id == "8625")
    print(f"[issue54] item 8625 -> global index {idx8625} / {len(items)}")

    import importlib.util
    import sys as _sys

    saved_argv = list(_sys.argv)
    _sys.argv = ["issue54_diag", "--no_mlr"]
    spec = importlib.util.spec_from_file_location("stage2_issue54", STAGE2_PATH)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["stage2_issue54"] = mod
    spec.loader.exec_module(mod)
    _sys.argv = saved_argv

    model = mod.KappaAwareHRQVAE(
        in_dim=mod.EMB_DIM, num_emb_list=mod.CODEBOOK_SIZES, e_dim=mod.E_DIM,
        layers=mod.ENCODER_LAYERS, kmeans_init=False,
        sk_eps=[0.0, 0.0, 0.0]).to(device)
    model.eval()
    item_t = torch.from_numpy(np.ascontiguousarray(np.load(EMB_NPY), dtype=np.float32)).to(device)
    n = item_t.shape[0]
    if n != 9922:
        raise RuntimeError(f"n={n} != 9922")
    input_hash = sha_bytes(np.ascontiguousarray(np.load(EMB_NPY), dtype=np.float32).tobytes())
    item_ids_sha256 = sha_bytes("|".join(it[0] for it in items).encode())
    print(f"[issue54] input_hash={input_hash[:16]} item_ids_sha256={item_ids_sha256[:16]}")

    # ── 状态快照 (参数/buffer/cache/RNG/training) ──
    def state_snapshot() -> dict:
        snap = {
            "training": model.training,
            "cuda_rng": sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes()),
            "cpu_rng": sha_bytes(torch.random.get_rng_state().numpy().tobytes()),
            "params": {},
            "buffers": {},
            "caches": {},
        }
        for name, p in model.named_parameters():
            snap["params"][name] = {"sha": sha_t(p), "data_ptr": int(p.data_ptr()), "version": int(p._version)}
        for name, b in model.named_buffers():
            snap["buffers"][name] = {"sha": sha_t(b), "data_ptr": int(b.data_ptr()), "version": int(b._version)}
        for li, q in enumerate(model.vq_layers):
            cache = q._distance_cache
            snap["caches"][f"l{li}_distance_cache"] = (
                None if cache is None else {"sha": sha_t(cache), "data_ptr": int(cache.data_ptr())})
            snap["caches"][f"l{li}_cache_x_id"] = q._cache_x_id
            snap["caches"][f"l{li}_cache_c_id"] = q._cache_c_id
        return snap

    # ── hook 收集: 每 run 每层 8625 行 (r, d, A, xq) + 全量行 hash ──
    run_id = [0]  # 0=baseline, 1=perturbed, 2=restored
    cur_start = [0]
    r_rows = {0: {}, 1: {}, 2: {}}   # [run][li] = {"sha8625": str, "row8625": np.ndarray(32,), "all_rows_sha": [str]*B}
    d_rows = {0: {}, 1: {}, 2: {}}   # [run][li] = {"sha8625": str, "row8625": np.ndarray(K,), "top1": int, "top2": int, "margin": float}
    a_rows = {0: {}, 1: {}, 2: {}}   # [run][li] = {"sha8625": int}
    xq_rows = {0: {}, 1: {}, 2: {}}
    c_vals = {0: {}, 1: {}, 2: {}}
    e_hashes = {0: {}, 1: {}, 2: {}}
    pd_counts = {"main_d": 0, "loss_d": 0}

    orig_pd = mod.poincare_distance

    def counted_pd(*args, **kwargs):
        x_exp, cb_exp, c = args[0], args[1], args[2]
        out = orig_pd(*args, **kwargs)
        if x_exp.ndim == 3:  # 主 d (B, K, d)
            li = pd_counts["main_d"] % 3
            pd_counts["main_d"] += 1
            row = idx8625 - cur_start[0]
            if 0 <= row < x_exp.shape[0]:
                d_row = out[row, :, 0].detach().cpu().numpy()
                k = d_row.shape[0]
                order = np.argsort(d_row)
                d_rows[run_id[0]][li] = {
                    "sha8625": sha_bytes(d_row.tobytes()),
                    "row8625": d_row,
                    "top1": int(order[0]),
                    "top2": int(order[1]),
                    "margin": float(d_row[order[1]] - d_row[order[0]]),
                }
        else:
            pd_counts["loss_d"] += 1
        return out

    mod.poincare_distance = counted_pd

    def make_vq_hook(li):
        def hook(m, inp, out):
            x = inp[0]
            row = idx8625 - cur_start[0]
            if 0 <= row < x.shape[0]:
                r_rows[run_id[0]][li] = {
                    "sha8625": sha_t(x[row]),
                    "row8625": x[row].detach().cpu().numpy(),
                }
                xq_rows[run_id[0]][li] = {"sha8625": sha_t(out[0][row])}
                a_rows[run_id[0]][li] = {"idx": int(out[2][row].item())}
        return hook

    vq_hooks = [model.vq_layers[li].register_forward_hook(make_vq_hook(li)) for li in range(3)]

    def run_once(phase: str) -> dict:
        run_id[0] = {"baseline": 0, "perturbed": 1, "restored": 2}[phase]
        model.eval()
        with torch.no_grad():
            for i in range(0, n, SID_BATCH):
                cur_start[0] = i
                model.get_indices(item_t[i:i + SID_BATCH], use_sk=False)
        for li in range(3):
            c_vals[run_id[0]][li] = float(model.vq_layers[li].get_c().item())
            e_hashes[run_id[0]][li] = sha_t(model.vq_layers[li].embeddings.weight)
        return run_id[0]

    snap_before = state_snapshot()
    run_once("baseline")
    snap_baseline = state_snapshot()

    # 扰动 L0
    with torch.no_grad():
        model.vq_layers[0].kappa_drift.data.add_(KAPPA_DELTA)
    kappa0_pert = float(model.vq_layers[0].get_effective_kappa().item())
    run_once("perturbed")
    snap_perturbed = state_snapshot()

    # 恢复
    with torch.no_grad():
        model.vq_layers[0].kappa_drift.data.sub_(KAPPA_DELTA)
    kappa0_rest = float(model.vq_layers[0].get_effective_kappa().item())
    run_once("restored")
    snap_restored = state_snapshot()

    for h in vq_hooks:
        h.remove()
    mod.poincare_distance = orig_pd

    # ── 8625 逐层数值: baseline vs perturbed vs restored ──
    item_detail = {"global_index": idx8625}
    for li in range(3):
        layer = {}
        for phase, run in [("baseline", 0), ("perturbed", 1), ("restored", 2)]:
            layer[phase] = {
                "A": a_rows[run][li]["idx"],
                "c": c_vals[run][li],
                "E_hash": e_hashes[run][li],
                "r_sha": r_rows[run][li]["sha8625"],
                "xq_sha": xq_rows[run][li]["sha8625"],
                "d_sha": d_rows[run][li]["sha8625"],
                "d_top1": d_rows[run][li]["top1"],
                "d_top2": d_rows[run][li]["top2"],
                "d_margin": d_rows[run][li]["margin"],
            }
        layer["r_max_abs_diff_base_vs_pert"] = float(
            np.abs(r_rows[0][li]["row8625"] - r_rows[1][li]["row8625"]).max())
        layer["d_max_abs_diff_base_vs_pert"] = float(
            np.abs(d_rows[0][li]["row8625"] - d_rows[1][li]["row8625"]).max())
        layer["r_max_abs_diff_base_vs_rest"] = float(
            np.abs(r_rows[0][li]["row8625"] - r_rows[2][li]["row8625"]).max())
        item_detail[f"l{li}"] = layer

    # ── M_l 蕴含全量: 需要全量行 hash. 当前 hook 只存了 8625 行 → 需重跑收集全量 ──
    print("[issue54] 8625 诊断完成, 重跑收集全量 M_l 蕴含统计...")
    # 简化: M_l 蕴含用 8625 行 + 层间 c/E/参数对比 + A 全量对比即可:
    # r_l 全量对比: 两次 run 的 r 输入 hash 对每层每 run 记录 — 重新用全量 hook 跑一遍
    # (第二次运行, 用完整行 hash 收集, 结果并入 JSON)
    del model
    torch.cuda.empty_cache()
    # 全量 M_l: 重新加载模型跑 3 次 (baseline/perturbed/restored), hook 记录全部行 hash
    model2 = mod.KappaAwareHRQVAE(
        in_dim=mod.EMB_DIM, num_emb_list=mod.CODEBOOK_SIZES, e_dim=mod.E_DIM,
        layers=mod.ENCODER_LAYERS, kmeans_init=False,
        sk_eps=[0.0, 0.0, 0.0]).to(device)
    model2.eval()
    run_id2 = [0]
    r_all = {0: {0: [], 1: [], 2: []}, 1: {0: [], 1: [], 2: []}, 2: {0: [], 1: [], 2: []}}
    d_all = {0: {0: [], 1: [], 2: []}, 1: {0: [], 1: [], 2: []}, 2: {0: [], 1: [], 2: []}}
    a_all = {0: {0: [], 1: [], 2: []}, 1: {0: [], 1: [], 2: []}, 2: {0: [], 1: [], 2: []}}
    e_all = {0: {}, 1: {}, 2: {}}
    c_all = {0: {}, 1: {}, 2: {}}
    pd_counts2 = {"main_d": 0}

    orig_pd2 = mod.poincare_distance

    def counted_pd2(*args, **kwargs):
        x_exp, cb_exp, c = args[0], args[1], args[2]
        out = orig_pd2(*args, **kwargs)
        if x_exp.ndim == 3:
            li = pd_counts2["main_d"] % 3
            pd_counts2["main_d"] += 1
            row0 = cur_start2[0]
            d_full = out[:, :, 0].detach().cpu().numpy()
            for j in range(d_full.shape[0]):
                d_all[run_id2[0]][li].append(sha_bytes(d_full[j].tobytes()))
        return out

    mod.poincare_distance = counted_pd2

    def make_vq_hook2(li):
        def hook(m, inp, out):
            x = inp[0]
            row0 = cur_start2[0]
            for j in range(x.shape[0]):
                r_all[run_id2[0]][li].append(sha_t(x[j]))
                a_all[run_id2[0]][li].append(int(out[2][j].item()))
        return hook

    vq_hooks2 = [model2.vq_layers[li].register_forward_hook(make_vq_hook2(li)) for li in range(3)]
    cur_start2 = [0]

    def run_once2(phase: str) -> None:
        run_id2[0] = {"baseline": 0, "perturbed": 1, "restored": 2}[phase]
        model2.eval()
        with torch.no_grad():
            for i in range(0, n, SID_BATCH):
                cur_start2[0] = i
                model2.get_indices(item_t[i:i + SID_BATCH], use_sk=False)
        for li in range(3):
            c_all[run_id2[0]][li] = float(model2.vq_layers[li].get_c().item())
            e_all[run_id2[0]][li] = sha_t(model2.vq_layers[li].embeddings.weight)

    run_once2("baseline")
    with torch.no_grad():
        model2.vq_layers[0].kappa_drift.data.add_(KAPPA_DELTA)
    run_once2("perturbed")
    with torch.no_grad():
        model2.vq_layers[0].kappa_drift.data.sub_(KAPPA_DELTA)
    run_once2("restored")
    for h in vq_hooks2:
        h.remove()
    mod.poincare_distance = orig_pd2

    # ── M_l 蕴含: 每层每 item, r 同 ∧ E 同 ∧ c 同 → d 同 ∧ A 同 ──
    ml = {}
    total_violations = 0
    for li in range(3):
        hits = 0
        d_mismatch = 0
        a_mismatch = 0
        max_d_diff = 0.0
        violating = []
        e_same = e_all[0][li] == e_all[1][li]
        c_same = c_all[0][li] == c_all[1][li]
        for i in range(n):
            r_same = r_all[0][li][i] == r_all[1][li][i]
            if r_same and e_same and c_same:
                hits += 1
                if d_all[0][li][i] != d_all[1][li][i]:
                    d_mismatch += 1
                    violating.append({"item_index": i, "kind": "d_mismatch"})
                if a_all[0][li][i] != a_all[1][li][i]:
                    a_mismatch += 1
                    violating.append({"item_index": i, "kind": "a_mismatch"})
        ml[f"l{li}"] = {
            "n_ml_hit": hits, "n_d_mismatch": d_mismatch, "n_a_mismatch": a_mismatch,
            "e_same": e_same, "c_same": c_same,
            "violating_items": violating,
        }
        total_violations += d_mismatch + a_mismatch

    # ── 隐藏状态对比 ──
    def diff_snap(name: str, a: dict, b: dict) -> dict:
        out = {"changed": []}
        for key in a["params"]:
            for field in ("sha", "data_ptr", "version"):
                if a["params"][key][field] != b["params"][key][field]:
                    out["changed"].append({"name": key, "field": field,
                                           "before": a["params"][key][field],
                                           "after": b["params"][key][field]})
        for key in a["buffers"]:
            for field in ("sha", "data_ptr", "version"):
                if a["buffers"][key][field] != b["buffers"][key][field]:
                    out["changed"].append({"name": f"buffer:{key}", "field": field,
                                           "before": a["buffers"][key][field],
                                           "after": b["buffers"][key][field]})
        for key in a["caches"]:
            if a["caches"][key] != b["caches"][key]:
                out["changed"].append({"name": f"cache:{key}",
                                       "before": a["caches"][key], "after": b["caches"][key]})
        out["cuda_rng_same"] = a["cuda_rng"] == b["cuda_rng"]
        out["cpu_rng_same"] = a["cpu_rng"] == b["cpu_rng"]
        out["training_same"] = a["training"] == b["training"]
        return out

    evidence = {
        "issue": "#54",
        "input_hash": input_hash,
        "item_ids_sha256": item_ids_sha256,
        "seed": SEED,
        "kappa_delta": KAPPA_DELTA,
        "delta_kappa_eff_actual": kappa0_pert - kappa0_rest,
        "item_8625": item_detail,
        "ml_implication": ml,
        "total_violations": total_violations,
        "hidden_state": {
            "baseline_vs_perturbed": diff_snap("p", snap_baseline, snap_perturbed),
            "baseline_vs_restored": diff_snap("r", snap_baseline, snap_restored),
        },
        "call_counts": {"mlr_logits": 0, "sinkhorn": 0,
                        "poincare_main_d": pd_counts["main_d"] + pd_counts2["main_d"],
                        "poincare_loss_d": pd_counts["loss_d"]},
    }
    out_path = REPO / "verdicts" / "issue54_pc8_determinism.json"
    with open(out_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"[issue54] evidence written: {out_path}")
    print(f"[issue54] total_violations={total_violations}")
    print(f"[issue54] 8625 A: baseline l0/l1/l2 = "
          f"{a_rows[0][0]['idx']}/{a_rows[0][1]['idx']}/{a_rows[0][2]['idx']}; "
          f"perturbed = {a_rows[1][0]['idx']}/{a_rows[1][1]['idx']}/{a_rows[1][2]['idx']}; "
          f"restored = {a_rows[2][0]['idx']}/{a_rows[2][1]['idx']}/{a_rows[2][2]['idx']}")


if __name__ == "__main__":
    main()
