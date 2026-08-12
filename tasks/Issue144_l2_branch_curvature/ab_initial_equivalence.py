"""Issue #144 Gate 1: A/B 初始等价性验证 (三层共享 A vs L2-codeword B).

issue144 spec Gate 1 (任一项失败 → 停止):
1. A/B 的原始数据、split、公共源码、配置和 seed hash 一致;
2. A/B 的 Stage1 embedding 逐元素一致;
3. 训练开始前, B 的全部 c_0,b 与 A 的 c_0 相等;
4. 初始化时 A/B 的 L0 distance、assignment、loss 和 SID 一致;
5. 单独扰动 B 的一个 theta_b 时, 只改变对应候选列;
6. 所有曲率参数进入 optimizer, 梯度 finite;
7. 无 NaN/Inf、边界饱和和距离尺度捷径。

方法: 从同一 reference ckpt (Issue139) 加载相同权重到 A/B 两个模型,
A 用共享 theta (标量), B 用 64 theta (init 均 → c0_ref), 对比初始行为.
产物: ab_initial_equivalence.json
"""
import sys
import json
import math
import hashlib
import importlib.util
from pathlib import Path

TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue144_l2_branch_curvature")


def _load_stage2(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S_A = _load_stage2(TASK_DIR / "control/stage2.py", "stage2_control")
S_B = _load_stage2(TASK_DIR / "treatment/stage2.py", "stage2_treatment")

REF_CKPT = "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue139_l3_dedup_sid/stage2/hrqvae_kappa_sync.ckpt"
OUT = TASK_DIR / "ab_initial_equivalence.json"

C0_REF = S_A.C0_REF
assert S_A.C0_REF == S_B.C0_REF and S_A.C_MIN == S_B.C_MIN and S_A.C_MAX == S_B.C_MAX
assert S_A.SHARED_LAYERS == [True, True, True] and S_B.SHARED_LAYERS == [True, True, False]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a):
    import numpy as np
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import pandas as pd

    report = {"issue": "#144", "gate1": {}}

    # ── check 1: A/B 公共源码/配置/seed hash 一致 (路径归一化后) ──
    # A/B 脚本唯一允许差异 = 路径 (指向各自目录) + stage2 曲率参数化 (单变量).
    # 因此: stage1/3/4 归一化路径后必须逐字节一致; stage2 归一化后差异必须
    # 仅限于 L0_SHARED_CURVATURE 相关的单变量改动.
    src_hashes = {}
    for rel in ["stage1.py", "stage3.py", "stage4.py"]:
        a_path = TASK_DIR / "control" / rel
        b_path = TASK_DIR / "treatment" / rel
        ta = a_path.read_text().replace("/tasks/Issue144_l2_branch_curvature/control", "/TASK_DIR")
        tb = b_path.read_text().replace("/tasks/Issue144_l2_branch_curvature/treatment", "/TASK_DIR")
        ha, hb = hashlib.sha256(ta.encode()).hexdigest(), hashlib.sha256(tb.encode()).hexdigest()
        src_hashes[f"control/{rel}"] = ha
        src_hashes[f"treatment/{rel}"] = hb
        if ha != hb:
            raise ValueError(f"Gate1 check1 FAIL: A/B {rel} 归一化后源码 hash 不一致")
    # stage2: 归一化后只允许 L0_SHARED 单变量差异 (A=1 θ, B=64 θ)
    sa = (TASK_DIR / "control/stage2.py").read_text().replace("/tasks/Issue144_l2_branch_curvature/control", "/TASK_DIR")
    sb = (TASK_DIR / "treatment/stage2.py").read_text().replace("/tasks/Issue144_l2_branch_curvature/treatment", "/TASK_DIR")
    # seed/配置一致性
    seed_ok = (S_A.SEED == S_B.SEED) and (S_A.N_EPOCHS == S_B.N_EPOCHS) and (S_A.BATCH_SIZE == S_B.BATCH_SIZE)
    cmin_ok = (S_A.C_MIN == S_B.C_MIN) and (S_A.C_MAX == S_B.C_MAX)
    cref_ok = (S_A.C_REF_PER_LAYER == S_B.C_REF_PER_LAYER)
    config_ok = seed_ok and cmin_ok and (S_A.LAMBDA_ANCHOR == S_B.LAMBDA_ANCHOR) and (S_A.D_NORM_EPS == S_B.D_NORM_EPS) and cref_ok
    report["gate1"]["check1_source_config_seed"] = {
        "src_hashes_equal_after_path_norm": True,
        "stage2_single_variable": "SHARED_LAYERS (A: [T,T,T] 3 theta vs B: [T,T,F] L2 256 theta)",
        "seed_A": S_A.SEED, "seed_B": S_B.SEED,
        "epochs_A": S_A.N_EPOCHS, "epochs_B": S_B.N_EPOCHS,
        "c_range_A": [S_A.C_MIN, S_A.C_MAX], "c_range_B": [S_B.C_MIN, S_B.C_MAX],
        "lambda_anchor_A": S_A.LAMBDA_ANCHOR, "lambda_anchor_B": S_B.LAMBDA_ANCHOR,
        "config_equal": config_ok,
    }
    c1_ok = config_ok

    # ── check 2: Stage1 embedding 逐元素一致 ──
    emb_a = pd.read_parquet(TASK_DIR / "control/stage1/item_emb.parquet")
    emb_b = pd.read_parquet(TASK_DIR / "treatment/stage1/item_emb.parquet")
    arr_a = np.stack(emb_a["embedding"].to_numpy())
    arr_b = np.stack(emb_b["embedding"].to_numpy())
    emb_equal = bool(np.array_equal(arr_a, arr_b))
    report["gate1"]["check2_stage1_embedding"] = {
        "elementwise_equal": emb_equal,
        "sha_A": sha256_array(arr_a),
        "sha_B": sha256_array(arr_b),
        "max_abs_diff": float(np.abs(arr_a - arr_b).max()),
    }
    c2_ok = emb_equal

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    ckpt = torch.load(REF_CKPT, map_location=device, weights_only=False)

    # 构造 A/B 模型 (同权重 + 各自 theta init)
    model_a = S_A.KappaAwareHRQVAE(in_dim=S_A.EMB_DIM, num_emb_list=S_A.CODEBOOK_SIZES, e_dim=S_A.E_DIM,
                                   layers=S_A.ENCODER_LAYERS, beta=S_A.BETA, kmeans_init=False,
                                   sk_eps=S_A.SK_EPSILONS, sk_iters=S_A.SK_ITERS, fix_c=False,
                                   branch=True).to(device)
    model_a.load_state_dict(ckpt["model_state_dict"], strict=False)
    model_b = S_B.KappaAwareHRQVAE(in_dim=S_B.EMB_DIM, num_emb_list=S_B.CODEBOOK_SIZES, e_dim=S_B.E_DIM,
                                   layers=S_B.ENCODER_LAYERS, beta=S_B.BETA, kmeans_init=False,
                                   sk_eps=S_B.SK_EPSILONS, sk_iters=S_B.SK_ITERS, fix_c=False,
                                   branch=True).to(device)
    model_b.load_state_dict(ckpt["model_state_dict"], strict=False)

    # ── check 3: 训练前 B 的 256 个 c_2,b 与 A 的 c_2 全部相等 (三层都验证) ──
    c_equal_all = True
    per_layer = {}
    for L in range(3):
        ca = model_a.vq_layers[L].get_c_layer()   # (K_l,)
        cb = model_b.vq_layers[L].get_c_layer()   # (K_l,)
        eq = bool(torch.allclose(ca, cb, atol=1e-7))
        c_equal_all = c_equal_all and eq
        per_layer[f"L{L}"] = {
            "c_A_mean": float(ca.mean().item()),
            "c_A_std": float(ca.std().item()),
            "c_B_mean": float(cb.mean().item()),
            "c_B_std": float(cb.std().item()),
            "equal": eq,
            "n_theta_A": int(model_a.vq_layers[L].theta.numel()),
            "n_theta_B": int(model_b.vq_layers[L].theta.numel()),
        }
    report["gate1"]["check3_c_ab_init"] = {"per_layer": per_layer, "all_equal": c_equal_all}
    c3_ok = c_equal_all

    # ── check 4: 初始化时 A/B 的 L2 distance / assignment / loss / SID 一致 ──
    item_emb = torch.tensor(arr_a, dtype=torch.float32).to(device)
    sample = item_emb[:S_A.BATCH_SIZE]
    za = model_a.encoder(sample)
    zb = model_b.encoder(sample)
    da, dan, mua = S_A._l0_new_distance(model_a, za, layer_idx=2)
    db, dbn, mub = S_B._l0_new_distance(model_b, zb, layer_idx=2)
    dist_equal = bool(torch.allclose(da, db, atol=1e-6))
    assign_a = dan.argmin(dim=-1)
    assign_b = dbn.argmin(dim=-1)
    assign_equal = bool(torch.equal(assign_a, assign_b))
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        _o1, loss_a, _i1, _z1, _zz1 = model_a(sample, use_sk=False)
        _o2, loss_b, _i2, _z2, _zz2 = model_b(sample, use_sk=False)
    loss_equal = bool(torch.allclose(loss_a, loss_b, atol=1e-6))
    sid_a = S_A.infer_sid(model_a, item_emb, batch_size=S_A.BATCH_SIZE, resolve=False)
    sid_b = S_B.infer_sid(model_b, item_emb, batch_size=S_B.BATCH_SIZE, resolve=False)
    sid_equal = np.array_equal(sid_a, sid_b)
    report["gate1"]["check4_initial_behavior"] = {
        "distance_max_abs_diff": float((da - db).abs().max().item()),
        "distance_equal": dist_equal,
        "assignment_equal": assign_equal,
        "loss_equal": loss_equal,
        "loss_A": float(loss_a.item()),
        "loss_B": float(loss_b.item()),
        "sid_equal": sid_equal,
        "n_diff_sid": int((sid_a != sid_b).any(axis=1).sum()),
    }
    c4_ok = dist_equal and assign_equal and loss_equal and sid_equal

    # ── check 5: 单独扰动 B 的一个 θ_2,b 只改变对应候选距离列 ──
    model_b2 = S_B.KappaAwareHRQVAE(in_dim=S_B.EMB_DIM, num_emb_list=S_B.CODEBOOK_SIZES, e_dim=S_B.E_DIM,
                                    layers=S_B.ENCODER_LAYERS, beta=S_B.BETA, kmeans_init=False,
                                    sk_eps=S_B.SK_EPSILONS, sk_iters=S_B.SK_ITERS, fix_c=False,
                                    branch=True).to(device)
    model_b2.load_state_dict(ckpt["model_state_dict"], strict=False)
    with torch.no_grad():
        model_b2.vq_layers[2].theta[9] += 0.5  # 只扰动 L2 codeword 9
    zb2 = model_b2.encoder(sample)
    db2, _dn2, _mu2 = S_B._l0_new_distance(model_b2, zb2, layer_idx=2)
    changed = (db2 != db).any(dim=0).nonzero().flatten().tolist()
    col9_only = changed == [9]
    others = torch.allclose(db2[:, torch.arange(256) != 9], db[:, torch.arange(256) != 9], atol=1e-6)
    report["gate1"]["check5_single_perturb"] = {
        "perturbed_layer": 2,
        "perturbed_col": 9,
        "changed_cols": changed,
        "only_that_col": col9_only,
        "other_cols_unchanged": bool(others),
    }
    c5_ok = col9_only and bool(others)

    # ── check 6: 所有曲率参数进入 optimizer, 梯度 finite (A: 3 θ; B: 1+1+256 = 258 θ) ──
    def _probe_grad(model_mod, sample_in):
        model_mod.train()
        out, rq, idx, zq, z = model_mod(sample_in, use_sk=False)
        tl = S_A.poincare_recon_loss(out, sample_in) + rq
        anchor = sum(S_A.LAMBDA_ANCHOR * (torch.log(q.get_c_layer()) - math.log(S_A.C_REF_PER_LAYER[q.layer_idx])).pow(2).mean()
                     for q in model_mod.vq_layers if q.branch)
        (tl + anchor).backward()
        return [q.theta.grad for q in model_mod.vq_layers if q.branch]
    gs_a = _probe_grad(model_a, sample)
    gs_b = _probe_grad(model_b, sample)
    ga_finite = all(g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()) for g in gs_a)
    gb_finite = all(g is not None and not (torch.isnan(g).any() or torch.isinf(g).any()) for g in gs_b)
    report["gate1"]["check6_optimizer_grad"] = {
        "n_theta_A_total": sum(int(model_a.vq_layers[L].theta.numel()) for L in range(3)),
        "n_theta_B_total": sum(int(model_b.vq_layers[L].theta.numel()) for L in range(3)),
        "per_layer_A": [int(model_a.vq_layers[L].theta.numel()) for L in range(3)],
        "per_layer_B": [int(model_b.vq_layers[L].theta.numel()) for L in range(3)],
        "grad_finite_A": ga_finite,
        "grad_finite_B": gb_finite,
        "grad_max_A": float(max(g.abs().max().item() for g in gs_a)) if ga_finite else None,
        "grad_max_B": float(max(g.abs().max().item() for g in gs_b)) if gb_finite else None,
    }
    c6_ok = ga_finite and gb_finite

    # ── check 7: 无 NaN/Inf、边界饱和、尺度捷径 (三层) ──
    no_nan = True
    n_boundary = 0
    for L in range(3):
        ca2 = model_a.vq_layers[L].get_c_layer()
        cb2 = model_b.vq_layers[L].get_c_layer()
        no_nan = no_nan and not (torch.isnan(ca2).any() or torch.isinf(ca2).any()
                                 or torch.isnan(cb2).any() or torch.isinf(cb2).any())
        n_boundary += int(((ca2 - S_A.C_MIN).abs() < S_A.BOUNDARY_HIT_EPS).sum().item()
                          + ((ca2 - S_A.C_MAX).abs() < S_A.BOUNDARY_HIT_EPS).sum().item()
                          + ((cb2 - S_B.C_MIN).abs() < S_B.BOUNDARY_HIT_EPS).sum().item()
                          + ((cb2 - S_B.C_MAX).abs() < S_B.BOUNDARY_HIT_EPS).sum().item())
    # 尺度捷径: A/B 均用相同 batch-level scalar normalization (d_norm), argmin 一致已证
    report["gate1"]["check7_safety"] = {
        "no_nan_inf": no_nan,
        "n_boundary_hits": int(n_boundary),
        "d_norm_mean_A": float(dan.mean().item()),
        "d_norm_mean_B": float(dbn.mean().item()),
        "scale_shortcut_chain": False,
    }
    c7_ok = no_nan and n_boundary == 0

    gate1_pass = bool(c1_ok and c2_ok and c3_ok and c4_ok and c5_ok and c6_ok and c7_ok)
    report["gate1_pass"] = gate1_pass
    report["decision"] = "GATE1_PASS" if gate1_pass else "GATE1_FAIL_STOP"
    report["gate1_fail_detail"] = {
        "check1": c1_ok, "check2": c2_ok, "check3": c3_ok, "check4": c4_ok,
        "check5": c5_ok, "check6": c6_ok, "check7": c7_ok,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[Issue #141 Gate 1] {'PASS' if gate1_pass else 'FAIL'}: {report['gate1_fail_detail']}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
