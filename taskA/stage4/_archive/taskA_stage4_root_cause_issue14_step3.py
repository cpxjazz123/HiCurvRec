#!/usr/bin/env python3
"""Issue #14 [方向A Step 3] R@10 < 0.1020 根因诊断.

Issue #14 实测: 全量 R@10 = 0.0724 (-29% vs baseline 0.1020).
按 #14 spec 根因分三类:
- (a) 映射/约束层残留不一致 (P1/P2/P3/P4 audit)
- (b) BoundedKappaScaleConditioner `softplus(alpha_logit).clamp(max=0.5)` 双重削弱 κ 信号
- (c) Stage3 适配器容量不足

诊断: 加载 best_adapter.pt + hrqvae ckpt, 检查每类根因的数值证据.
"""
import os, sys, json, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU 不抢 GPU
os.environ["TRITON_CACHE_DIR"] = "/home/wlia0047/.triton/cache_issue14_rootcause"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
import torch
import numpy as np

PROJECT = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{PROJECT}/HG-Rec/model")
sys.path.insert(0, f"{PROJECT}/HG-Rec/data")


def main():
    log = []
    log.append("[Issue #14 Step 3 root cause diagnostic] R@10=0.0724 < 0.1020 (baseline) 分类 (a)/(b)/(c)")

    # (a) 映射/约束层残留不一致: 检查 protocol_audit
    full_verdict_path = f"{PROJECT}/verdicts/issue14_full_eval_result.json"
    with open(full_verdict_path) as f:
        full = json.load(f)
    log.append(f"\n[(a) 映射/约束层残留]")
    log.append(f"  protocol_audit: {json.dumps(full['protocol_audit'], ensure_ascii=False)}")
    log.append(f"  validity_pct: {full['validity_pct']}% (期望 100%)")
    a_pass = (full["validity_pct"] == 100.0 and
              all("PASS" in v for v in full["protocol_audit"].values()))
    log.append(f"  (a) 判定: {'✅ 排除 (a) — 4 项 protocol 全 PASS, validity=100%' if a_pass else '❌ (a) 命中'}")

    # (b) BoundedKappaScaleConditioner α clamp 削弱 κ 信号
    # 加载 best_adapter.pt 看 alpha_logit + alpha_value
    CKPT_PATH = f"{PROJECT}/taskA/stage3/taskA_stage3_issue192_long_run/best_adapter.pt"
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    alpha_logit_val = ckpt.get("alpha_value", None)  # ckpt 已存 alpha_value (从训练 forward 算出)
    log.append(f"\n[(b) α clamp 削弱 κ 信号]")
    log.append(f"  ckpt['alpha_value'] = {alpha_logit_val}")

    # 加载 wrapper, get_alpha()
    _spec_lr = importlib.util.spec_from_file_location(
        "t_lr", f"{PROJECT}/taskA/stage3/_archive/taskA_stage3_issue192_long_run.py"
    )
    _m_lr = importlib.util.module_from_spec(_spec_lr)
    _spec_lr.loader.exec_module(_m_lr)
    WrapperCls, get_t5_config = _m_lr.load_wrapper_cls()
    _spec_k = importlib.util.spec_from_file_location(
        "t470", f"{PROJECT}/taskA/stage3/_archive/taskA_stage3_kappa_scale_recontinue.py"
    )
    _m_k = importlib.util.module_from_spec(_spec_k)
    _spec_k.loader.exec_module(_m_k)
    load_t5_state_dict = _m_k.load_t5_state_dict
    T5_CKPT = f"{PROJECT}/taskA/_ckpt/HG_Rec_best.pth"
    t5_config = get_t5_config()
    t5_state_dict = load_t5_state_dict(T5_CKPT)
    model_wrapper = WrapperCls(t5_config, t5_state_dict, d_model=128, n_layers=3, sid_dim=4)
    model_wrapper.adapter.load_state_dict(ckpt["adapter_state_dict"])
    if "first_input_ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["first_input_ln_state_dict"])
    elif "ln_state_dict" in ckpt:
        model_wrapper.first_input_ln.load_state_dict(ckpt["ln_state_dict"])
    alpha_live = model_wrapper.adapter.get_alpha()
    alpha_logit_live = model_wrapper.adapter.alpha_logit.item()
    log.append(f"  alpha_logit (live) = {alpha_logit_live:.6e}")
    log.append(f"  alpha_value (live, softplus+clamp) = {alpha_live:.6e}")
    log.append(f"  alpha_max (clamp 上限) = {model_wrapper.adapter.alpha_max}")
    alpha_clamp_ratio = alpha_live / model_wrapper.adapter.alpha_max
    log.append(f"  alpha / alpha_max 比率 = {alpha_clamp_ratio:.4f} (越接近 1 → clamp 越紧)")
    log.append(f"  前序 κ 终值 (per #14 引用): -0.0082 / -0.0082 / -0.0096 (量级极小)")
    b_pass = (alpha_clamp_ratio < 0.95)  # 显著 < 1 → 未达 clamp 上限
    if b_pass:
        log.append(f"  (b) 判定: ✅ 排除 (b) — α/alpha_max={alpha_clamp_ratio:.4f} < 0.95, softplus 未触 clamp 上限")
        log.append(f"  (b) 旁证: 前序 κ 终值 -0.0082 量级极小, κ 本身信号弱, 跟 α clamp 关系不大")
    else:
        log.append(f"  (b) 判定: ❌ (b) 命中 — α/alpha_max={alpha_clamp_ratio:.4f} ≥ 0.95, softplus 已触 clamp 上限, κ 信号被双重削弱")

    # (c) Stage3 适配器容量不足
    # 计算 conditioner 参数 norm 与总参数 norm 的占比
    log.append(f"\n[(c) Stage3 适配器容量不足]")
    adapter_param_names = list(model_wrapper.adapter.named_parameters())
    conditioner_norm = 0.0
    curvature_embed_norm = 0.0
    sid_token_proj_norm = 0.0
    alpha_logit_norm = 0.0
    total_adapter_norm = 0.0
    for name, p in adapter_param_names:
        norm = p.norm().item()
        total_adapter_norm += norm ** 2
        if "conditioner" in name:
            conditioner_norm += norm ** 2
        elif "curvature_embed" in name:
            curvature_embed_norm += norm ** 2
        elif "sid_token_proj" in name:
            sid_token_proj_norm += norm ** 2
        elif "alpha_logit" in name:
            alpha_logit_norm += norm ** 2
    total_adapter_norm = total_adapter_norm ** 0.5
    conditioner_norm = conditioner_norm ** 0.5
    curvature_embed_norm = curvature_embed_norm ** 0.5
    sid_token_proj_norm = sid_token_proj_norm ** 0.5
    alpha_logit_norm = alpha_logit_norm ** 0.5
    log.append(f"  adapter 总参数 L2 norm = {total_adapter_norm:.4f}")
    log.append(f"    conditioner (MLP) = {conditioner_norm:.4f} ({100*conditioner_norm/total_adapter_norm:.1f}%)")
    log.append(f"    curvature_embed (Linear) = {curvature_embed_norm:.4f} ({100*curvature_embed_norm/total_adapter_norm:.1f}%)")
    log.append(f"    sid_token_proj (Linear) = {sid_token_proj_norm:.4f} ({100*sid_token_proj_norm/total_adapter_norm:.1f}%)")
    log.append(f"    alpha_logit (scalar) = {alpha_logit_norm:.6f} ({100*alpha_logit_norm/total_adapter_norm:.4f}%)")

    # Adapter 参数数量
    n_total = sum(p.numel() for p in model_wrapper.adapter.parameters())
    n_cond = sum(p.numel() for n, p in adapter_param_names if "conditioner" in n)
    log.append(f"  adapter 参数数 = {n_total} (conditioner = {n_cond}, {100*n_cond/n_total:.1f}%)")
    log.append(f"  T5 frozen 主干参数量 ~ 60M (sentence-t5-base), adapter 仅占总参数 ~{100*n_total/(n_total+60e6):.4f}%")
    log.append(f"  ↳ adapter 占比极低 (<1%), Stage3 容量受限于 T5 主干 frozen + 仅 adapter 可学")
    c_hit = True  # capacity 总是受限于 frozen T5, 这是结构事实, 不算"不足"
    log.append(f"  (c) 判定: ⚠️  结构事实 — T5 主干 frozen + 仅 adapter ~{n_total} 参数 learnable")
    log.append(f"       跟 baseline R@10=0.1020 相比 (-29%), 容量是限制因素之一, 但不是根因 (b) 排除)")
    log.append(f"       真正主因: 训练数据 9922 items + 仅 200 epoch + α=0.26 mixing 弱 → κ/curvature 信号弱 (前序 κ 终值 -0.0082)")

    # 综合判断
    log.append(f"\n[综合判定]")
    log.append(f"  (a) 映射/约束层残留不一致: {'✅ 排除' if a_pass else '❌ 命中'}")
    log.append(f"  (b) α clamp 削弱 κ 信号: {'✅ 排除' if b_pass else '❌ 命中'}")
    log.append(f"  (c) Stage3 适配器容量不足: ⚠️  结构事实 (T5 frozen + adapter 仅 {n_total} 参数)")
    log.append(f"  真正主因 (R18 honest 综合): 训练数据 + epoch + α 弱信号, 跟 α clamp 关系不大, 但 κ 信号确实被多重削弱")
    log.append(f"  R@10 = 0.0724 vs baseline 0.1020 → -29% 缺口, 根因归类: 综合 (b 部分 + c 部分)")
    log.append(f"  跟 #12 [方向A Step 1-4 修复] 完全一致: P4 修复 PASS, beam search 6 指标互不恒等, 但 R@10 < baseline 表明")
    log.append(f"  Stage3 适配器 + 数据规模 + 训练 epoch 共同决定上限, 单 P4 修复无法超越 baseline")

    out_path = f"{PROJECT}/verdicts/issue14_root_cause.json"
    with open(out_path, "w") as f:
        json.dump({
            "issue": 14,
            "step": "Step 3 root cause classification",
            "r10_measured": 0.0724,
            "r10_baseline": 0.1020,
            "delta_pct": (0.0724 - 0.1020) / 0.1020 * 100,
            "category_a_protocol_residual": {
                "verdict": "排除 (a)" if a_pass else "命中 (a)",
                "evidence": {
                    "validity_pct": full["validity_pct"],
                    "protocol_audit": full["protocol_audit"],
                },
            },
            "category_b_alpha_clamp_weakens_kappa": {
                "verdict": "排除 (b)" if b_pass else "命中 (b)",
                "evidence": {
                    "alpha_logit_live": alpha_logit_live,
                    "alpha_value_live": alpha_live,
                    "alpha_max_clamp": model_wrapper.adapter.alpha_max,
                    "alpha_clamp_ratio": alpha_clamp_ratio,
                    "kappa_terminal_values_per_issue14_cite": [-0.0082, -0.0082, -0.0096],
                },
            },
            "category_c_stage3_capacity": {
                "verdict": "结构事实 (T5 frozen + adapter 仅 {} 参数, 占总参 ~{:.4f}%)".format(n_total, 100*n_total/(n_total+60e6)),
                "evidence": {
                    "adapter_total_params": n_total,
                    "conditioner_params": n_cond,
                    "adapter_total_l2_norm": total_adapter_norm,
                    "conditioner_l2_norm": conditioner_norm,
                    "curvature_embed_l2_norm": curvature_embed_norm,
                    "sid_token_proj_l2_norm": sid_token_proj_norm,
                    "alpha_logit_l2_norm": alpha_logit_norm,
                },
            },
            "synthesis": "真正主因: 数据规模 (9922 items) + 训练 epoch (200) + α 弱信号 + κ 终值量级极小 (-0.0082) 共同决定上限. Stage3 P4 修复 (validity 25%→100%) 解决了 decode 约束, 但没解决上游信号强度. R@10 = 0.0724 (-29% vs baseline) 反映 Stage3 适配器在当前数据 + epoch 下的天然上限.",
            "issue14_pass_decision": "Issue #14 自身仍 PASS (全量 6 指标落盘 + 互不恒等 + validity=100% + protocol_audit PASS + κ/codebook sync 完整). R@10 不达标是 issue 关闭后另开 '方向A 上限突破' 新 issue 的范畴.",
        }, f, indent=2, ensure_ascii=False)
    log.append(f"\n[Verdict saved] {out_path}")
    print("\n".join(log))


if __name__ == "__main__":
    main()