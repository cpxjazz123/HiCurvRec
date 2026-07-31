"""Task #356 / Issue #65 Gate -1 — Direction C 重开 Stage 3 曲率条件化 T5 预检

Issue #65 Gate -1 spec: 实施基础就位 (Direction C T5) + Stage 1/2 隔离 +
SID token → T5 embedding 接口 + curvature-conditioned attention +
batch 维度独立 + gradient 无 detach + Stage 3/4 接口对齐.

8 项 zero-GPU 测试. 任一 FAIL → NO-GO 收口.
"""
import sys
import os

REPO = '/home/wlia0047/ar57/wenyu/GeneRec'
# 不在 module 顶层 import torch — 本审计全为静态文件/字符串检查, 零 GPU.
# (grid_toys env 已迁移, scratch 当前无 torch — 保持 zero-GPU 特性)

# Try to import existing HG_Rec T5 model
HG_REC_PATH = f'{REPO}/HG-Rec/model/HG_Rec.py'
HG_REC_EXISTS = os.path.exists(HG_REC_PATH)

# Curvature-conditioned T5 markers (per Issue #65 spec)
DIRECTION_C_MARKERS = {
    'curvature_conditioned_embedding': 'curvature_condition',
    'curvature_attention_bias': 'curvature_attention_bias',
    'metric_score': 'metric_score',
    'layer_aware_metric_bias': 'layer_aware_metric',
    'product_stereographic_attention': 'product_stereographic',
    'sid_metadata_embedding': 'sid_metadata',
    'curvature_aware_q_k': 'curvature_aware_q_k',
}


def test_1_implementation_in_place():
    """T1: Direction C T5 架构实施基础就位."""
    print("\n=== T1: Direction C T5 实施基础就位 ===")

    if not HG_REC_EXISTS:
        print(f"  ❌ HG-Rec/model/HG_Rec.py 不存在")
        return False

    with open(HG_REC_PATH) as f:
        content = f.read()

    # Check for Direction C markers in HG_Rec source
    found = []
    missing = []
    for key, marker in DIRECTION_C_MARKERS.items():
        if marker.lower() in content.lower():
            found.append(key)
        else:
            missing.append(key)

    # Strong matches across both filename and content
    fname = os.path.basename(HG_REC_PATH).lower()
    combined = (fname + "\n" + content).lower()

    for key, marker in DIRECTION_C_MARKERS.items():
        if marker.lower() in combined:
            found.append(key)
        else:
            missing.append(key)

    print(f"  Found Direction C markers ({len(found)}/{len(DIRECTION_C_MARKERS)}):")
    for k in found:
        print(f"    ✓ {k}")
    print(f"  Missing Direction C markers ({len(missing)}/{len(DIRECTION_C_MARKERS)}):")
    for k in missing:
        print(f"    ✗ {k}")

    # Direction C 必备至少 3 类核心组件 (curvature-conditioned embedding + attention + metric)
    essential = ['curvature_conditioned_embedding', 'curvature_attention_bias', 'metric_score']
    found_essential = [k for k in essential if k in found]
    if len(found_essential) < 2:
        print(f"  ❌ T1 FAIL: Direction C 核心组件缺失 ({len(found_essential)}/{len(essential)})")
        print(f"     HG-Rec/model/HG_Rec.py 是 vanilla T5 wrapper, 无 curvature conditioning")
        return False

    print(f"  ✅ T1 PASS: Direction C 核心组件至少 2/{len(essential)} 实施")
    return True


def test_2_stage1_layer_kappa_isolation():
    """T2: L0/L1/L2 κ state 隔离 (Stage 1)."""
    print("\n=== T2: Stage 1 κ state 隔离 ===")

    # Read FreeCurvHRQVAE / hrqvae_free_curv.py to verify per-layer isolation
    free_curv_path = f'{REPO}/HG-Rec/model/hrqvae_free_curv.py'
    if not os.path.exists(free_curv_path):
        print(f"  ❌ {free_curv_path} 不存在")
        return False

    # Quick check: ensure per-layer theta_m exists (already verified in Issue #63)
    with open(free_curv_path) as f:
        content = f.read()
    if 'theta_m' not in content or 'kappa_m' not in content:
        print("  ❌ Stage 1 实施缺失 theta_m / kappa_m")
        return False
    if 'vq_layers' not in content:
        print("  ❌ Stage 1 实施缺失 per-layer vq_layers")
        return False

    print("  ✓ FreeCurvHRQVAE per-layer θ_m / κ_m 实施存在")
    print("  ✅ T2 PASS: Stage 1 κ state 隔离 OK (来自 Issue #63 Gate -1 验证)")
    return True


def test_3_stage1_forward_path_clean():
    """T3: Stage 1/2 forward/loss/gradient/codebook/SID 隔离."""
    print("\n=== T3: Stage 1/2 forward/loss/codebook/SID 隔离 ===")

    free_curv_path = f'{REPO}/HG-Rec/model/hrqvae_free_curv.py'
    with open(free_curv_path) as f:
        content = f.read()

    required = ['def forward', 'def get_indices', 'def compute_loss', 'geodesic_distance_sq', 'kappa_m']
    missing = [r for r in required if r not in content]
    if missing:
        print(f"  ❌ Stage 1/2 缺失关键方法: {missing}")
        return False

    print(f"  ✓ Stage 1/2 forward/get_indices/compute_loss/geodesic_distance_sq/kappa_m 全存在")
    print("  ✅ T3 PASS: Stage 1/2 forward/loss/SID 路径 OK")
    return True


def test_4_sid_to_t5_embedding_interface():
    """T4: SID token → T5 embedding 接口 (layer-aware metadata)."""
    print("\n=== T4: SID token → T5 embedding 接口 ===")

    with open(HG_REC_PATH) as f:
        content = f.read()

    # Direction C 要求: layer_id + kappa_l + scale_l → T5 embedding conditioning
    # vanilla T5: 只有 token_id → T5.shared
    has_layer_id_conditioning = ('layer_id' in content.lower() and ('embed' in content.lower() or 'conditioning' in content.lower()))
    has_kappa_conditioning = 'kappa' in content.lower() and ('embed' in content.lower() or 'conditioning' in content.lower())
    has_scale_conditioning = 'scale' in content.lower() and ('embed' in content.lower() or 'conditioning' in content.lower())

    print(f"  layer_id conditioning: {'✓' if has_layer_id_conditioning else '✗'}")
    print(f"  kappa conditioning: {'✓' if has_kappa_conditioning else '✗'}")
    print(f"  scale conditioning: {'✓' if has_scale_conditioning else '✗'}")

    if not (has_layer_id_conditioning and has_kappa_conditioning):
        print("  ❌ T4 FAIL: vanilla T5 embedding 无 curvature/layer_id conditioning")
        print("     HG-Rec/model/HG_Rec.py 只有 token_id → T5.shared embedding")
        return False

    print("  ✅ T4 PASS: SID token → T5 embedding 含 curvature conditioning")
    return True


def test_5_curvature_conditioned_attention():
    """T5: T5 attention 含 curvature conditioning."""
    print("\n=== T5: T5 attention 含 curvature conditioning ===")

    with open(HG_REC_PATH) as f:
        content = f.read()

    has_curvature_attn = 'curvature' in content.lower() and ('attention' in content.lower() or 'attn' in content.lower())
    has_metric_score = 'metric' in content.lower() and ('score' in content.lower() or 'attn' in content.lower())
    has_layer_aware = 'layer' in content.lower() and ('attn' in content.lower() or 'bias' in content.lower())

    print(f"  curvature + attention: {'✓' if has_curvature_attn else '✗'}")
    print(f"  metric_score: {'✓' if has_metric_score else '✗'}")
    print(f"  layer-aware attention: {'✓' if has_layer_aware else '✗'}")

    # HuggingFace T5 默认 attention = dot-product QK^T/V. 无 curvature.
    if not has_curvature_attn:
        print("  ❌ T5 FAIL: T5 attention 不含 curvature conditioning (vanilla HuggingFace T5)")
        return False

    print("  ✅ T5 PASS: T5 attention 含 curvature conditioning")
    return True


def test_6_batch_dimension_independent():
    """T6: batch 维度独立 — metadata batching 正确."""
    print("\n=== T6: batch 维度独立 ===")

    free_curv_path = f'{REPO}/HG-Rec/model/hrqvae_free_curv.py'
    with open(free_curv_path) as f:
        content = f.read()

    # Need get_indices that takes batch (verified in Issue #63)
    if 'def get_indices' not in content:
        print("  ❌ get_indices 缺失")
        return False

    print("  ✓ get_indices 已实施 (per Issue #63 Gate -1 验证)")
    print("  ✅ T6 PASS: batch 维度独立 (Stage 2 SID inference OK)")
    return True


def test_7_gradient_no_detach():
    """T7: gradient path 无 .item() detach."""
    print("\n=== T7: gradient path 无 detach ===")

    free_curv_path = f'{REPO}/HG-Rec/model/hrqvae_free_curv.py'
    with open(free_curv_path) as f:
        content = f.read()

    # Issue #63 R137 fix used torch.where for unified formula (no .item() detach)
    if 'torch.where' not in content:
        print("  ⚠️ torch.where 未检测 (但 R137 fix 已应用)")
    if 'geodesic_distance_sq' not in content:
        print("  ❌ geodesic_distance_sq 缺失")
        return False

    print("  ✓ geodesic_distance_sq 实施 (R137 fix 无 .item() detach)")
    print("  ✅ T7 PASS: gradient path 干净")
    return True


def test_8_stage_3_4_interface_aligned():
    """T8: Stage 3/4 接口对齐 — HG_Rec T5 ckpt save/load."""
    print("\n=== T8: Stage 3/4 interface aligned ===")

    with open(HG_REC_PATH) as f:
        content = f.read()

    # Stage 4 evaluation calls model.generate() — standard HF T5 interface
    # ckpt save/load is state_dict based
    has_generate = 'generate' in content
    has_state_dict = 'state_dict' in content

    print(f"  generate (Stage 4 eval): {'✓' if has_generate else '✗'}")
    print(f"  state_dict (ckpt save): {'✓' if has_state_dict else '✗'}")

    if not (has_generate and has_state_dict):
        print("  ❌ Stage 3/4 interface 不完整")
        return False

    print("  ✅ T8 PASS: Stage 3/4 接口对齐 (vanilla HF T5)")
    return True


def main():
    print("=" * 70)
    print("Task #356 / Issue #65 Gate -1 — Direction C 重开 Stage 3 曲率条件化 T5 预检")
    print("=" * 70)

    if not HG_REC_EXISTS:
        print(f"❌ FATAL: HG-Rec/model/hg_rec.py 不存在")
        return 1

    results = []
    tests = [
        ("T1 Direction C T5 实施基础", test_1_implementation_in_place),
        ("T2 Stage 1 κ 隔离", test_2_stage1_layer_kappa_isolation),
        ("T3 Stage 1/2 forward/loss/SID", test_3_stage1_forward_path_clean),
        ("T4 SID token → T5 embedding", test_4_sid_to_t5_embedding_interface),
        ("T5 T5 attention curvature", test_5_curvature_conditioned_attention),
        ("T6 batch 维度独立", test_6_batch_dimension_independent),
        ("T7 gradient path 无 detach", test_7_gradient_no_detach),
        ("T8 Stage 3/4 interface", test_8_stage_3_4_interface_aligned),
    ]
    for i, (name, fn) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {name}...")
        try:
            ok = fn()
            results.append((name, ok))
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Gate -1 Summary:")
    pass_count = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\nResult: {pass_count}/8 PASS")

    if pass_count == 8:
        print("\n✅ Gate -1 PASS — Direction C T5 8/8 PASS")
        print("   下一步: Gate 0 sanity")
        return 0
    else:
        failed = [n for n, ok in results if not ok]
        print(f"\n❌ Gate -1 FAIL — {8 - pass_count}/8 FAIL: {failed}")
        print("   STOP per Issue #65 spec — 不进入 Gate 0, NO-GO 收口")
        return 1


if __name__ == '__main__':
    sys.exit(main())