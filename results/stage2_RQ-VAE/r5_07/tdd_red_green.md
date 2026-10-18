# R5_07 TDD record

## RED

Command (run against the pinned R4-08 parent):

    R5_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_07/test_r5_07.py

Exit code: 1

```
FAIL: test_red_weak_conflict_skipped: weak conflict projected (parent would project)
PASS: test_red_strong_conflict_projected
FAIL: test_threshold_flag: 
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_07/test_r5_07.py

Exit code: 0

```
PASS: test_red_weak_conflict_skipped
PASS: test_red_strong_conflict_projected
PASS: test_threshold_flag
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

```diff
--- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
+++ /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r5_07/train_rqvae.py	2026-10-18 20:54:29.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r5_07"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "r4_08":
+if EXPERIMENT_ARM != "r5_07":
     raise RuntimeError(
-        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r5_07' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -451,6 +451,7 @@
 R4_08_PCGRAD_ENC   = True
 R4_08_EPS          = 1e-12
 R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+R5_07_COS_THRESHOLD = -0.2  # issue #67: project only for strong conflicts
 
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
@@ -857,19 +858,17 @@
 
 
 def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
-    """PCGrad step for one parameter: drop g_base when the two gradients conflict.
-
-    ``g_l1 <- g_l1 - (<g_l1, g_base> / (||g_base||^2 + eps)) * g_base`` when the dot
-    product is negative, otherwise g_l1 passes through untouched. A zero or non-finite
-    base gradient disables the projection and fails closed rather than emitting NaNs.
-    """
+    """R5-07: apply the original R4-08 projection only when cos(g_L1,g_base) < -0.2."""
     if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
-        raise RuntimeError("R4-08 received a non-finite gradient; failing closed")
+        raise RuntimeError("R5-07 received a non-finite gradient; failing closed")
     base_sq = float((g_base * g_base).sum())
-    if base_sq <= 0.0:
+    l1_sq = float((g_l1 * g_l1).sum())
+    if base_sq <= 0.0 or l1_sq <= 0.0:
         return g_l1
     dot = float((g_l1 * g_base).sum())
-    if dot >= 0.0:
+    cos = dot / ((base_sq ** 0.5) * (l1_sq ** 0.5) + R4_08_EPS)
+    # Strict inequality: cos == -0.2 must NOT project.
+    if cos >= R5_07_COS_THRESHOLD:
         return g_l1
     R4_08_PROJECTION_TRACE["active_count"] += 1
     return g_l1 - (dot / (base_sq + R4_08_EPS)) * g_base

```
