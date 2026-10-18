# R5_09 TDD record

## RED

Command (run against the pinned R4-08 parent):

    R5_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_09/test_r5_09.py

Exit code: 1

```
FAIL: test_red_ema_flag_and_helper: EMA decay missing on parent
FAIL: test_ema_two_step: EMA state/helper missing on parent
FAIL: test_apply_uses_ema_reference: 
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
3 test(s) failed
```

Behavioural failures observed on the parent: 3

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_09/test_r5_09.py

Exit code: 0

```
PASS: test_red_ema_flag_and_helper
PASS: test_ema_two_step
PASS: test_apply_uses_ema_reference
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

```diff
--- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
+++ /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r5_09/train_rqvae.py	2026-10-18 20:54:29.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r5_09"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "r4_08":
+if EXPERIMENT_ARM != "r5_09":
     raise RuntimeError(
-        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r5_09' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -451,6 +451,8 @@
 R4_08_PCGRAD_ENC   = True
 R4_08_EPS          = 1e-12
 R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+R5_09_EMA_DECAY = 0.9  # issue #69: EMA-smoothed base as PCGrad reference
+R5_09_EMA_STATE = {"buffers": None}
 
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
@@ -904,6 +906,20 @@
     return pooled
 
 
+def _r5_09_update_ema(g_base):
+    """Maintain m_t = 0.9 m_{t-1} + 0.1 g_base_t; first step seeds with g_base."""
+    buffers = R5_09_EMA_STATE["buffers"]
+    if buffers is None:
+        R5_09_EMA_STATE["buffers"] = [g.detach().clone() for g in g_base]
+        return R5_09_EMA_STATE["buffers"]
+    decay = float(R5_09_EMA_DECAY)
+    updated = []
+    for prev, cur in zip(buffers, g_base):
+        updated.append(decay * prev + (1.0 - decay) * cur.detach())
+    R5_09_EMA_STATE["buffers"] = updated
+    return updated
+
+
 def _r4_08_apply_projection(
     base_loss: torch.Tensor,
     l1_loss: torch.Tensor,
@@ -911,15 +927,11 @@
     encoder_parameters,
     behavior_weight: float,
     valid_levels: int = 3,
+    recon_loss: torch.Tensor | None = None,
+    epoch: int | None = None,
+    encoder_module=None,
 ) -> None:
-    """Overwrite the encoder gradients with the conflict-projected assembly.
-
-    ``g_base + w_b/3 * (g'_L1 + g_L2 + g_L3)`` with the averaging the loop actually uses:
-    the behaviour term contributes ``behavior_weight / valid_levels * (L1 + L2 + L3)``,
-    and ``g'_L1`` is the single PCGrad projection of the full encoder ``g_L1`` vector
-    against the full encoder ``g_base`` vector. Decoder and codebook gradients are never
-    touched.
-    """
+    """R5-09: project g_L1 against EMA(g_base); assemble with current unprojected g_base."""
     encoder_parameters = list(encoder_parameters)
     if not encoder_parameters:
         return
@@ -934,10 +946,11 @@
         )
     else:
         g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
+    g_ref = _r5_09_update_ema(g_base)
     if R4_08_PCGRAD_ENC:
         flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
-        flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
-        flat_proj = r4_project_conflict(flat_l1, flat_base)
+        flat_ref = torch.cat([grad.reshape(-1) for grad in g_ref])
+        flat_proj = r4_project_conflict(flat_l1, flat_ref)
         g_l1_proj = []
         offset = 0
         for grad in g_l1:

```
