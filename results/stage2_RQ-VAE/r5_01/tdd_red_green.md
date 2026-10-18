# R5_01 TDD record

## RED

Command (run against the pinned R4-08 parent):

    R5_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_01/test_r5_01.py

Exit code: 1

```
FAIL: test_red_reverse_flag_active: reverse PCGrad flag absent — parent still projects L1 against base
FAIL: test_red_apply_projects_base_operand: apply_projection does not project g_base against g_L1
PASS: test_helper_reverse_call_orthogonalizes_base
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_01/test_r5_01.py

Exit code: 0

```
PASS: test_red_reverse_flag_active
PASS: test_red_apply_projects_base_operand
PASS: test_helper_reverse_call_orthogonalizes_base
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

```diff
--- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
+++ /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r5_01/train_rqvae.py	2026-10-18 21:00:51.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r5_01"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "r4_08":
+if EXPERIMENT_ARM != "r5_01":
     raise RuntimeError(
-        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r5_01' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -451,6 +451,7 @@
 R4_08_PCGRAD_ENC   = True
 R4_08_EPS          = 1e-12
 R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+R5_01_REVERSE_PCGRAD = True  # issue #61: protect L1, project base
 
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
@@ -856,23 +857,23 @@
         return sum(parameter.numel() for parameter in self.parameters())
 
 
-def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
-    """PCGrad step for one parameter: drop g_base when the two gradients conflict.
+def r4_project_conflict(g_operand: torch.Tensor, g_reference: torch.Tensor) -> torch.Tensor:
+    """R5-01 reverse PCGrad: project the *base* operand against the L1 reference.
 
-    ``g_l1 <- g_l1 - (<g_l1, g_base> / (||g_base||^2 + eps)) * g_base`` when the dot
-    product is negative, otherwise g_l1 passes through untouched. A zero or non-finite
-    base gradient disables the projection and fails closed rather than emitting NaNs.
+    When called as ``r4_project_conflict(g_base, g_l1)`` this removes the L1-conflicting
+    component from g_base. The helper keeps the R4-08 name so the call sites and the
+    RED suite still locate a single projection entry point.
     """
-    if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
-        raise RuntimeError("R4-08 received a non-finite gradient; failing closed")
-    base_sq = float((g_base * g_base).sum())
-    if base_sq <= 0.0:
-        return g_l1
-    dot = float((g_l1 * g_base).sum())
+    if not bool(torch.isfinite(g_operand).all()) or not bool(torch.isfinite(g_reference).all()):
+        raise RuntimeError("R5-01 received a non-finite gradient; failing closed")
+    ref_sq = float((g_reference * g_reference).sum())
+    if ref_sq <= 0.0:
+        return g_operand
+    dot = float((g_operand * g_reference).sum())
     if dot >= 0.0:
-        return g_l1
+        return g_operand
     R4_08_PROJECTION_TRACE["active_count"] += 1
-    return g_l1 - (dot / (base_sq + R4_08_EPS)) * g_base
+    return g_operand - (dot / (ref_sq + R4_08_EPS)) * g_reference
 
 
 def _r4_08_encoder_gradients(loss: torch.Tensor, parameters, retain: bool = True):
@@ -911,15 +912,11 @@
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
+    """R5-01: assemble g'_base + scale*(g_L1 + g_other); only g_base is projected."""
     encoder_parameters = list(encoder_parameters)
     if not encoder_parameters:
         return
@@ -934,23 +931,23 @@
         )
     else:
         g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
-    if R4_08_PCGRAD_ENC:
-        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
+    if R4_08_PCGRAD_ENC and R5_01_REVERSE_PCGRAD:
         flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
-        flat_proj = r4_project_conflict(flat_l1, flat_base)
-        g_l1_proj = []
+        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
+        flat_proj = r4_project_conflict(flat_base, flat_l1)
+        g_base_proj = []
         offset = 0
-        for grad in g_l1:
+        for grad in g_base:
             count = grad.numel()
-            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
+            g_base_proj.append(flat_proj[offset : offset + count].view_as(grad))
             offset += count
     else:
-        g_l1_proj = g_l1
+        g_base_proj = g_base
     scale = float(behavior_weight) / float(max(valid_levels, 1))
-    for parameter, base_grad, l1_proj_grad, other_grad in zip(
-        encoder_parameters, g_base, g_l1_proj, g_other
+    for parameter, base_grad, l1_grad, other_grad in zip(
+        encoder_parameters, g_base_proj, g_l1, g_other
     ):
-        assembled = base_grad + scale * (l1_proj_grad + other_grad)
+        assembled = base_grad + scale * (l1_grad + other_grad)
         if parameter.grad is None:
             parameter.grad = assembled
         else:

```
