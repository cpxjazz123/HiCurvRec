# R5_03 TDD record

## RED

Command (run against the pinned R4-08 parent):

    R5_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_03/test_r5_03.py

Exit code: 1

```
FAIL: test_red_quant_flag_active: quant-reference flag absent on parent
FAIL: test_red_apply_builds_quant_reference: 
PASS: test_project_formula
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_03/test_r5_03.py

Exit code: 0

```
PASS: test_red_quant_flag_active
PASS: test_red_apply_builds_quant_reference
PASS: test_project_formula
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

```diff
--- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
+++ /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r5_03/train_rqvae.py	2026-10-18 20:54:29.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r5_03"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "r4_08":
+if EXPERIMENT_ARM != "r5_03":
     raise RuntimeError(
-        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r5_03' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -451,6 +451,7 @@
 R4_08_PCGRAD_ENC   = True
 R4_08_EPS          = 1e-12
 R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+R5_03_QUANT_REF = True  # issue #63: project L1 vs quant-only
 
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
@@ -911,22 +912,24 @@
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
+    """R5-03: project g_L1 against g_quant = grad(base - recon); assemble with full g_base."""
     encoder_parameters = list(encoder_parameters)
     if not encoder_parameters:
         return
+    if recon_loss is None:
+        raise RuntimeError("R5-03 requires recon_loss to form the quant reference")
     has_other = other_loss is not None and bool(other_loss.requires_grad)
     g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
     g_l1 = _r4_08_global_average(
-        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
+        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=True)
+    )
+    quant_loss = base_loss - recon_loss
+    g_ref = _r4_08_global_average(
+        _r4_08_encoder_gradients(quant_loss, encoder_parameters, retain=has_other)
     )
     if has_other:
         g_other = _r4_08_global_average(
@@ -934,10 +937,10 @@
         )
     else:
         g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
-    if R4_08_PCGRAD_ENC:
+    if R4_08_PCGRAD_ENC and R5_03_QUANT_REF:
         flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
-        flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
-        flat_proj = r4_project_conflict(flat_l1, flat_base)
+        flat_ref = torch.cat([grad.reshape(-1) for grad in g_ref])
+        flat_proj = r4_project_conflict(flat_l1, flat_ref)
         g_l1_proj = []
         offset = 0
         for grad in g_l1:
@@ -1529,9 +1532,6 @@
                     parameter.square().mean() for parameter in _crowd_control.parameters()
                 )
             if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
-                # One backward still produces the baseline gradient for every parameter,
-                # including the decoder and the codebooks. Only then is the encoder
-                # gradient replaced by the conflict-projected assembly.
                 loss.backward(retain_graph=True)
                 _r4_08_apply_projection(
                     base_loss,
@@ -1540,6 +1540,7 @@
                     list(raw_module.encoder.parameters()),
                     epoch_weight,
                     valid_levels=behavior_valid_levels,
+                    recon_loss=recon_loss,
                 )
             else:
                 loss.backward()

```
