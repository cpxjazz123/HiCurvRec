# R5_04 TDD record

## RED

Command (run against the pinned R4-08 parent):

    R5_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_04/test_r5_04.py

Exit code: 1

```
FAIL: test_red_blockwise_flag_and_helper: blockwise flag absent on parent
FAIL: test_red_apply_loops_blocks: 
PASS: test_single_block_math
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_04/test_r5_04.py

Exit code: 0

```
PASS: test_red_blockwise_flag_and_helper
PASS: test_red_apply_loops_blocks
PASS: test_single_block_math
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

```diff
--- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
+++ /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r5_04/train_rqvae.py	2026-10-18 20:54:29.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r5_04"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "r4_08":
+if EXPERIMENT_ARM != "r5_04":
     raise RuntimeError(
-        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r5_04' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -451,6 +451,7 @@
 R4_08_PCGRAD_ENC   = True
 R4_08_EPS          = 1e-12
 R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+R5_04_BLOCKWISE = True  # issue #64: per-Linear block PCGrad
 
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
@@ -904,6 +905,33 @@
     return pooled
 
 
+def _r5_04_linear_blocks(encoder_module, encoder_parameters):
+    """Group each nn.Linear's weight+bias as one block; leftover params stay alone."""
+    param_ids = {id(p): i for i, p in enumerate(encoder_parameters)}
+    used = set()
+    blocks = []
+    modules = []
+    if encoder_module is not None:
+        modules = list(encoder_module.modules())
+    for module in modules:
+        if not isinstance(module, torch.nn.Linear):
+            continue
+        idxs = []
+        for tensor in (module.weight, module.bias):
+            if tensor is None:
+                continue
+            idx = param_ids.get(id(tensor))
+            if idx is not None and idx not in used:
+                idxs.append(idx)
+                used.add(idx)
+        if idxs:
+            blocks.append(idxs)
+    for i in range(len(encoder_parameters)):
+        if i not in used:
+            blocks.append([i])
+    return blocks
+
+
 def _r4_08_apply_projection(
     base_loss: torch.Tensor,
     l1_loss: torch.Tensor,
@@ -911,15 +939,11 @@
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
+    """R5-04: same PCGrad rule, applied independently per Linear weight+bias block."""
     encoder_parameters = list(encoder_parameters)
     if not encoder_parameters:
         return
@@ -934,18 +958,17 @@
         )
     else:
         g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
-    if R4_08_PCGRAD_ENC:
-        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
-        flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
-        flat_proj = r4_project_conflict(flat_l1, flat_base)
-        g_l1_proj = []
-        offset = 0
-        for grad in g_l1:
-            count = grad.numel()
-            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
-            offset += count
-    else:
-        g_l1_proj = g_l1
+    g_l1_proj = [g.clone() for g in g_l1]
+    if R4_08_PCGRAD_ENC and R5_04_BLOCKWISE:
+        for block in _r5_04_linear_blocks(encoder_module, encoder_parameters):
+            flat_l1 = torch.cat([g_l1[i].reshape(-1) for i in block])
+            flat_base = torch.cat([g_base[i].reshape(-1) for i in block])
+            flat_proj = r4_project_conflict(flat_l1, flat_base)
+            offset = 0
+            for i in block:
+                count = g_l1[i].numel()
+                g_l1_proj[i] = flat_proj[offset : offset + count].view_as(g_l1[i])
+                offset += count
     scale = float(behavior_weight) / float(max(valid_levels, 1))
     for parameter, base_grad, l1_proj_grad, other_grad in zip(
         encoder_parameters, g_base, g_l1_proj, g_other
@@ -1529,9 +1552,6 @@
                     parameter.square().mean() for parameter in _crowd_control.parameters()
                 )
             if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
-                # One backward still produces the baseline gradient for every parameter,
-                # including the decoder and the codebooks. Only then is the encoder
-                # gradient replaced by the conflict-projected assembly.
                 loss.backward(retain_graph=True)
                 _r4_08_apply_projection(
                     base_loss,
@@ -1540,6 +1560,7 @@
                     list(raw_module.encoder.parameters()),
                     epoch_weight,
                     valid_levels=behavior_valid_levels,
+                    encoder_module=raw_module.encoder,
                 )
             else:
                 loss.backward()

```
