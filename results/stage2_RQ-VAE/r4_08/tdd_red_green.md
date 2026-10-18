# R4_08 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_08/test_r4_08.py

Exit code: 1

```
FAIL: test_red_conflicting_l1_gradient_becomes_orthogonal_to_base: projected L1 gradient still conflicts with base (dot=-1.000000)
FAIL: test_red_conflict_projection_differs_from_the_plain_sum: projection left the L1 gradient unchanged
PASS: test_non_conflicting_gradients_pass_through_unchanged
PASS: test_zero_base_gradient_is_not_projected
FAIL: test_non_finite_inputs_fail_closed: conflict projection is not implemented
FAIL: test_off_flag_matches_exact_resample: enabled mechanism is a silent no-op on the encoder gradient
PASS: test_only_encoder_parameters_are_projected
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
4 test(s) failed
```

Behavioural failures observed on the parent: 4

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_08/test_r4_08.py

Exit code: 0

```
PASS: test_red_conflicting_l1_gradient_becomes_orthogonal_to_base
PASS: test_red_conflict_projection_differs_from_the_plain_sum
PASS: test_non_conflicting_gradients_pass_through_unchanged
PASS: test_zero_base_gradient_is_not_projected
PASS: test_non_finite_inputs_fail_closed
PASS: test_off_flag_matches_exact_resample
PASS: test_only_encoder_parameters_are_projected
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

### Files

```
train_rqvae.py
```

### Diff

```diff
--- stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/train_rqvae.py	2026-10-17 13:07:42.000000000 +1100
+++ stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_08":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -208,7 +208,7 @@
 REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
 OUT_ROOT  = REPO_ROOT / "results/stage2_RQ-VAE"
 _ARM_SEED = {"resample_s43": 43, "resample_s44": 44}.get(EXPERIMENT_ARM, 42)
-_OUT = OUT_ROOT / TREE_NAME / EXPERIMENT_ARM
+_OUT = OUT_ROOT / EXPERIMENT_ARM
 OUTPUT_SID     = _OUT / "sids_for_hgrec.npy"
 OUTPUT_JSON    = _OUT / "item_sids.json"
 CHECKPOINT     = _OUT / "rqvae_best.pth"
@@ -442,6 +442,16 @@
 PAIR_BATCH_SIZE_PER_RANK = 512
 RESAMPLE_PAIRS          = True
 
+# === R4-08 (issue #58): encoder-only conflict projection of the L1 gradient ===
+# The L1 behaviour gradient can point against the reconstruction gradient. Projecting
+# ONLY that conflict away, and only on encoder parameters, changes the direction of the
+# update without adding an objective, a weight or a projection head. The forward CE
+# losses, the decoder and codebook updates, the optimizer hyperparameters and every
+# other gradient component stay exactly as the baseline computes them.
+R4_08_PCGRAD_ENC   = True
+R4_08_EPS          = 1e-12
+R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -846,6 +856,141 @@
         return sum(parameter.numel() for parameter in self.parameters())
 
 
+def r4_project_conflict(g_l1: torch.Tensor, g_base: torch.Tensor) -> torch.Tensor:
+    """PCGrad step for one parameter: drop g_base when the two gradients conflict.
+
+    ``g_l1 <- g_l1 - (<g_l1, g_base> / (||g_base||^2 + eps)) * g_base`` when the dot
+    product is negative, otherwise g_l1 passes through untouched. A zero or non-finite
+    base gradient disables the projection and fails closed rather than emitting NaNs.
+    """
+    if not bool(torch.isfinite(g_l1).all()) or not bool(torch.isfinite(g_base).all()):
+        raise RuntimeError("R4-08 received a non-finite gradient; failing closed")
+    base_sq = float((g_base * g_base).sum())
+    if base_sq <= 0.0:
+        return g_l1
+    dot = float((g_l1 * g_base).sum())
+    if dot >= 0.0:
+        return g_l1
+    R4_08_PROJECTION_TRACE["active_count"] += 1
+    return g_l1 - (dot / (base_sq + R4_08_EPS)) * g_base
+
+
+def _r4_08_encoder_gradients(loss: torch.Tensor, parameters, retain: bool = True):
+    """Gradient of one loss term on the encoder parameters only."""
+    grads = torch.autograd.grad(
+        loss,
+        parameters,
+        retain_graph=retain,
+        allow_unused=True,
+    )
+    return [
+        torch.zeros_like(parameter) if grad is None else grad
+        for parameter, grad in zip(parameters, grads)
+    ]
+
+
+def _r4_08_global_average(grads):
+    """Pool per-rank gradients so every rank projects the same global vectors."""
+    if not (dist.is_available() and dist.is_initialized()):
+        return grads
+    world_size = dist.get_world_size()
+    if world_size <= 1:
+        return grads
+    pooled = []
+    for grad in grads:
+        reduced = grad.clone()
+        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
+        pooled.append(reduced / world_size)
+    return pooled
+
+
+def _r4_08_apply_projection(
+    base_loss: torch.Tensor,
+    l1_loss: torch.Tensor,
+    other_loss: torch.Tensor | None,
+    encoder_parameters,
+    behavior_weight: float,
+    valid_levels: int = 3,
+) -> None:
+    """Overwrite the encoder gradients with the conflict-projected assembly.
+
+    ``g_base + w_b/3 * (g'_L1 + g_L2 + g_L3)`` with the averaging the loop actually uses:
+    the behaviour term contributes ``behavior_weight / valid_levels * (L1 + L2 + L3)``,
+    and ``g'_L1`` is the single PCGrad projection of the full encoder ``g_L1`` vector
+    against the full encoder ``g_base`` vector. Decoder and codebook gradients are never
+    touched.
+    """
+    encoder_parameters = list(encoder_parameters)
+    if not encoder_parameters:
+        return
+    has_other = other_loss is not None and bool(other_loss.requires_grad)
+    g_base = _r4_08_global_average(_r4_08_encoder_gradients(base_loss, encoder_parameters))
+    g_l1 = _r4_08_global_average(
+        _r4_08_encoder_gradients(l1_loss, encoder_parameters, retain=has_other)
+    )
+    if has_other:
+        g_other = _r4_08_global_average(
+            _r4_08_encoder_gradients(other_loss, encoder_parameters, retain=False)
+        )
+    else:
+        g_other = [torch.zeros_like(parameter) for parameter in encoder_parameters]
+    if R4_08_PCGRAD_ENC:
+        flat_l1 = torch.cat([grad.reshape(-1) for grad in g_l1])
+        flat_base = torch.cat([grad.reshape(-1) for grad in g_base])
+        flat_proj = r4_project_conflict(flat_l1, flat_base)
+        g_l1_proj = []
+        offset = 0
+        for grad in g_l1:
+            count = grad.numel()
+            g_l1_proj.append(flat_proj[offset : offset + count].view_as(grad))
+            offset += count
+    else:
+        g_l1_proj = g_l1
+    scale = float(behavior_weight) / float(max(valid_levels, 1))
+    for parameter, base_grad, l1_proj_grad, other_grad in zip(
+        encoder_parameters, g_base, g_l1_proj, g_other
+    ):
+        assembled = base_grad + scale * (l1_proj_grad + other_grad)
+        if parameter.grad is None:
+            parameter.grad = assembled
+        else:
+            parameter.grad.copy_(assembled)
+
+
+def behavior_contrastive_components(
+    residuals: torch.Tensor,
+    source_ids: torch.Tensor,
+    target_ids: torch.Tensor,
+    successor_indicator=None,
+) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
+    """Split the baseline contrastive term into its L1 and its L2+L3 sum.
+
+    ``(l1_loss + other_loss) / valid_levels`` reproduces ``behavior_contrastive_loss``
+    exactly, which is what the forward pass still uses. Only the gradient assembly
+    changes.
+    """
+    batch_size = int(source_ids.shape[0])
+    l1_loss = None
+    other: list[torch.Tensor] = []
+    for level in ACTIVE_LEVELS:
+        source = residuals[level, :batch_size]
+        candidates = residuals[level, batch_size:]
+        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
+        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
+        if loss is None:
+            continue
+        if level == 0:
+            l1_loss = loss
+        else:
+            other.append(loss)
+    other_loss = torch.stack(other).sum() if other else None
+    valid_levels = (1 if l1_loss is not None else 0) + len(other)
+    return l1_loss, other_loss, valid_levels
+
+
+_r4_08_split_behavior_loss = behavior_contrastive_components
+
+
 def behavior_contrastive_loss(
     residuals: torch.Tensor,
     source_ids: torch.Tensor,
@@ -1152,6 +1297,9 @@
             # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
             base_loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
             behavior_loss = torch.zeros((), device=device)
+            behavior_l1_loss = None
+            behavior_other_loss = None
+            behavior_valid_levels = 0
             crowd_value = torch.zeros((), device=device)
             crowd_metric = torch.zeros((), device=device)
             crowding_weight = epoch_crowding_weight
@@ -1182,9 +1330,25 @@
                         DIRECTIONAL_MODE == "sym", successor_indicator,
                     )
                 else:
-                    behavior_loss = behavior_contrastive_loss(
-                        residuals, source_ids, target_ids, successor_indicator
-                    )
+                    if R4_08_PCGRAD_ENC:
+                        # R4-08 keeps the baseline mean intact and additionally keeps
+                        # the L1 component addressable, so the projection can replace
+                        # exactly that component in the encoder gradient.
+                        behavior_l1_loss, behavior_other_loss, behavior_valid_levels = (
+                            _r4_08_split_behavior_loss(
+                                residuals, source_ids, target_ids, successor_indicator
+                            )
+                        )
+                        if behavior_valid_levels > 0:
+                            total_beh = (
+                                (behavior_l1_loss if behavior_l1_loss is not None else 0.0)
+                                + (behavior_other_loss if behavior_other_loss is not None else 0.0)
+                            )
+                            behavior_loss = total_beh / float(behavior_valid_levels)
+                    else:
+                        behavior_loss = behavior_contrastive_loss(
+                            residuals, source_ids, target_ids, successor_indicator
+                        )
                 if _GEOMETRY_CONFIG is not None:
                     geometry_negative_ids = _matched_geometry_negative_ids(
                         source_ids, target_ids, geometry_bucket_padded_t,
@@ -1364,7 +1528,21 @@
                 loss = loss + 0.0 * sum(
                     parameter.square().mean() for parameter in _crowd_control.parameters()
                 )
-            loss.backward()
+            if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
+                # One backward still produces the baseline gradient for every parameter,
+                # including the decoder and the codebooks. Only then is the encoder
+                # gradient replaced by the conflict-projected assembly.
+                loss.backward(retain_graph=True)
+                _r4_08_apply_projection(
+                    base_loss,
+                    behavior_l1_loss,
+                    behavior_other_loss,
+                    list(raw_module.encoder.parameters()),
+                    epoch_weight,
+                    valid_levels=behavior_valid_levels,
+                )
+            else:
+                loss.backward()
             if directional_heads is not None and epoch_weight > 0.0 and world_size > 1:
                 for parameter in directional_heads.parameters():
                     if parameter.grad is not None:

```

