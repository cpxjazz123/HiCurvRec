# R4_02 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_02/test_r4_02.py

Exit code: 1

```
FAIL: test_red_l1_is_row_column_symmetric_mean: L1 is not the symmetric mean (actual 11.27564240, expected 11.16545868)
FAIL: test_red_reverse_term_actually_differs: symmetric L1 not applied: actual 29.55503654 vs expected 36.69789505
FAIL: test_red_l2_l3_components_unchanged: L2/L3 contribution was disturbed
PASS: test_off_flag_matches_exact_resample
PASS: test_symmetric_uniform_toy_is_a_no_op
PASS: test_finite_forward_and_backward
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
3 test(s) failed
```

Behavioural failures observed on the parent: 3

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_02/test_r4_02.py

Exit code: 0

```
PASS: test_red_l1_is_row_column_symmetric_mean
PASS: test_red_reverse_term_actually_differs
PASS: test_red_l2_l3_components_unchanged
PASS: test_off_flag_matches_exact_resample
PASS: test_symmetric_uniform_toy_is_a_no_op
PASS: test_finite_forward_and_backward
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
+++ stage2_RQ-VAE/TIGER_r4_02/train_rqvae.py	2026-10-18 18:11:13.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_02"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_02":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_02' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
@@ -442,6 +442,13 @@
 PAIR_BATCH_SIZE_PER_RANK = 512
 RESAMPLE_PAIRS          = True
 
+# === R4-02 (issue #52): symmetric row-column L1 InfoNCE ===
+# The baseline normalises each anchor row only. Adding the column-normalised term on
+# the very same score matrix and the very same mask sharpens L1 pairing without a new
+# projection head, a second sampler, extra negatives, a multi-successor label or a
+# new loss weight. L2/L3 are untouched.
+R4_02_SYMMETRIC_L1    = True
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -846,6 +853,53 @@
         return sum(parameter.numel() for parameter in self.parameters())
 
 
+def _duplicate_mask(
+    source_ids: torch.Tensor,
+    target_ids: torch.Tensor,
+    successor_indicator,
+    device: torch.device,
+) -> torch.Tensor:
+    """The baseline's duplicate / false-negative mask, factored out unchanged."""
+    batch_size = int(source_ids.shape[0])
+    duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
+    duplicate_targets.fill_diagonal_(False)
+    if successor_indicator is not None:
+        rows = successor_indicator[source_ids.cpu().numpy()]
+        blocked = torch.from_numpy(
+            np.asarray(rows[:, target_ids.cpu().numpy()].todense())
+        ).to(device=device, dtype=torch.bool)
+        blocked.fill_diagonal_(False)
+        duplicate_targets = duplicate_targets | blocked
+    return duplicate_targets
+
+
+def _symmetric_row_column_ce(
+    source: torch.Tensor,
+    candidates: torch.Tensor,
+    source_ids: torch.Tensor,
+    target_ids: torch.Tensor,
+    successor_indicator=None,
+) -> torch.Tensor | None:
+    """R4-02: 0.5 * CE_rows(S) + 0.5 * CE_rows(S.T) on the SAME tensors and mask.
+
+    The reverse term reuses the same scores transposed and the same mask transposed,
+    so a blocked item stays blocked in both directions and the diagonal positives stay
+    unmasked. No new graph is built and no new object is learned.
+    """
+    batch_size = int(source_ids.shape[0])
+    scores = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
+    mask = _duplicate_mask(source_ids, target_ids, successor_indicator, scores.device)
+    valid = source_ids.ne(target_ids)
+    if not bool(valid.any()):
+        return None
+    labels = torch.arange(batch_size, device=source_ids.device)
+    forward = F.cross_entropy(scores.masked_fill(mask, -torch.inf)[valid], labels[valid])
+    reverse = F.cross_entropy(
+        scores.t().masked_fill(mask.t(), -torch.inf)[valid], labels[valid]
+    )
+    return 0.5 * (forward + reverse)
+
+
 def behavior_contrastive_loss(
     residuals: torch.Tensor,
     source_ids: torch.Tensor,
@@ -858,7 +912,15 @@
     for level in ACTIVE_LEVELS:
         source = residuals[level, :batch_size]
         candidates = residuals[level, batch_size:]
-        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
+        distances = torch.cdist(source, candidates, p=2)
+        logits = -distances / BEHAVIOR_TEMPERATURE
+        if level == 0 and R4_02_SYMMETRIC_L1:
+            loss = _symmetric_row_column_ce(
+                source, candidates, source_ids, target_ids, successor_indicator
+            )
+            if loss is not None:
+                losses.append(loss)
+            continue
         loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
         if loss is not None:
             losses.append(loss)

```

