# R4_03 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_03/test_r4_03.py

Exit code: 1

```
FAIL: test_red_weighted_rows_match_hand_calculation: L1 rows are not frequency weighted (actual 14.15513611, expected 13.83658314)
PASS: test_red_equal_anchor_frequencies_give_baseline
PASS: test_red_frequent_anchor_contributes_less_fractionally
PASS: test_off_flag_matches_exact_resample
FAIL: test_on_flag_has_measurable_effect: enabled mechanism is a silent no-op
PASS: test_finite_forward_and_backward
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_03/test_r4_03.py

Exit code: 0

```
PASS: test_red_weighted_rows_match_hand_calculation
PASS: test_red_equal_anchor_frequencies_give_baseline
PASS: test_red_frequent_anchor_contributes_less_fractionally
PASS: test_off_flag_matches_exact_resample
PASS: test_on_flag_has_measurable_effect
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
+++ stage2_RQ-VAE/TIGER_r4_03/train_rqvae.py	2026-10-18 18:26:17.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_03"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_03":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_03' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
@@ -442,6 +442,17 @@
 PAIR_BATCH_SIZE_PER_RANK = 512
 RESAMPLE_PAIRS          = True
 
+# === R4-03 (issue #53): inverse-anchor-frequency row weighting of L1 CE ===
+# Repeated high-frequency source items dominate the row-wise L1 cross entropy. With a
+# frozen train-only count f(a) and the fixed power -0.5, w(a) = (1 + f(a))**-0.5, each
+# row is weighted before the reduction and the weights are renormalised, so the scale
+# of the L1 term is unchanged and L2/L3, the masks, the logits and the sampler are all
+# byte-for-byte the baseline's.
+R4_03_ANCHOR_WEIGHTING = True
+R4_03_WEIGHT_POWER     = -0.5
+R4_03_ANCHOR_FREQ     = None    # filled in main() from train.parquet; frozen thereafter
+MECHANISM_TRAINABLE_PARAMETERS = ()
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -846,6 +857,57 @@
         return sum(parameter.numel() for parameter in self.parameters())
 
 
+def _duplicate_mask(
+    source_ids: torch.Tensor,
+    target_ids: torch.Tensor,
+    successor_indicator,
+    device: torch.device,
+) -> torch.Tensor:
+    """The baseline's duplicate / false-negative mask, factored out unchanged."""
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
+def _r4_03_row_weights(source_ids: torch.Tensor, device: torch.device):
+    """w(a) = (1 + f(a))**-0.5 from the frozen train-only anchor frequency."""
+    if not R4_03_ANCHOR_WEIGHTING or R4_03_ANCHOR_FREQ is None:
+        return None
+    frequency = R4_03_ANCHOR_FREQ.to(device=device, dtype=torch.float32)
+    counts = frequency[source_ids.to(device)]
+    weights = torch.pow(1.0 + counts, R4_03_WEIGHT_POWER)
+    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
+        raise RuntimeError("R4-03 anchor weights are non-finite or non-positive")
+    return weights
+
+
+def _weighted_row_ce(
+    logits: torch.Tensor,
+    source_ids: torch.Tensor,
+    target_ids: torch.Tensor,
+    successor_indicator,
+    weights: torch.Tensor,
+) -> torch.Tensor | None:
+    """Per-row CE under the baseline mask, then the prescribed weighted mean."""
+    mask = _duplicate_mask(source_ids, target_ids, successor_indicator, logits.device)
+    valid = source_ids.ne(target_ids)
+    if not bool(valid.any()):
+        return None
+    labels = torch.arange(int(source_ids.shape[0]), device=source_ids.device)
+    rows = F.cross_entropy(
+        logits.masked_fill(mask, -torch.inf)[valid], labels[valid], reduction="none"
+    )
+    selected = weights[valid]
+    return (selected * rows).sum() / selected.sum()
+
+
 def behavior_contrastive_loss(
     residuals: torch.Tensor,
     source_ids: torch.Tensor,
@@ -859,6 +921,15 @@
         source = residuals[level, :batch_size]
         candidates = residuals[level, batch_size:]
         logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
+        if level == 0:
+            weights = _r4_03_row_weights(source_ids, logits.device)
+            if weights is not None:
+                loss = _weighted_row_ce(
+                    logits, source_ids, target_ids, successor_indicator, weights
+                )
+                if loss is not None:
+                    losses.append(loss)
+                continue
         loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
         if loss is not None:
             losses.append(loss)
@@ -957,6 +1028,26 @@
         geometry_item_bucket_t = torch.as_tensor(geometry_item_bucket, device=device)
     all_embeddings = torch.from_numpy(embeddings)
     train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
+    global R4_03_ANCHOR_FREQ
+    if R4_03_ANCHOR_WEIGHTING:
+        sources = np.asarray(
+            [int(history[-1]) for history in train_frame["seen_history"].to_numpy()
+             if history is not None and len(history) > 0],
+            dtype=np.int64,
+        )
+        R4_03_ANCHOR_FREQ = torch.tensor(
+            np.bincount(sources, minlength=len(embeddings))[: len(embeddings)].astype(np.float32)
+        )
+        if rank == 0:
+            _record(
+                rank,
+                "r4_03_anchor_frequency",
+                mechanism="inverse_anchor_frequency_row_weighting",
+                weight_power=R4_03_WEIGHT_POWER,
+                n_rows=int(sources.shape[0]),
+                f_max=float(R4_03_ANCHOR_FREQ.max()),
+                requires_grad=bool(R4_03_ANCHOR_FREQ.requires_grad),
+            )
     # iter48's full transition stream, drawn by its own sampler so the item
     # batch above keeps exactly the rows, order and Sinkhorn balancing the
     # baseline sees.

```

