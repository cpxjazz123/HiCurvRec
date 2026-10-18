# R4_05 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_05/test_r4_05.py

Exit code: 1

```
FAIL: test_red_sources_use_the_one_skip_item_when_available: one-skip anchor not applied: sources [7, 8, 9, 13] != [7, 7, 8, 12]
PASS: test_red_length_one_sequence_falls_back_to_last_item
PASS: test_red_targets_are_untouched
PASS: test_off_flag_matches_exact_resample
FAIL: test_pair_count_and_row_order_preserved: enabled mechanism is a silent no-op
PASS: test_source_is_always_in_the_rows_past_and_never_the_future_target
PASS: test_finite_contrastive_gradients_on_lag2_pairs
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_05/test_r4_05.py

Exit code: 0

```
PASS: test_red_sources_use_the_one_skip_item_when_available
PASS: test_red_length_one_sequence_falls_back_to_last_item
PASS: test_red_targets_are_untouched
PASS: test_off_flag_matches_exact_resample
PASS: test_pair_count_and_row_order_preserved
PASS: test_source_is_always_in_the_rows_past_and_never_the_future_target
PASS: test_finite_contrastive_gradients_on_lag2_pairs
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
+++ stage2_RQ-VAE/TIGER_r4_05/train_rqvae.py	2026-10-18 18:11:13.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_05"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_05":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_05' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
@@ -442,6 +442,15 @@
 PAIR_BATCH_SIZE_PER_RANK = 512
 RESAMPLE_PAIRS          = True
 
+# === R4-05 (issue #55): one-skip-history anchor ===
+# Supervising the next item only from the immediately previous one misses slower
+# behavioural transitions. source becomes history[-2] whenever the sequence has at
+# least two items and stays history[-1] otherwise; target, pair count, sample order,
+# the in-batch candidate construction and every other knob are unchanged. This is
+# still exactly one positive pair per train row.
+R4_05_LAG2_SOURCE = True
+R4_05_LAG        = 2
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -612,7 +621,13 @@
         ):
             if history is None or len(history) == 0:
                 continue
-            pairs.append((int(history[-1]), int(target)))
+            # R4-05: one skip back, with the documented fallback to the last item.
+            source = (
+                int(history[-R4_05_LAG])
+                if R4_05_LAG2_SOURCE and len(history) >= R4_05_LAG
+                else int(history[-1])
+            )
+            pairs.append((source, int(target)))
         if not pairs:
             raise ValueError("Training parquet contains no history-target pairs")
         self.pairs = torch.tensor(pairs, dtype=torch.long)

```

