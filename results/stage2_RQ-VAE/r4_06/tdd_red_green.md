# R4_06 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_06/test_r4_06.py

Exit code: 1

```
FAIL: test_red_l1_logits_are_cosine_scores: L1 is not cosine: max|delta| = 2.782528e+01
FAIL: test_red_positive_rescaling_changes_euclid_but_not_cosine: cosine logits changed under rescaling
PASS: test_red_l2_l3_logits_are_bitwise_identical
PASS: test_zero_and_near_zero_norms_are_finite
FAIL: test_matching_directions_score_equally_for_any_positive_magnitude: aligned directions did not score ~0: -571.4285888671875
PASS: test_off_flag_matches_exact_resample
PASS: test_no_projection_head_parameters_added
PASS: test_finite_forward_and_backward
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
3 test(s) failed
```

Behavioural failures observed on the parent: 3

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_06/test_r4_06.py

Exit code: 0

```
PASS: test_red_l1_logits_are_cosine_scores
PASS: test_red_positive_rescaling_changes_euclid_but_not_cosine
PASS: test_red_l2_l3_logits_are_bitwise_identical
PASS: test_zero_and_near_zero_norms_are_finite
PASS: test_matching_directions_score_equally_for_any_positive_magnitude
PASS: test_off_flag_matches_exact_resample
PASS: test_no_projection_head_parameters_added
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
+++ stage2_RQ-VAE/TIGER_r4_06/train_rqvae.py	2026-10-18 18:11:13.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_06"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_06":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_06' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
 
+# === R4-06 (issue #56): unit-normalized cosine distance for L1 logits ===
+# The Euclidean L1 score reads latent norm as behaviour. Replacing it with
+# -(1 - cos)/0.07 on the same pairs, the same temperature, the same mask and the same
+# cross entropy measures direction only. The normalisation exists solely inside this
+# score path: no normalised tensor ever reaches the quantizer, and L2/L3 keep their
+# Euclidean scores.
+R4_06_COSINE_L1 = True
+R4_06_EPS      = 1e-8
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -858,7 +867,12 @@
     for level in ACTIVE_LEVELS:
         source = residuals[level, :batch_size]
         candidates = residuals[level, batch_size:]
-        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
+        if level == 0 and R4_06_COSINE_L1:
+            unit_source = F.normalize(source, eps=R4_06_EPS)
+            unit_candidates = F.normalize(candidates, eps=R4_06_EPS)
+            logits = -(1.0 - unit_source @ unit_candidates.t()) / BEHAVIOR_TEMPERATURE
+        else:
+            logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
         loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
         if loss is not None:
             losses.append(loss)

```

