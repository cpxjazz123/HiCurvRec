# R4_01 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_01/test_r4_01.py

Exit code: 1

```
FAIL: test_red_l1_logits_carry_minus_half_log_q: L1 logits are not candidate-popularity corrected: max|delta| = 5.671124e+00
PASS: test_red_l2_l3_logits_are_untouched
FAIL: test_red_uniform_q_recovers_baseline_and_nonuniform_changes_loss: the correction had no effect on a skewed catalogue
PASS: test_off_flag_matches_exact_resample
PASS: test_q_is_train_only_and_finite
PASS: test_no_new_trainable_parameters
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_01/test_r4_01.py

Exit code: 0

```
PASS: test_red_l1_logits_carry_minus_half_log_q
PASS: test_red_l2_l3_logits_are_untouched
PASS: test_red_uniform_q_recovers_baseline_and_nonuniform_changes_loss
PASS: test_off_flag_matches_exact_resample
PASS: test_q_is_train_only_and_finite
PASS: test_no_new_trainable_parameters
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
+++ stage2_RQ-VAE/TIGER_r4_01/train_rqvae.py	2026-10-18 18:10:47.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_01"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_01":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_01' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
@@ -442,6 +442,20 @@
 PAIR_BATCH_SIZE_PER_RANK = 512
 RESAMPLE_PAIRS          = True
 
+# === R4-01 (issue #51): candidate-popularity logit correction on L1 ===
+# The row-wise InfoNCE treats in-batch targets drawn from the skewed train target
+# marginal as if uniform. Subtracting 0.5 * log q_train(j) from every candidate's
+# L1 logit is the log-Q correction: it removes the marginal's contribution to the
+# softmax without changing the distance scores, the temperature, the masks, the
+# cross entropy, the sampler, the codebooks or the architecture. q_train(j) is fit
+# ONCE from train.parquet targets only and stays a detached table: no gradient, no
+# optimizer entry, and never recomputed from valid/test.
+R4_01_Q_CORRECTION      = True
+R4_01_Q_SMOOTHING       = 1.0     # additive count smoothing, as prescribed
+R4_01_Q_EXPONENT        = 0.5     # the logit bias is -R4_01_Q_EXPONENT * log q
+R4_01_Q_TABLE           = None    # filled in main() from train.parquet; frozen thereafter
+MECHANISM_TRAINABLE_PARAMETERS = ()
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -846,6 +860,36 @@
         return sum(parameter.numel() for parameter in self.parameters())
 
 
+def _r4_01_q_log_bias(
+    target_ids: torch.Tensor, device: torch.device
+) -> torch.Tensor | None:
+    """``-0.5 * log q_train(j)`` for the batch targets, or None when disabled.
+
+    ``q_train(j) = (count_train_target(j) + 1) / (num_train_rows + num_catalogue_items)``,
+    so a zero-count item still receives a finite positive probability. The table is
+    built once in ``main()`` from the original train targets and is a plain detached
+    tensor: it never carries gradient and never enters the optimizer.
+    """
+    if not R4_01_Q_CORRECTION or R4_01_Q_TABLE is None:
+        return None
+    table = R4_01_Q_TABLE.to(device=device, dtype=torch.float32)
+    picked = table[target_ids.to(device)]
+    if not bool(torch.isfinite(picked).all()) or bool((picked <= 0).any()):
+        raise RuntimeError("R4-01 q_train table is non-finite or non-positive")
+    return -R4_01_Q_EXPONENT * torch.log(picked)
+
+
+def _r4_01_build_q_table(train_frame: pd.DataFrame, n_items: int) -> torch.Tensor:
+    """Fit q_train once from the ORIGINAL train targets. Train-only by construction."""
+    targets = train_frame["target"].to_numpy(dtype=np.int64)
+    counts = np.bincount(targets, minlength=n_items)[:n_items].astype(np.float64)
+    denominator = float(len(targets)) + float(n_items)
+    q = (counts + R4_01_Q_SMOOTHING) / denominator
+    if not np.isfinite(q).all() or (q <= 0).any():
+        raise RuntimeError("R4-01 q_train estimate is non-finite or non-positive")
+    return torch.tensor(q, dtype=torch.float32)
+
+
 def behavior_contrastive_loss(
     residuals: torch.Tensor,
     source_ids: torch.Tensor,
@@ -859,6 +903,13 @@
         source = residuals[level, :batch_size]
         candidates = residuals[level, batch_size:]
         logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
+        if level == 0:
+            # R4-01: only the first level is corrected, and only by the frozen
+            # marginal of the train target distribution. L2/L3 keep the baseline
+            # byte-for-byte.
+            bias = _r4_01_q_log_bias(target_ids, logits.device)
+            if bias is not None:
+                logits = logits + bias.unsqueeze(0)
         loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
         if loss is not None:
             losses.append(loss)
@@ -957,6 +1008,25 @@
         geometry_item_bucket_t = torch.as_tensor(geometry_item_bucket, device=device)
     all_embeddings = torch.from_numpy(embeddings)
     train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
+    # R4-01: fit q_train ONCE from the original train targets, before the pair
+    # stream is built, and never touch it again. Train-only: this reads train_frame
+    # and nothing else.
+    global R4_01_Q_TABLE
+    if R4_01_Q_CORRECTION:
+        R4_01_Q_TABLE = _r4_01_build_q_table(train_frame, len(embeddings))
+        if rank == 0:
+            _record(
+                rank,
+                "r4_01_q_train",
+                mechanism="candidate_popularity_logit_correction",
+                n_items=len(embeddings),
+                num_train_rows=int(len(train_frame)),
+                smoothing=R4_01_Q_SMOOTHING,
+                exponent=R4_01_Q_EXPONENT,
+                q_min=float(R4_01_Q_TABLE.min()),
+                q_max=float(R4_01_Q_TABLE.max()),
+                requires_grad=bool(R4_01_Q_TABLE.requires_grad),
+            )
     # iter48's full transition stream, drawn by its own sampler so the item
     # batch above keeps exactly the rows, order and Sinkhorn balancing the
     # baseline sees.

```

