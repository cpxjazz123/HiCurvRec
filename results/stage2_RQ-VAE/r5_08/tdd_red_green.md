# R5_08 TDD record

## RED

Command (run against the pinned R4-08 parent):

    R5_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08 /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_08/test_r5_08.py

Exit code: 1

```
FAIL: test_red_late_start_flag: late-start flag missing/wrong on parent
FAIL: test_red_loop_gates_projection: 
PASS: test_projection_math_unchanged
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
2 test(s) failed
```

Behavioural failures observed on the parent: 2

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r5_08/test_r5_08.py

Exit code: 0

```
PASS: test_red_late_start_flag
PASS: test_red_loop_gates_projection
PASS: test_projection_math_unchanged
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
ALL TESTS PASSED
```

## Single delta

```diff
--- /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r4_08/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
+++ /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_r5_08/train_rqvae.py	2026-10-18 20:54:29.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "r4_08"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r5_08"  # R5 sole delta over pinned parent TIGER_r4_08 @ bc96625    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "r4_08":
+if EXPERIMENT_ARM != "r5_08":
     raise RuntimeError(
-        f"only the 'r4_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r5_08' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
     )
 _ANTI_COLLAPSE_FLAGS = ARM_FLAGS.get(EXPERIMENT_ARM)
 # Seed replication arms. Exp1a beat the R48 baseline by only ~1.7 unpaired
@@ -451,6 +451,7 @@
 R4_08_PCGRAD_ENC   = True
 R4_08_EPS          = 1e-12
 R4_08_PROJECTION_TRACE = {"active_count": 0}   # implementation trace only, never a metric
+R5_08_LATE_START_EPOCH = 2250  # issue #68: PCGrad only for epoch>=2250
 
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
@@ -1529,18 +1530,21 @@
                     parameter.square().mean() for parameter in _crowd_control.parameters()
                 )
             if R4_08_PCGRAD_ENC and epoch_weight > 0.0 and behavior_l1_loss is not None:
-                # One backward still produces the baseline gradient for every parameter,
-                # including the decoder and the codebooks. Only then is the encoder
-                # gradient replaced by the conflict-projected assembly.
-                loss.backward(retain_graph=True)
-                _r4_08_apply_projection(
-                    base_loss,
-                    behavior_l1_loss,
-                    behavior_other_loss,
-                    list(raw_module.encoder.parameters()),
-                    epoch_weight,
-                    valid_levels=behavior_valid_levels,
-                )
+                if epoch >= R5_08_LATE_START_EPOCH:
+                    loss.backward(retain_graph=True)
+                    _r4_08_apply_projection(
+                        base_loss,
+                        behavior_l1_loss,
+                        behavior_other_loss,
+                        list(raw_module.encoder.parameters()),
+                        epoch_weight,
+                        valid_levels=behavior_valid_levels,
+                        epoch=epoch,
+                    )
+                else:
+                    # Epochs 1..2249: full Resample-style unprojected backward inside the
+                    # R4-08 tree (behavior loss still present; only the projection is off).
+                    loss.backward()
             else:
                 loss.backward()
             if directional_heads is not None and epoch_weight > 0.0 and world_size > 1:

```

## Hotfix (runtime crash @ epoch 2250)

Call site initially passed `epoch=epoch` into `_r4_08_apply_projection`, but the parent
signature has no `epoch` kwarg. Crash:

`TypeError: _r4_08_apply_projection() got an unexpected keyword argument 'epoch'`

Fix: drop `epoch=epoch` from the call (gate already uses `epoch` outside). Re-applied in
`tools/r5_apply_mechanisms.py` and `TIGER_r5_08/train_rqvae.py`. Retrain restarted from
epoch 1 (no mid-run checkpoint).
