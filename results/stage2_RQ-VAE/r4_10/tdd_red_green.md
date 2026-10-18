# R4_10 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_10/test_r4_10.py

Exit code: 1

```
FAIL: test_red_weighted_l1_init_beats_unweighted_on_the_weighted_objective: L1 initialisation did not use the weights (weighted inertia 1036.2282)
PASS: test_red_uniform_weights_reproduce_unweighted_kmeans
PASS: test_weights_are_train_only_positive_and_finite
PASS: test_l2_l3_paths_shapes_and_seed_unchanged
PASS: test_off_flag_matches_exact_resample
PASS: test_finite_forward_and_backward
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
1 test(s) failed
```

Behavioural failures observed on the parent: 1

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_10/test_r4_10.py

Exit code: 0

```
PASS: test_red_weighted_l1_init_beats_unweighted_on_the_weighted_objective
PASS: test_red_uniform_weights_reproduce_unweighted_kmeans
PASS: test_weights_are_train_only_positive_and_finite
PASS: test_l2_l3_paths_shapes_and_seed_unchanged
PASS: test_off_flag_matches_exact_resample
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
model/layers.py
model/model.py
```

### Diff

```diff
--- stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/train_rqvae.py	2026-10-17 13:07:42.000000000 +1100
+++ stage2_RQ-VAE/TIGER_r4_10/train_rqvae.py	2026-10-18 18:53:00.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_10"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_10":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_10' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
 
+# === R4-10 (issue #60): train-target-frequency-weighted L1 KMeans initialisation ===
+# Canonical KMeans weights every catalogue item equally even though the train target
+# marginal is heavily skewed. Fitting L1 with w_i = sqrt(1 + count_train_target(i)),
+# mean-normalised over the original train catalogue, places the initial first-level
+# boundaries with that marginal in view. L2/L3 keep the same algorithm and are fed the
+# ordinary residuals afterwards; there is no post-initialisation weighting, no centroid
+# refresh, no ER auxiliary loss and no warm start.
+R4_10_WEIGHTED_KMEANS = True
+R4_10_WEIGHT_POWER    = 0.5
+R4_10_TARGET_WEIGHTS  = None    # filled in main() from train.parquet; frozen thereafter
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -957,6 +968,29 @@
         geometry_item_bucket_t = torch.as_tensor(geometry_item_bucket, device=device)
     all_embeddings = torch.from_numpy(embeddings)
     train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
+    global R4_10_TARGET_WEIGHTS
+    if R4_10_WEIGHTED_KMEANS:
+        # w_i = sqrt(1 + count_train_target(i)) over the exact original train item
+        # catalogue, mean-normalised. Train-only: reads train_frame and nothing else.
+        counts = np.bincount(
+            train_frame["target"].to_numpy(dtype=np.int64), minlength=len(embeddings)
+        )[train_ids].astype(np.float64)
+        weights = np.power(1.0 + counts, R4_10_WEIGHT_POWER)
+        weights = weights / weights.mean()
+        if not np.isfinite(weights).all() or (weights <= 0).any():
+            raise RuntimeError("R4-10 KMeans weights are non-finite or non-positive")
+        R4_10_TARGET_WEIGHTS = torch.tensor(weights, dtype=torch.float32)
+        if rank == 0:
+            _record(
+                rank,
+                "r4_10_kmeans_weights",
+                mechanism="train_target_frequency_weighted_l1_kmeans",
+                weight_power=R4_10_WEIGHT_POWER,
+                n_train_items=int(train_embeddings.shape[0]),
+                w_min=float(R4_10_TARGET_WEIGHTS.min()),
+                w_max=float(R4_10_TARGET_WEIGHTS.max()),
+                requires_grad=bool(R4_10_TARGET_WEIGHTS.requires_grad),
+            )
     # iter48's full transition stream, drawn by its own sampler so the item
     # batch above keeps exactly the rows, order and Sinkhorn balancing the
     # baseline sees.
@@ -1095,7 +1129,14 @@
     if rank == 0:
         print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
     with torch.no_grad():
-        raw_module.init_codebook(train_embeddings.to(device))
+        raw_module.init_codebook(
+            train_embeddings.to(device),
+            sample_weight=(
+                R4_10_TARGET_WEIGHTS.to(device)
+                if R4_10_WEIGHTED_KMEANS and R4_10_TARGET_WEIGHTS is not None
+                else None
+            ),
+        )
     if world_size > 1:
         dist.barrier()
 

--- stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/model/layers.py	2026-10-17 13:07:42.000000000 +1100
+++ stage2_RQ-VAE/TIGER_r4_10/model/layers.py	2026-10-18 18:11:54.000000000 +1100
@@ -10,6 +10,7 @@
 
 from typing import Any
 
+import numpy as np
 import torch
 import torch.distributed as distributed
 import torch.nn as nn
@@ -145,8 +146,22 @@
     def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
         return F.embedding(embed_id, self.get_code_embs())
 
-    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
-        kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(x.detach().cpu().numpy())
+    def init_codebook(self, x: torch.Tensor, device: torch.device, sample_weight=None) -> torch.Tensor:
+        points = x.detach().cpu().numpy()
+        if sample_weight is None:
+            # Baseline path, byte-for-byte: unweighted KMeans with n_init="auto".
+            kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(points)
+        else:
+            weights = sample_weight.detach().cpu().numpy().astype(np.float64)
+            if weights.shape[0] != points.shape[0]:
+                raise ValueError(
+                    f"sample_weight has {weights.shape[0]} rows for {points.shape[0]} points"
+                )
+            if not np.isfinite(weights).all() or (weights <= 0).any():
+                raise ValueError("sample_weight must be finite and strictly positive")
+            kmeans = KMeans(n_clusters=self.n_embed, n_init="auto").fit(
+                points, sample_weight=weights
+            )
         centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
         if distributed.is_initialized():
             distributed.broadcast(centers, 0)
@@ -298,8 +313,15 @@
             output[:, level] = indices
         return quantized_x, sum_quant_loss / self.codebook_num, num_unused_codes, output
 
-    def init_codebook(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
+    def init_codebook(self, x: torch.Tensor, device: torch.device, sample_weight=None) -> torch.Tensor:
+        """Fit every level in order; the first level may take per-item weights.
+
+        R4-10: ``sample_weight`` applies to level 0 only and is ignored afterwards, so
+        L2/L3 run the same unweighted algorithm on the ordinary residuals. The
+        residual each level hands on is unchanged by the weights.
+        """
         residual = x
-        for vq_layer in self.vq_layers:
-            residual = vq_layer.init_codebook(residual, device)
+        for level, vq_layer in enumerate(self.vq_layers):
+            weights = sample_weight if level == 0 else None
+            residual = vq_layer.init_codebook(residual, device, sample_weight=weights)
         return residual

--- stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/model/model.py	2026-10-17 13:07:42.000000000 +1100
+++ stage2_RQ-VAE/TIGER_r4_10/model/model.py	2026-10-18 18:12:05.000000000 +1100
@@ -36,9 +36,10 @@
         return tokens
 
     @torch.no_grad()
-    def init_codebook(self, embeddings: torch.Tensor) -> None:
+    def init_codebook(self, embeddings: torch.Tensor, sample_weight=None) -> None:
+        # R4-10: sample_weight is applied to the first level only, inside RQLayer.
         encoded = self.encoder(embeddings)
-        self.rq.init_codebook(encoded, embeddings.device)
+        self.rq.init_codebook(encoded, embeddings.device, sample_weight=sample_weight)
 
     def compute_loss(
         self,

```

