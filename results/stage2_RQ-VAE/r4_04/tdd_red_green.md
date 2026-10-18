# R4_04 TDD record

## RED

Command (run against the unchanged canonical Resample parent):

    R4_TEST_TREE=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_04/test_r4_04.py

Exit code: 1

```
FAIL: test_red_edge_probability_ratio_is_uniform_over_unique_edges: edge sampling is still row-frequency proportional (hot/cold = 96.09, expected ~1)
PASS: test_red_unique_support_is_unchanged
PASS: test_red_repeated_row_keeps_the_pair_count_of_the_baseline
PASS: test_off_flag_matches_exact_resample
FAIL: test_determinism_of_the_uniform_draw: uniform-over-unique-edges draw is not implemented
FAIL: test_every_drawn_edge_is_a_real_train_edge: uniform-over-unique-edges draw is not implemented
PASS: test_finite_forward_and_backward
PASS: test_train_only_data_audit
PASS: test_sid_three_token_contract
PASS: test_stage3_gate_fail_closed_and_mock_launcher
3 test(s) failed
```

Behavioural failures observed on the parent: 3

## GREEN

Command (run against this arm's own tree, mechanism implemented):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 results/stage2_RQ-VAE/r4_04/test_r4_04.py

Exit code: 0

```
PASS: test_red_edge_probability_ratio_is_uniform_over_unique_edges
PASS: test_red_unique_support_is_unchanged
PASS: test_red_repeated_row_keeps_the_pair_count_of_the_baseline
PASS: test_off_flag_matches_exact_resample
PASS: test_determinism_of_the_uniform_draw
PASS: test_every_drawn_edge_is_a_real_train_edge
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
+++ stage2_RQ-VAE/TIGER_r4_04/train_rqvae.py	2026-10-18 18:52:47.000000000 +1100
@@ -114,13 +114,13 @@
     "/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet"   # stage0
 )
 # === 本轮实验臂 (改这一行切换 arm; 每个 arm 的提交状态即实际跑过的配置) ===
-EXPERIMENT_ARM = "resample"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
+EXPERIMENT_ARM = "r4_04"    # 唯一机制: resample (Stage3 test_recall@10 = 0.0637372)
 # === 唯一机制 (CLAUDE.md §1: 只有 resample, 没有可选臂) ===
 # resample 是本树唯一允许的机制 (Stage3 test_recall@10 = 0.0637372)。其余臂与其
 # 查表仅作历史记录保留在文件里, 但一旦被选中即在导入时失败, 不允许运行。
-if EXPERIMENT_ARM != "resample":
+if EXPERIMENT_ARM != "r4_04":
     raise RuntimeError(
-        f"only the 'resample' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
+        f"only the 'r4_04' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
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
 
+# === R4-04 (issue #54): unique-transition-uniform behaviour-pair sampling ===
+# The baseline draws pair rows with probability proportional to how often that row's
+# transition repeats in train, so a handful of transitions dominate L1. Drawing IID
+# uniform WITH REPLACEMENT from the sorted unique (history[-1], target) edges keeps the
+# same relation and the same number of pair samples per optimizer step, total pair
+# steps, per-rank batch size and epoch length. It is a sampler swap, not multi-positive
+# training: the InfoNCE loss, masks, temperature, 3-level weights, model, KMeans,
+# optimizer, schedule and the item reconstruction loader are untouched.
+R4_04_UNIQUE_UNIFORM = True
+
 FALSE_NEGATIVE_MASK     = EXPERIMENT_ARM == "resample_fn_mask"
 # CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
 # 候选是否采用完全交由下游 stage3 test_R@10 裁决.
@@ -846,6 +856,97 @@
         return sum(parameter.numel() for parameter in self.parameters())
 
 
+def _r4_04_unique_edges(dataset: "TransitionPairs") -> torch.Tensor:
+    """Sorted unique (history[-1], target) edges of the ORIGINAL train relation.
+
+    Sorting makes the support a function of the data alone, so every DDP rank builds
+    byte-identical tables and a fixed seed reproduces a run exactly.
+    """
+    pairs = dataset.pairs
+    unique = torch.unique(pairs, dim=0)
+    return unique[torch.argsort(unique[:, 0] * (unique[:, 1].max() + 1) + unique[:, 1])]
+
+
+def r4_sample_uniform_pairs(
+    edges: torch.Tensor, count: int, generator: torch.Generator
+) -> torch.Tensor:
+    """IID uniform draws WITH REPLACEMENT from the unique edge table.
+
+    Every returned edge is a real train transition; nothing synthetic is introduced
+    and no future transition can appear. A uniform draw over unique edges replaces the
+    baseline's row-frequency draw, which is what this arm is testing.
+    """
+    if edges.numel() == 0:
+        raise ValueError("the unique edge table is empty")
+    positions = torch.randint(
+        edges.shape[0], (count,), generator=generator, dtype=torch.long
+    )
+    return edges[positions]
+
+
+class _R4UniformEdgePairs(torch.utils.data.Dataset):
+    """Per-epoch uniform draws over the unique edges, same length as the baseline."""
+
+    def __init__(
+        self,
+        edges: torch.Tensor,
+        samples: int,
+        seed: int,
+        epoch: int = 0,
+    ) -> None:
+        self.edges = edges
+        self.samples = int(samples)
+        self.seed = int(seed)
+        self.epoch = -1
+        self.drawn = edges[:0]
+        self.set_epoch(epoch)
+
+    def __len__(self) -> int:
+        return self.samples
+
+    def set_epoch(self, epoch: int) -> None:
+        epoch = int(epoch)
+        if epoch == self.epoch and int(self.drawn.shape[0]) == self.samples:
+            return
+        self.epoch = epoch
+        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * epoch)
+        self.drawn = r4_sample_uniform_pairs(self.edges, self.samples, generator)
+
+    def __setitem__(self, key, value) -> None:  # DataLoader's sampler hook
+        if key == "epoch":
+            self.set_epoch(int(value))
+
+    def __getitem__(self, index: int) -> torch.Tensor:
+        return self.drawn[index]
+
+
+def _r4_04_uniform_pair_loader(
+    edges: torch.Tensor,
+    num_samples: int,
+    batch_size: int,
+    rank: int,
+    world_size: int,
+    device: torch.device,
+    pair_sampler,
+) -> DataLoader:
+    """Uniform-over-unique-edges loader with the baseline's shapes and step count."""
+    # One draw per row keeps len(dataset) equal to the baseline's row count, so the
+    # number of pair batches per epoch (and therefore total pair steps) is unchanged.
+    dataset = _R4UniformEdgePairs(edges, int(num_samples), SEED)
+    if world_size > 1 and pair_sampler is not None:
+        pair_sampler.set_epoch(0)
+    return DataLoader(
+        dataset,
+        batch_size=batch_size,
+        shuffle=False,
+        sampler=pair_sampler,
+        num_workers=NUM_WORKERS,
+        pin_memory=device.type == "cuda",
+        persistent_workers=False,
+        drop_last=False,
+    )
+
+
 def behavior_contrastive_loss(
     residuals: torch.Tensor,
     source_ids: torch.Tensor,
@@ -1039,6 +1140,21 @@
         persistent_workers=False,
         drop_last=False,
     )
+    # R4-04: replace the row-frequency pair loader with uniform draws over the sorted
+    # unique edges. Same batch size, same number of batches per epoch, same total pair
+    # steps; only the distribution over edges changes.
+    unique_pair_edges = None
+    if R4_04_UNIQUE_UNIFORM:
+        unique_pair_edges = _r4_04_unique_edges(pair_dataset)
+        pair_loader = _r4_04_uniform_pair_loader(
+            unique_pair_edges,
+            len(pair_dataset),
+            loader_pair_batch,
+            rank,
+            world_size,
+            device,
+            pair_sampler,
+        )
     # === Round 2 crowding precompute (successor CSR, popularity-matched negative
     # pools, PMI weights). All numpy, all once, no per-batch Python loops. ===
     c2_state = None
@@ -1118,6 +1234,8 @@
         # roughly 54 times; with RESAMPLE_PAIRS=False that behaviour is restored
         # exactly, which is what the "no_resample" control arm is for.
         if RESAMPLE_PAIRS or epoch == 1:
+            if R4_04_UNIQUE_UNIFORM and isinstance(pair_loader.dataset, _R4UniformEdgePairs):
+                pair_loader.dataset.set_epoch(epoch - 1)
             pair_iter = iter(pair_loader)
         model.train()
         losses: list[float] = []

```

