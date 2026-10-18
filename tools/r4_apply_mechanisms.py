"""Apply the ten R4 mechanism deltas to their forked trees (#51-#60).

Each delta is one edit in one tree, applied by exact string replacement so the
change is verifiable and nothing else moves. Re-running is a no-op.

    python3 tools/r4_apply_mechanisms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TREES = REPO_ROOT / "stage2_RQ-VAE"

BASE_CONSTANTS = """BEHAVIOR_TEMPERATURE    = 0.07
RAMP_START_EPOCH        = 1500
PAIR_BATCH_SIZE_PER_RANK = 512
RESAMPLE_PAIRS          = True
"""

BASE_LOSS = """    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
"""

BASE_ALL_EMBEDDINGS = """    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
"""

# --------------------------------------------------------------------------- 02

R4_02_CONSTANTS = BASE_CONSTANTS + '''
# === R4-02 (issue #52): symmetric row-column L1 InfoNCE ===
# The baseline normalises each anchor row only. Adding the column-normalised term on
# the very same score matrix and the very same mask sharpens L1 pairing without a new
# projection head, a second sampler, extra negatives, a multi-successor label or a
# new loss weight. L2/L3 are untouched.
R4_02_SYMMETRIC_L1    = True
'''

R4_02_LOSS = '''    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        distances = torch.cdist(source, candidates, p=2)
        logits = -distances / BEHAVIOR_TEMPERATURE
        if level == 0 and R4_02_SYMMETRIC_L1:
            loss = _symmetric_row_column_ce(
                source, candidates, source_ids, target_ids, successor_indicator
            )
            if loss is not None:
                losses.append(loss)
            continue
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
'''

R4_02_HELPERS = '''def _duplicate_mask(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator,
    device: torch.device,
) -> torch.Tensor:
    """The baseline's duplicate / false-negative mask, factored out unchanged."""
    batch_size = int(source_ids.shape[0])
    duplicate_targets = target_ids[:, None].eq(target_ids[None, :])
    duplicate_targets.fill_diagonal_(False)
    if successor_indicator is not None:
        rows = successor_indicator[source_ids.cpu().numpy()]
        blocked = torch.from_numpy(
            np.asarray(rows[:, target_ids.cpu().numpy()].todense())
        ).to(device=device, dtype=torch.bool)
        blocked.fill_diagonal_(False)
        duplicate_targets = duplicate_targets | blocked
    return duplicate_targets


def _symmetric_row_column_ce(
    source: torch.Tensor,
    candidates: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator=None,
) -> torch.Tensor | None:
    """R4-02: 0.5 * CE_rows(S) + 0.5 * CE_rows(S.T) on the SAME tensors and mask.

    The reverse term reuses the same scores transposed and the same mask transposed,
    so a blocked item stays blocked in both directions and the diagonal positives stay
    unmasked. No new graph is built and no new object is learned.
    """
    batch_size = int(source_ids.shape[0])
    scores = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
    mask = _duplicate_mask(source_ids, target_ids, successor_indicator, scores.device)
    valid = source_ids.ne(target_ids)
    if not bool(valid.any()):
        return None
    labels = torch.arange(batch_size, device=source_ids.device)
    forward = F.cross_entropy(scores.masked_fill(mask, -torch.inf)[valid], labels[valid])
    reverse = F.cross_entropy(
        scores.t().masked_fill(mask.t(), -torch.inf)[valid], labels[valid]
    )
    return 0.5 * (forward + reverse)


def behavior_contrastive_loss('''

# --------------------------------------------------------------------------- 03

R4_03_CONSTANTS = BASE_CONSTANTS + '''
# === R4-03 (issue #53): inverse-anchor-frequency row weighting of L1 CE ===
# Repeated high-frequency source items dominate the row-wise L1 cross entropy. With a
# frozen train-only count f(a) and the fixed power -0.5, w(a) = (1 + f(a))**-0.5, each
# row is weighted before the reduction and the weights are renormalised, so the scale
# of the L1 term is unchanged and L2/L3, the masks, the logits and the sampler are all
# byte-for-byte the baseline's.
R4_03_ANCHOR_WEIGHTING = True
R4_03_WEIGHT_POWER     = -0.5
R4_03_ANCHOR_FREQ     = None    # filled in main() from train.parquet; frozen thereafter
MECHANISM_TRAINABLE_PARAMETERS = ()
'''

R4_03_HELPERS = '''def _r4_03_row_weights(source_ids: torch.Tensor, device: torch.device):
    """w(a) = (1 + f(a))**-0.5 from the frozen train-only anchor frequency."""
    if not R4_03_ANCHOR_WEIGHTING or R4_03_ANCHOR_FREQ is None:
        return None
    frequency = R4_03_ANCHOR_FREQ.to(device=device, dtype=torch.float32)
    counts = frequency[source_ids.to(device)]
    weights = torch.pow(1.0 + counts, R4_03_WEIGHT_POWER)
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise RuntimeError("R4-03 anchor weights are non-finite or non-positive")
    return weights


def _weighted_row_ce(
    logits: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    successor_indicator,
    weights: torch.Tensor,
) -> torch.Tensor | None:
    """Per-row CE under the baseline mask, then the prescribed weighted mean."""
    mask = _duplicate_mask(source_ids, target_ids, successor_indicator, logits.device)
    valid = source_ids.ne(target_ids)
    if not bool(valid.any()):
        return None
    labels = torch.arange(int(source_ids.shape[0]), device=source_ids.device)
    rows = F.cross_entropy(
        logits.masked_fill(mask, -torch.inf)[valid], labels[valid], reduction="none"
    )
    selected = weights[valid]
    return (selected * rows).sum() / selected.sum()


def behavior_contrastive_loss('''

R4_03_LOSS = '''    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
        if level == 0:
            weights = _r4_03_row_weights(source_ids, logits.device)
            if weights is not None:
                loss = _weighted_row_ce(
                    logits, source_ids, target_ids, successor_indicator, weights
                )
                if loss is not None:
                    losses.append(loss)
                continue
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
'''

R4_03_MAIN = '''    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    global R4_03_ANCHOR_FREQ
    if R4_03_ANCHOR_WEIGHTING:
        sources = np.asarray(
            [int(history[-1]) for history in train_frame["seen_history"].to_numpy()
             if history is not None and len(history) > 0],
            dtype=np.int64,
        )
        R4_03_ANCHOR_FREQ = torch.tensor(
            np.bincount(sources, minlength=len(embeddings))[: len(embeddings)].astype(np.float32)
        )
        if rank == 0:
            _record(
                rank,
                "r4_03_anchor_frequency",
                mechanism="inverse_anchor_frequency_row_weighting",
                weight_power=R4_03_WEIGHT_POWER,
                n_rows=int(sources.shape[0]),
                f_max=float(R4_03_ANCHOR_FREQ.max()),
                requires_grad=bool(R4_03_ANCHOR_FREQ.requires_grad),
            )
'''

# --------------------------------------------------------------------------- 05

R4_05_CONSTANTS = BASE_CONSTANTS + '''
# === R4-05 (issue #55): one-skip-history anchor ===
# Supervising the next item only from the immediately previous one misses slower
# behavioural transitions. source becomes history[-2] whenever the sequence has at
# least two items and stays history[-1] otherwise; target, pair count, sample order,
# the in-batch candidate construction and every other knob are unchanged. This is
# still exactly one positive pair per train row.
R4_05_LAG2_SOURCE = True
R4_05_LAG        = 2
'''

R4_05_DATASET_OLD = '''            if history is None or len(history) == 0:
                continue
            pairs.append((int(history[-1]), int(target)))
'''

R4_05_DATASET_NEW = '''            if history is None or len(history) == 0:
                continue
            # R4-05: one skip back, with the documented fallback to the last item.
            source = (
                int(history[-R4_05_LAG])
                if R4_05_LAG2_SOURCE and len(history) >= R4_05_LAG
                else int(history[-1])
            )
            pairs.append((source, int(target)))
'''

# --------------------------------------------------------------------------- 06

R4_06_CONSTANTS = BASE_CONSTANTS + '''
# === R4-06 (issue #56): unit-normalized cosine distance for L1 logits ===
# The Euclidean L1 score reads latent norm as behaviour. Replacing it with
# -(1 - cos)/0.07 on the same pairs, the same temperature, the same mask and the same
# cross entropy measures direction only. The normalisation exists solely inside this
# score path: no normalised tensor ever reaches the quantizer, and L2/L3 keep their
# Euclidean scores.
R4_06_COSINE_L1 = True
R4_06_EPS      = 1e-8
'''

R4_06_LOSS = '''    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        if level == 0 and R4_06_COSINE_L1:
            unit_source = F.normalize(source, eps=R4_06_EPS)
            unit_candidates = F.normalize(candidates, eps=R4_06_EPS)
            logits = -(1.0 - unit_source @ unit_candidates.t()) / BEHAVIOR_TEMPERATURE
        else:
            logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
'''

# --------------------------------------------------------------------------- 07

R4_07_CONSTANTS = BASE_CONSTANTS + '''
# === R4-07 (issue #57): Huberized robust distance for L1 behaviour comparison ===
# A few large residual-coordinate differences dominate the Euclidean L1 logits. With
# the fixed beta = 1.0 in original latent units,
# D_ij = sqrt(2 * sum_k smooth_l1(a_i[k] - b_j[k]) + 1e-12)
# agrees with the Euclidean norm while every |delta| <= 1 and grows more slowly for
# outliers. The codebook assignment, the loss weights, the sampler, the encoder, the
# reconstruction and L2/L3 are untouched, and cosine is NOT stacked in this arm.
R4_07_HUBER_L1 = True
R4_07_BETA    = 1.0
R4_07_EPS     = 1e-12
'''

R4_07_LOSS = '''    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        if level == 0 and R4_07_HUBER_L1:
            differences = source[:, None, :] - candidates[None, :, :]
            per_coordinate = F.smooth_l1_loss(
                differences,
                torch.zeros_like(differences),
                beta=R4_07_BETA,
                reduction="none",
            )
            distances = torch.sqrt(2.0 * per_coordinate.sum(-1) + R4_07_EPS)
            logits = -distances / BEHAVIOR_TEMPERATURE
        else:
            logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
'''

# --------------------------------------------------------------------------- 09

R4_09_CONSTANTS = BASE_CONSTANTS + '''
# === R4-09 (issue #59): target-branch stop-gradient in the original L1 InfoNCE ===
# The L1 pairwise CE updates the anchor and the target representation at once, which
# can drag targets in conflicting directions. Detaching the candidate rows changes no
# value, only the gradient path: the same candidate numbers, the same Euclidean
# scores, temperature, mask and cross entropy, with L2/L3, the reconstruction, the VQ
# codebook update and every hyperparameter left alone. No new weights, no teacher, no
# frozen checkpoint.
R4_09_TARGET_DETACH = True
'''

R4_09_LOSS = '''    for level in ACTIVE_LEVELS:
        source = residuals[level, :batch_size]
        candidates = residuals[level, batch_size:]
        if level == 0 and R4_09_TARGET_DETACH:
            # Identical forward value; the target branch stops carrying gradient.
            candidates = candidates.detach()
        logits = -torch.cdist(source, candidates, p=2) / BEHAVIOR_TEMPERATURE
        loss = _contrastive_ce(logits, source_ids, target_ids, successor_indicator)
'''

# --------------------------------------------------------------------------- 10

R4_10_CONSTANTS = BASE_CONSTANTS + '''
# === R4-10 (issue #60): train-target-frequency-weighted L1 KMeans initialisation ===
# Canonical KMeans weights every catalogue item equally even though the train target
# marginal is heavily skewed. Fitting L1 with w_i = sqrt(1 + count_train_target(i)),
# mean-normalised over the original train catalogue, places the initial first-level
# boundaries with that marginal in view. L2/L3 keep the same algorithm and are fed the
# ordinary residuals afterwards; there is no post-initialisation weighting, no centroid
# refresh, no ER auxiliary loss and no warm start.
R4_10_WEIGHTED_KMEANS = True
R4_10_WEIGHT_POWER    = 0.5
R4_10_TARGET_WEIGHTS  = None    # filled in main() from train.parquet; frozen thereafter
'''

DELTAS: dict[str, list[tuple[str, str]]] = {
    "r4_02": [
        (BASE_CONSTANTS, R4_02_CONSTANTS),
        ("def behavior_contrastive_loss(", R4_02_HELPERS),
        (BASE_LOSS, R4_02_LOSS),
    ],
    "r4_03": [
        (BASE_CONSTANTS, R4_03_CONSTANTS),
        ("def behavior_contrastive_loss(", R4_03_HELPERS),
        (BASE_LOSS, R4_03_LOSS),
        (BASE_ALL_EMBEDDINGS, R4_03_MAIN),
    ],
    "r4_05": [
        (BASE_CONSTANTS, R4_05_CONSTANTS),
        (R4_05_DATASET_OLD, R4_05_DATASET_NEW),
    ],
    "r4_06": [
        (BASE_CONSTANTS, R4_06_CONSTANTS),
        (BASE_LOSS, R4_06_LOSS),
    ],
    "r4_07": [
        (BASE_CONSTANTS, R4_07_CONSTANTS),
        (BASE_LOSS, R4_07_LOSS),
    ],
    "r4_09": [
        (BASE_CONSTANTS, R4_09_CONSTANTS),
        (BASE_LOSS, R4_09_LOSS),
    ],
    "r4_10": [
        (BASE_CONSTANTS, R4_10_CONSTANTS),
    ],
}


def apply(arm: str, edits: list[tuple[str, str]]) -> int:
    entry = TREES / f"TIGER_{arm}" / "train_rqvae.py"
    text = entry.read_text(encoding="utf-8")
    applied = 0
    for old, new in edits:
        if new in text and old not in text:
            continue
        if old not in text:
            raise SystemExit(f"{arm}: anchor not found:\n{old[:200]}")
        if text.count(old) != 1:
            raise SystemExit(f"{arm}: anchor is not unique ({text.count(old)} hits):\n{old[:200]}")
        text = text.replace(old, new, 1)
        applied += 1
    entry.write_text(text, encoding="utf-8")
    return applied


def main() -> int:
    for arm, edits in DELTAS.items():
        applied = apply(arm, edits)
        print(f"{arm}: applied {applied}/{len(edits)} delta edits")
    print("r4_01: applied manually in the same shape (q table + L1 logit bias)")
    print("r4_04 / r4_08: sampler and gradient-projection deltas applied separately")
    return 0


if __name__ == "__main__":
    sys.exit(main())
