"""Runs the F1 history-conditioned probes for every frozen arm and seed.

Train/dev come from train.parquet only, split by user so a user's rows never
straddle the boundary. Nothing here reads valid.parquet or test.parquet, and no
existing Stage 2 or Stage 3 weight is loaded: every probe starts from a fresh
initialisation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from history_probe import (  # noqa: E402
    LEVELS,
    PRIMARY_SEEDS,
    HistoryProbe,
    ProbeConfig,
    build_sequences,
    evaluate_probe,
    token_ids,
    train_frequency_prior,
)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TRAIN_PARQUET = REPO_ROOT / "results/stage0_build_parquet/train.parquet"
SPLIT_SEED = 20240719

_ROW_CACHE: dict[str, Any] = {}


def load_train_rows() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """History arrays, target item ids and user ids. Loaded once and cached."""
    if "rows" not in _ROW_CACHE:
        df = pd.read_parquet(TRAIN_PARQUET, columns=["user", "seen_history", "target"])
        _ROW_CACHE["rows"] = (
            [np.asarray(h, dtype=np.int64) for h in df["seen_history"].to_numpy()],
            df["target"].to_numpy(dtype=np.int64),
            df["user"].to_numpy(dtype=np.int64),
        )
    return _ROW_CACHE["rows"]


def split_by_user(
    users: np.ndarray, dev_fraction: float, seed: int = SPLIT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic user-level split, identical for every arm and every run."""
    uniq = np.unique(users)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    n_dev = int(round(uniq.size * dev_fraction))
    dev_users = np.zeros(uniq.max() + 1, dtype=bool)
    dev_users[uniq[perm[:n_dev]]] = True
    is_dev = dev_users[users]
    return np.flatnonzero(~is_dev), np.flatnonzero(is_dev)


def run_arm(
    sid: np.ndarray,
    cfg: ProbeConfig,
    device: torch.device,
    seeds: tuple[int, ...] = PRIMARY_SEEDS,
) -> list[dict[str, Any]]:
    """Train one probe per seed on this arm and score it against the matched prior."""
    hist, targets, users = load_train_rows()
    catalogue = np.flatnonzero(sid[:, 0] >= 0)
    tokens_by_level = [token_ids(sid, catalogue, level) for level in LEVELS]
    if (tokens_by_level[0][targets] < 0).any():
        raise ValueError("a train target item is absent from this arm's catalogue")

    tr_idx, dev_idx = split_by_user(users, cfg.dev_user_fraction)
    n_tr, n_dev = len(tr_idx), len(dev_idx)

    # Runtime guard: the split is a scattered set of SOURCE row ids, so anything
    # that assumes a contiguous [train | dev] layout is wrong. Assert the two index
    # sets are disjoint, cover every row exactly once, and share no user.
    assert set(tr_idx.tolist()).isdisjoint(dev_idx.tolist()), "train/dev rows overlap"
    assert n_tr + n_dev == len(users), "split does not cover every row"
    assert set(users[tr_idx].tolist()).isdisjoint(users[dev_idx].tolist()), \
        "a user appears on both sides of the split"
    assert tr_idx.max() < len(users) and dev_idx.max() < len(users), "row id out of range"

    seq_all, _ = build_sequences(hist, targets, tokens_by_level, cfg.seq_len)
    del hist
    seq_all_t = torch.from_numpy(seq_all).to(device)

    tgt_tokens = [
        torch.from_numpy(tokens_by_level[l][targets]).to(device)
        for l in range(len(LEVELS))
    ]

    # The frequency prior is a TRAIN-only reference, fitted once per arm from
    # targets[tr_idx] and used unchanged for both train and dev scoring. Fitting it
    # on the dev labels would make G_H1 a measurement of memorisation, not of the
    # information a history carries.
    priors_train = [
        torch.as_tensor(p, device=device, dtype=torch.float32)
        for p in train_frequency_prior(tokens_by_level, targets[tr_idx])
    ]

    train_rows = torch.from_numpy(tr_idx).to(device)
    dev_rows = torch.from_numpy(dev_idx).to(device)

    results: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = HistoryProbe(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        started = time.time()
        model.train()
        for _ in range(cfg.epochs):
            perm = torch.randperm(n_tr, generator=gen)
            for start in range(0, n_tr, cfg.batch_size):
                rows = train_rows[perm[start : start + cfg.batch_size].to(device)]
                hb = seq_all_t[rows]
                tgt = [tgt_tokens[l][rows] for l in range(len(LEVELS))]
                logits = model(hb, [tgt_tokens[l][rows] for l in range(len(LEVELS))])
                loss = sum(
                    F.cross_entropy(logits[l], tgt[l], label_smoothing=cfg.label_smoothing)
                    for l in range(len(LEVELS))
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

        metrics = evaluate_probe(model, seq_all_t, tgt_tokens, dev_rows, priors_train)
        train_metrics = evaluate_probe(
            model, seq_all_t, tgt_tokens, train_rows, priors_train
        )
        assert metrics["_n_scored"] == n_dev, "dev scoring did not cover dev_idx"
        assert train_metrics["_n_scored"] == n_tr, "train scoring did not cover tr_idx"
        metrics.pop("_n_scored", None)
        metrics["seed"] = seed
        metrics["seq_len"] = cfg.seq_len
        metrics["train_rows"] = n_tr
        metrics["dev_rows"] = n_dev
        metrics["dev_row_id_sha256"] = _index_digest(dev_idx)
        metrics["train_row_id_sha256"] = _index_digest(tr_idx)
        metrics["train_gain_l1"] = train_metrics["gain_l1"]
        metrics["train_probe_ce_l1"] = train_metrics["probe_ce_l1"]
        metrics["train_prior_ce_l1"] = train_metrics["prior_ce_l1"]
        metrics["fit_seconds"] = round(time.time() - started, 2)
        results.append(metrics)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _index_digest(idx: np.ndarray) -> str:
    """Hash of the exact scored row ids, so a leakage regression is visible in the CSV."""
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()
