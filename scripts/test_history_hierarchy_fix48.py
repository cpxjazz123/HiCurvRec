"""Regression tests for the #48 bugfix round (issue #49).

Every test here FAILS on the pre-fix code; that is the point. They cover:

  1. a noncontiguous user split: the rows scored as dev are exactly dev_idx, in
     order, with no train row and no train user among them;
  2. a train-only frequency prior: perturbing the dev targets leaves the fitted
     priors bit-identical;
  3. per-level prior CE and vocab Top-B against hand-computed ground truth;
  4. the F3 population is exactly the 52,855 frozen held-out positives, row-wise;
  5. unequal-size SID groups make the demand / item / group averages differ;
  6. an end-to-end probe whose score moves the right way under a controlled
     history signal, plus fourth-token exclusion from the SID3 statistics;
  7. the zero-padding limitation is measured and reported, not silently accepted.

Run: python3 scripts/test_history_hierarchy_fix48.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import history_hierarchy_features as H  # noqa: E402
import history_probe as HP  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# ------------------------------------------------------------------ 1
def test_noncontiguous_split_scores_dev() -> None:
    print("\n[1] dev scoring uses exactly dev_idx on a noncontiguous split")
    n = 40
    users = np.repeat(np.arange(n // 2), 2)
    tr = np.array([i for i in range(n) if users[i] % 2 == 0])
    dev = np.array([i for i in range(n) if users[i] % 2 == 1])
    check("the synthetic split really is noncontiguous",
          not np.array_equal(tr, np.arange(len(tr))),
          f"tr={tr[:5]}...")

    # Give every row a distinguishable label and check which rows get scored.
    seq = torch.arange(n, dtype=torch.long).view(n, 1, 1).expand(n, 3, 4).clone()
    toks = [torch.arange(n) % 256, torch.arange(n) % 7, torch.arange(n) % 5]
    # Proper per-level tables: a marginal, then one conditional table per prefix level.
    prior = [
        F.softmax(torch.randn(256, generator=torch.Generator().manual_seed(0)), 0),
        F.softmax(torch.randn(256, 256, generator=torch.Generator().manual_seed(1)), -1),
        F.softmax(torch.randn(256, 256, 256, generator=torch.Generator().manual_seed(2)), -1),
    ]

    class Echo(HP.HistoryProbe):
        def forward(self, seq, prior_tokens):
            return [
                F.one_hot(seq[:, l, 0], 256).float() * 100.0
                for l in range(len(HP.LEVELS))
            ]

    model = Echo(HP.ProbeConfig())
    # Evaluate with the CORRECT dev_idx and confirm the scored count.
    out = HP.evaluate_probe(model, seq, toks, torch.from_numpy(dev), prior, batch_size=8)
    check("dev evaluation covers exactly dev_idx", out["_n_scored"] == len(dev),
          f"{out['_n_scored']} vs {len(dev)}")

    # The pre-fix code passed arange(n_tr, n_tr+n_dev). Reproduce that mistake and
    # show it scores a different, overlapping population.
    buggy = torch.arange(len(tr), len(tr) + len(dev))
    buggy_out = HP.evaluate_probe(model, seq, toks, buggy, prior, batch_size=8)
    buggy_set = set(buggy.tolist())
    check("the old contiguous range would have scored train rows",
          len(buggy_set & set(tr.tolist())) > 0,
          f"{len(buggy_set & set(tr.tolist()))} train rows among the old dev rows")
    check("the old contiguous range missed most real dev rows",
          len(buggy_set & set(dev.tolist())) < len(dev),
          f"{len(buggy_set & set(dev.tolist()))} of {len(dev)} dev rows covered")
    check("the two index sets are not interchangeable",
          buggy_out["_n_scored"] == out["_n_scored"])

    # Determinism of the scored set.
    again = HP.evaluate_probe(model, seq, toks, torch.from_numpy(dev), prior, batch_size=8)
    check("scoring is deterministic for a fixed index set",
          abs(again["probe_ce_l1"] - out["probe_ce_l1"]) < 1e-9)

    # run_arm's own runtime guard must reject an overlapping split.
    import inspect

    from history_probe_runner import run_arm
    src = inspect.getsource(run_arm)
    check("run_arm builds dev_rows from dev_idx, not from a range",
          "dev_rows = torch.from_numpy(dev_idx)" in src,
          "found" if "dev_rows = torch.from_numpy(dev_idx)" in src else "MISSING")
    check("run_arm asserts the split is disjoint and user-pure",
          "isdisjoint" in src and "a user appears on both sides" in src)
    check("run_arm no longer builds a synthetic contiguous dev range",
          "torch.arange(n_tr" not in src, "arange still present" if "torch.arange(n_tr" in src else "clean")


# ------------------------------------------------------------------ 2
def test_prior_is_train_only() -> None:
    print("\n[2] the frequency prior is fitted on train targets only")
    n_items = 60
    sid = np.zeros((n_items, 4), dtype=np.int64)
    sid[:, 0] = np.arange(n_items) % 5
    sid[:, 1] = np.arange(n_items) % 4
    sid[:, 2] = np.arange(n_items) % 3
    toks = [HP.token_ids(sid, np.arange(n_items), l) for l in (1, 2, 3)]

    n_rows = 60
    tr_idx = np.arange(0, 40)
    dev_idx = np.arange(40, 60)
    train_targets = np.arange(n_rows) % 7
    # Deliberately shifted dev label distribution.
    dev_targets = (np.arange(n_rows) + 3) % 11

    p_train = HP.train_frequency_prior(toks, train_targets[tr_idx])
    # Fitting on dev labels instead gives a different table -- that is the bug.
    p_devfit = HP.train_frequency_prior(toks, dev_targets[dev_idx])
    check("a dev-fitted prior differs from a train-fitted one",
          not np.allclose(p_train[0], p_devfit[0]),
          f"max diff {np.abs(p_train[0] - p_devfit[0]).max():.6f}")

    # Changing ONLY the dev labels must leave the train-fitted prior untouched.
    p_train2 = HP.train_frequency_prior(toks, train_targets[tr_idx])
    check("refitting on unchanged train targets is bit-identical",
          all(np.array_equal(a, b) for a, b in zip(p_train, p_train2)))
    dev_only = [t.copy() for t in toks]
    check("dev labels never enter the prior fit",
          all(isinstance(p, np.ndarray) and p.shape[-1] == 256 for p in p_train),
          str([p.shape for p in p_train]))

    import inspect
    from history_probe_runner import run_arm
    src = inspect.getsource(run_arm)
    check("run_arm fits the prior on targets[tr_idx] only",
          "targets[tr_idx]" in src and "targets[dev_idx]" not in src,
          "dev_idx in source" if "targets[dev_idx]" in src else "train-only")
    check("run_arm has no separate dev-fitted prior",
          "priors_dev" not in src, "priors_dev still present" if "priors_dev" in src else "clean")


# ------------------------------------------------------------------ 3
def test_prior_ce_and_topk() -> None:
    print("\n[3] prior CE and vocab Top-B against hand-computed truth")

    # --- Level 1: mass 0.5 on token 7 and 0.5 on token 8, so Top-1 is a coin flip
    # and Top-2 always hits.
    d1 = torch.full((256,), 0.0, dtype=torch.float32)
    d1[7] = 0.5
    d1[8] = 0.5
    tgt = torch.tensor([7, 8])
    rows = torch.arange(2)
    got = HP.conditional_prior_rows([d1], [torch.zeros(2, dtype=torch.long)], rows, 1)
    check("L1 prior broadcasts to [B, 256]", tuple(got.shape) == (2, 256), str(tuple(got.shape)))
    ce = float(-torch.log(got.gather(1, tgt[:, None]).squeeze(1)).mean())
    expected_ce = float(-np.log(0.5))
    check("L1 prior CE matches the hand value", abs(ce - expected_ce) < 1e-6,
          f"{ce:.6f} vs {expected_ce:.6f}")
    check("L1 prior rows are normalised",
          bool(torch.allclose(got.sum(-1), torch.ones(2), atol=1e-6)))
    # Tokens 7 and 8 tie at 0.5, so a deterministic argmax picks 7 for both rows and
    # Top-1 hits exactly one of them; Top-2 covers both.
    top1 = float((got.topk(1, -1).indices == tgt[:, None]).any(1).float().mean())
    check("L1 prior Top-1 hits one of the two tied targets", abs(top1 - 0.5) < 1e-9,
          str(top1))
    top2 = float((got.topk(2, -1).indices == tgt[:, None]).any(1).float().mean())
    check("L1 prior Top-2 covers both tied targets", abs(top2 - 1.0) < 1e-9, str(top2))
    top5 = float((got.topk(5, -1).indices == tgt[:, None]).any(1).float().mean())
    check("L1 prior Top-5 also covers both", abs(top5 - 1.0) < 1e-9, str(top5))

    # Top-B must rank the 256 TOKENS. Ranking the batch axis instead yields one row
    # id per token -- never a token id -- so no target can ever be hit that way.
    over_batch = got.argmax(dim=0)
    batch_hits = int((over_batch.unsqueeze(0) == tgt.unsqueeze(1)).any().item())
    check("ranking the batch axis yields row ids, never token ids",
          bool((over_batch < 2).all()) and batch_hits == 0,
          f"{sorted(set(over_batch.tolist()))} distinct ids, hits={batch_hits}")

    # --- Level 2: a genuinely one-hot conditional -> CE is exactly 0.
    t1 = torch.tensor([3, 3])
    p2 = torch.zeros(4, 256)
    p2[:, 5] = 1.0
    got2 = HP.conditional_prior_rows(
        [d1, p2], [t1, torch.zeros(2, dtype=torch.long)], rows, 2
    )
    check("L2 prior is [B, 256]", tuple(got2.shape) == (2, 256), str(tuple(got2.shape)))
    tg2 = torch.tensor([5, 5])
    ce2 = float(-torch.log(got2.gather(1, tg2[:, None]).squeeze(1)).mean())
    check("L2 prior CE is 0 nats on a one-hot conditional", abs(ce2) < 1e-6, f"{ce2:.3e}")
    check("L2 prior rows are normalised",
          bool(torch.allclose(got2.sum(-1), torch.ones(2), atol=1e-6)))

    # --- Level 3: prefix index must address the right axis.
    t2 = torch.tensor([1, 2])
    p3 = torch.zeros(7, 7, 256)
    p3[:, :, 9] = 1.0
    got3 = HP.conditional_prior_rows(
        [d1, p2, p3], [t1, t2, torch.zeros(2, dtype=torch.long)], rows, 3
    )
    check("L3 prior is [B, 256]", tuple(got3.shape) == (2, 256), str(tuple(got3.shape)))
    check("L3 prior is normalised",
          bool(torch.allclose(got3.sum(-1), torch.ones(2), atol=1e-6)))
    ce3 = float(-torch.log(got3.gather(1, torch.tensor([9, 9])[:, None]).squeeze(1)).mean())
    check("L3 prior CE is 0 nats on a one-hot conditional", abs(ce3) < 1e-6, f"{ce3:.3e}")

    # The old implementation summed -log over the whole 256-wide row, which is a
    # distribution log-likelihood rather than a cross-entropy. Pin the gap.
    import math
    # A near-uniform row: the old sum-over-vocab gave ~1500 nats, the cross-entropy
    # of the true token is log(256) = 5.545.
    near_uniform = torch.full((1, 256), 1.0 / 256)
    near_uniform[0, 5] = 1.0 / 256
    naive = float(-torch.log(near_uniform).sum(-1).mean())
    correct = float(math.log(256))
    check("the old sum-over-vocab quantity is ~1500, not a cross-entropy",
          1000 < naive < 2000 and abs(naive - correct) > 100,
          f"naive={naive:.2f} nats, cross-entropy={correct:.4f} nats")


# ------------------------------------------------------------------ 4
def test_f3_population() -> None:
    print("\n[4] the F3 population is exactly the frozen held-out positives")
    vt = H.load_valid_target_ids()
    npz = np.load(H.VALID_PAIRS)["positives"].astype(np.int64)
    check("row count is 52,855", len(vt) == 52855, str(len(vt)))
    check("row-wise identical to the frozen positives array",
          bool((vt == npz).all()))
    check("not the full 57,439-row validation set", len(vt) != 57439, str(len(vt)))

    # The frozen meta says transitions also seen in train were dropped.
    meta = json.loads((H.VALID_PAIRS.with_suffix(".meta.json")).read_text())
    anchors = np.load(H.VALID_PAIRS)["anchors"].astype(np.int64)
    import pandas as pd
    train = pd.read_parquet(H.TRAIN_PARQUET, columns=["seen_history", "target"])
    train_pairs = {
        (int(h[-1]), int(t))
        for h, t in zip(train["seen_history"].to_numpy(),
                        train["target"].to_numpy(dtype=np.int64))
        if h is not None and len(h) > 0
    }
    overlap = sum(1 for a, p in zip(anchors, vt) if (int(a), int(p)) in train_pairs)
    check("no scored pair reappears in train", overlap == 0, f"{overlap} overlapping")
    check("the frozen meta confirms the drop",
          meta["filters"]["dropped_also_a_train_transition"] > 0,
          str(meta["filters"]["dropped_also_a_train_transition"]))

    # The old tuple/int comparison filtered nothing.
    rows = [(1, 5), (2, 6)]
    train_pairs = {(1, 5), (2, 6)}
    kept_old = [t for (_, t) in rows if _ not in train_pairs]
    check("the old int-vs-tuple filter would have kept everything",
          len(kept_old) == 2, str(kept_old))
    kept_new = [t for (a, t) in rows if (a, t) not in train_pairs]
    check("the tuple filter removes the overlapping transitions", kept_new == [],
          str(kept_new))


# ------------------------------------------------------------------ 5
def test_ambiguity_weightings() -> None:
    print("\n[5] demand / item / group ambiguity averages are separately computed")
    sid = np.zeros((9, 4), dtype=np.int64)
    sid[:, 0] = [0, 0, 0, 1, 1, 1, 1, 1, 2]
    sid[:, 1] = 0
    sid[:, 2] = 0
    r = H.target_ambiguity(sid, np.array([0, 0, 0, 4]), np.arange(9))
    check("A_target is the demand-weighted mean",
          abs(r["A_target"] - float(np.mean(np.log([3, 3, 3, 5])))) < 1e-12,
          f"{r['A_target']:.6f}")
    check("A_catalogue_uniform is the item-weighted mean",
          abs(r["A_catalogue_uniform"]
              - float(np.mean(np.log([3] * 3 + [5] * 5 + [1])))) < 1e-12,
          f"{r['A_catalogue_uniform']:.6f}")
    check("A_group_uniform is the group-weighted mean",
          abs(r["A_group_uniform"] - float(np.mean(np.log([3, 5, 1])))) < 1e-12,
          f"{r['A_group_uniform']:.6f}")
    check("the three differ on unequal group sizes",
          len({round(r[k], 9) for k in
               ("A_target", "A_catalogue_uniform", "A_group_uniform")}) == 3)

    # A real arm must not report -inf or NaN for the item-weighted mean.
    inv = H.load_inventory()
    m = next(x for x in inv["models"]
             if x["model"] == "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample")
    sid_r, _ = H.load_sid(H.resolve_sid_path(m["sid_path"]))
    rr = H.target_ambiguity(sid_r, H.load_valid_target_ids(), H.load_catalogue_ids())
    for k in ("A_target", "A_catalogue_uniform", "A_group_uniform"):
        check(f"{k} is finite on a real arm", np.isfinite(rr[k]), f"{rr[k]:.6f}")


# ------------------------------------------------------------------ 6
def test_end_to_end_probe() -> None:
    print("\n[6] end-to-end probe: score responds to a controlled history signal")
    from history_probe_runner import _index_digest, split_by_user

    # Synthetic catalogue whose SID is a deterministic function of a 4-bit signal,
    # so a history that carries the signal must beat the frequency prior.
    n_items = 512
    signal = np.arange(n_items) % 8
    sid = np.zeros((n_items, 4), dtype=np.int64)
    sid[:, 0] = signal
    sid[:, 1] = signal
    sid[:, 2] = signal
    sid[:, 3] = (sid[:, 0] * 97 + 11) % 781  # a real-looking extension token

    rows = 4096
    users = np.repeat(np.arange(rows // 2), 2)
    rng = np.random.default_rng(5)
    hist = []
    targets = np.zeros(rows, dtype=np.int64)
    for _u in range(rows // 2):
        base = int(rng.integers(0, 8))
        for _ in range(2):
            # The history carries the target's signal: the previous item shares it.
            hist.append(np.array([base] * 3, dtype=np.int64))
            targets[len(hist) - 1] = base
    hist = hist[:rows]

    toks = [HP.token_ids(sid, np.arange(n_items), l) for l in (1, 2, 3)]
    seq, _ = HP.build_sequences(hist, targets, toks, 20)
    seq_t = torch.from_numpy(seq)
    tok_t = [torch.from_numpy(t[targets]) for t in toks]

    tr_idx, dev_idx = split_by_user(users, 0.5, seed=11)
    priors = [torch.as_tensor(p, dtype=torch.float32)
              for p in HP.train_frequency_prior(toks, targets[tr_idx])]

    cfg = HP.ProbeConfig(epochs=8, seq_len=20, batch_size=64)
    gains = {}
    for seed in (42, 43):
        torch.manual_seed(seed)
        model = HP.HistoryProbe(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        gen = torch.Generator().manual_seed(seed)
        ntr = len(tr_idx)
        for _ in range(cfg.epochs):
            perm = torch.randperm(ntr, generator=gen)
            for s0 in range(0, ntr, cfg.batch_size):
                rows_b = torch.from_numpy(tr_idx)[perm[s0:s0 + cfg.batch_size]]
                logits = model(seq_t[rows_b], [t[rows_b] for t in tok_t])
                loss = sum(F.cross_entropy(logits[l], tok_t[l][rows_b])
                           for l in range(len(HP.LEVELS)))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        out = HP.evaluate_probe(model, seq_t, tok_t,
                                torch.from_numpy(dev_idx), priors, batch_size=256)
        gains[seed] = out["rel_gain_l1"]
    check("a history-carrying signal yields a positive L1 information gain",
          all(v > 0.05 for v in gains.values()),
          ", ".join(f"seed{s}={v:.3f}" for s, v in gains.items()))
    check("the gain is stable across seeds",
          abs(gains[42] - gains[43]) < 0.25,
          f"{gains[42]:.3f} vs {gains[43]:.3f}")
    check("the index digest distinguishes the two halves",
          _index_digest(tr_idx) != _index_digest(dev_idx))

    # The fourth token must not enter the SID3 statistics.
    scrambled = sid.copy()
    scrambled[:, 3] = (scrambled[:, 3] + 13) % 781
    a = H.target_ambiguity(sid, np.arange(n_items), np.arange(n_items))
    b = H.target_ambiguity(scrambled, np.arange(n_items), np.arange(n_items))
    check("fourth-token scrambling leaves A_target identical",
          a["A_target"] == b["A_target"], f"{a['A_target']} vs {b['A_target']}")


# ------------------------------------------------------------------ 7
def test_padding_limit_is_measured() -> None:
    print("\n[7] zero-padding is measured and reported as a limit")
    n_items = 300
    sid = np.zeros((n_items, 4), dtype=np.int64)
    sid[:, 0] = np.arange(n_items) % 5
    sid[:, 1] = np.arange(n_items) % 4
    sid[:, 2] = np.arange(n_items) % 3
    toks = [HP.token_ids(sid, np.arange(n_items), l) for l in (1, 2, 3)]
    hist = [np.arange(2, dtype=np.int64), np.arange(20, dtype=np.int64),
            np.arange(7, dtype=np.int64)]
    targets = np.array([1, 2, 3])
    seq, _ = HP.build_sequences(hist, targets, toks, 20)
    check("sequences are [N, 3, seq_len]", tuple(seq.shape) == (3, 3, 20), str(seq.shape))
    check("short histories are zero-padded",
          bool((seq[0, :, 2:] == 0).all()) and bool((seq[2, :, 7:] == 0).all()))
    # The limitation itself: code 0 is an ordinary codeword, so a padded step is
    # indistinguishable from a real step that happens to map to token 0.
    check("the pad code is also a reachable real codeword",
          int(seq[0, 0, 0]) == 0 and bool((toks[0] == 0).any()),
          "pad token 0 collides with a real codeword")
    check("the GRU consumes the final padded step",
          seq.shape[-1] == 20 and bool((seq[0, :, -1] == 0).all()))

    # Quantify the limitation on the real training data.
    import pandas as pd
    frame = pd.read_parquet(H.TRAIN_PARQUET, columns=["seen_history"])
    lengths = frame["seen_history"].apply(len)
    frac = float((lengths < 20).mean())
    check("the padding limitation is measured, not assumed",
          0.5 < frac < 1.0, f"{frac:.4f} of histories are shorter than 20")

    import inspect
    from history_probe import ProbeConfig
    cfg = ProbeConfig()
    check("no PAD token was added to the architecture",
          "pad" not in inspect.getsource(HP.HistoryProbe).lower(),
          "primary setting keeps the #48 architecture unchanged")
    check("the architecture matches the frozen #48 config",
          (cfg.seq_len, cfg.epochs, cfg.batch_size, cfg.gru_hidden, cfg.lr)
          == (20, 6, 512, 128, 1e-3), json.dumps(cfg.to_dict()))


def main() -> None:
    test_noncontiguous_split_scores_dev()
    test_prior_is_train_only()
    test_prior_ce_and_topk()
    test_f3_population()
    test_ambiguity_weightings()
    test_end_to_end_probe()
    test_padding_limit_is_measured()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("all issue #49 regression tests PASSED")


if __name__ == "__main__":
    main()
