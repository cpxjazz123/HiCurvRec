"""Unit tests for the latent-interest-hierarchy study. Written before the experiment.

Run: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis/latent_interest_hierarchy/test_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import data as D  # noqa: E402
import metrics as M  # noqa: E402
import models as MD  # noqa: E402

_ok = [True]


def req(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    _ok[0] = _ok[0] and bool(cond)


# --------------------------------------------------------------------------- data
def test_sequence_recovery() -> None:
    """The parquet rows are NOT in temporal order per user; the order lives in
    len(seen_history). A frame whose rows are deliberately shuffled must still
    recover the same sequence."""
    rows = [
        # user 0, written out of order on purpose
        dict(user=0, seen_history=[1, 2], target=3),
        dict(user=0, seen_history=[], target=1),
        dict(user=0, seen_history=[1], target=2),
        dict(user=1, seen_history=[], target=9),
        dict(user=1, seen_history=[9], target=7),
    ]
    f = pd.DataFrame(rows)
    seqs = D.user_train_sequences(f)
    req(seqs[0] == [1, 2, 3],
        f"T1 user sequence is recovered from len(seen_history), not file order ({seqs[0]})")
    req(seqs[1] == [9, 7], f"T1 second user ({seqs[1]})")
    f2 = f.sample(frac=1.0, random_state=3)
    req(D.user_train_sequences(f2) == seqs,
        "T1 shuffling the rows changes nothing")

    inst = D.train_instances(seqs)
    req(len(inst) == (3 - 1) + (2 - 1), f"T1 one instance per adjacent pair ({len(inst)})")
    req(inst[0][1][-1] == 1 and inst[0][2] == 2, f"T1 first pair is 1 -> 2 ({inst[0][1][-1]} -> {inst[0][2]})")


def test_evaluation_history_is_checked() -> None:
    """An evaluation history that is not a suffix of the user's train prefix must
    RAISE, not be silently accepted."""
    seqs = {0: [1, 2, 3]}
    good = pd.DataFrame({"user": [0], "seen_history": [[1, 2, 3]], "target": [4]})
    out = D.evaluation_rows(good, seqs)
    req(out == [(0, [1, 2, 3], 4)], f"T1 a matching history is accepted ({out})")
    bad = pd.DataFrame({"user": [0], "seen_history": [[7, 8]], "target": [4]})
    raised = False
    try:
        D.evaluation_rows(bad, seqs)
    except AssertionError:
        raised = True
    req(raised, "T1 a mismatching history raises rather than being used")


def test_shuffled_control_preserves_frequencies() -> None:
    """The order control must keep every user's item multiset exactly, so any
    difference is attributable to order alone."""
    seqs = {u: [3, 9, 4, 9, 11, 4] for u in range(20)}
    sh = D.shuffled_train_instances(seqs, seed=42)
    ref = D.shuffled_train_instances(seqs, seed=42)
    req([(a[1], a[2]) for a in sh] == [(a[1], a[2]) for a in ref],
        "T1 the shuffled control is reproducible under the same seed")
    shuf = D.shuffled_sequences(seqs, seed=42)
    req(all(sorted(shuf[u]) == sorted(seqs[u]) for u in seqs),
        "T1 every user's item MULTISET is preserved by the shuffle")
    req(any(shuf[u] != seqs[u] for u in seqs),
        "T1 and the order actually changed for at least one user")
    inst = D.shuffled_train_instances(seqs, seed=42)
    req(len(inst) == len(D.train_instances(seqs)),
        f"T1 the shuffled control has the same number of instances ({len(inst)})")
    freq = D.item_frequency(seqs)
    freq_shuf = D.item_frequency(shuf)
    req(np.array_equal(freq, freq_shuf), "T1 and identical global item frequencies")

def test_no_leakage_of_evaluation_targets() -> None:
    """No evaluation target may be a train target for the SAME user.

    The train prefix of a user is everything up to their last train target; the
    evaluation row's history must be exactly that prefix and its target must be new.
    """
    f = pd.DataFrame({"user": [0, 0], "seen_history": [[], [3]], "target": [3, 7]})
    seqs = D.user_train_sequences(f)
    req(seqs[0] == [3, 7], f"T1 the train prefix is both train targets ({seqs[0]})")
    ev = D.evaluation_rows(
        pd.DataFrame({"user": [0], "seen_history": [[3, 7]], "target": [11]}), seqs)
    req(ev[0][1] == [3, 7] and ev[0][2] == 11,
        f"T1 the evaluation pair is (train prefix -> new target) ({ev[0]})")
    req(ev[0][2] not in seqs[0], "T1 and the evaluation target is not a train target")
    inst = D.train_instances(seqs)
    tgts = [t for _u, _h, t in inst]
    req(tgts == [7], f"T1 training pairs predict TRAIN targets only ({tgts})")
    req(ev[0][2] not in tgts, "T1 and never the held-out evaluation target")


# -------------------------------------------------------------------------- models
def test_parameter_parity() -> None:
    """All arms must carry IDENTICAL parameters, so no NLL difference can be
    capacity."""
    counts = {}
    for mode in ("flat", "hier", "random_hier"):
        m = MD.InterestModel(500, C.EMBED_DIM, mode, seed=42)
        counts[mode] = m.n_parameters()
    req(len(set(counts.values())) == 1, f"T1 identical parameter counts ({counts})")
    expected = 500 * C.EMBED_DIM + C.N_INTEREST * C.EMBED_DIM + 500
    req(counts["flat"] == expected,
        f"T1 the count is exactly the item table, the 64 centroids and one bias per "
        f"item ({counts['flat']} == {expected})")


def test_grouping_semantics() -> None:
    hier = MD.InterestModel(100, C.EMBED_DIM, "hier", seed=42)
    rnd = MD.InterestModel(100, C.EMBED_DIM, "random_hier", seed=42)
    flat = MD.InterestModel(100, C.EMBED_DIM, "flat", seed=42)
    req(torch.equal(hier.group, flat.group),
        "T1 the hierarchical grouping is the identity partition")
    req(sorted(rnd.group.reshape(-1).tolist()) == list(range(C.N_INTEREST)),
        "T1 the random control is a permutation of the same leaves")
    req(not torch.equal(rnd.group, hier.group), "T1 and it is a different partition")
    rnd2 = MD.InterestModel(100, C.EMBED_DIM, "random_hier", seed=42)
    req(torch.equal(rnd.group, rnd2.group), "T1 and it is FIXED before training")
    rnd3 = MD.InterestModel(100, C.EMBED_DIM, "random_hier", seed=43)
    req(not torch.equal(rnd.group, rnd3.group),
        "T1 while still varying with the run seed")


def test_score_is_dot_with_profile() -> None:
    """The item score must be <E_i, m(u)> in every arm, so the scoring function is
    identical and only the bottleneck differs."""
    torch.manual_seed(0)
    B, L = 5, 4
    H = torch.randint(1, 100, (B, L, C.EMBED_DIM)).float()
    MASK = torch.ones(B, L, 1)
    for mode in ("flat", "hier", "random_hier"):
        m = MD.InterestModel(100, C.EMBED_DIM, mode, seed=1)
        prof = m.profile(H, MASK)
        out = m(H, MASK)
        req(tuple(out.shape) == (B, 100), f"T1 {mode}: score shape {tuple(out.shape)}")
        raw = prof @ m.embed.weight.t() + m.bias
        req(torch.allclose(out[:, 1:], raw[:, 1:], atol=1e-5),
            f"T1 {mode}: score equals the item table dotted with the profile, plus bias")
        req(float(out[:, 0].max()) < -1e8,
            f"T1 {mode}: and the padding column is masked out of the softmax")


def test_profile_depends_on_the_bottleneck() -> None:
    """hier and random_hier must produce DIFFERENT profiles for the same input,
    otherwise the control is not actually a different function."""
    torch.manual_seed(1)
    H = torch.randn(6, 5, C.EMBED_DIM)
    a = MD.InterestModel(100, C.EMBED_DIM, "hier", seed=2)
    b = MD.InterestModel(100, C.EMBED_DIM, "random_hier", seed=2)
    b.centroids.data = a.centroids.data.clone()
    mk = torch.ones(6, 5, 1)
    req(not torch.allclose(a.profile(H, mk), b.profile(H, mk), atol=1e-5),
        "T1 the same centroids give different profiles under the two groupings")


def test_arm_to_mode_mapping() -> None:
    """Regression: the negative control must map to its own mode. A prefix rule once
    mapped random_hier onto hier, so the control reproduced the treatment exactly."""
    req(set(C.ARM_TO_MODE) == set(C.ARMS),
        f"T1 every arm has a declared mode ({sorted(C.ARM_TO_MODE)})")
    req(C.ARM_TO_MODE["random_hier"] == "random_hier",
        "T1 random_hier maps to the random bottleneck, not to hier")
    req(C.ARM_TO_MODE["hier"] == "hier" and C.ARM_TO_MODE["flat"] == "flat",
        "T1 and the other arms map to themselves")
    req(C.ARM_TO_MODE["hier_shuffled"] == "hier"
        and C.ARM_TO_MODE["flat_shuffled"] == "flat",
        "T1 the shuffled controls keep their bottleneck, only the data order changes")
    modes = {}
    for arm in C.ARMS:
        modes.setdefault(C.ARM_TO_MODE[arm], []).append(arm)
    req(len(modes) == 3, f"T1 exactly three distinct bottlenecks are exercised ({modes})")


def test_padding_does_not_dilute_the_profile() -> None:
    """A user with two purchases in a twenty-slot batch must get the SAME profile as
    the same two purchases alone. An unmasked average would mix a uniform
    distribution into the profile and make short histories meaningless."""
    torch.manual_seed(3)
    for mode in ("flat", "hier", "random_hier"):
        m = MD.InterestModel(100, C.EMBED_DIM, mode, seed=5)
        short = torch.randn(1, 2, C.EMBED_DIM)
        padded = torch.zeros(1, 20, C.EMBED_DIM)
        padded[0, 18:] = short[0]
        mk_short = torch.ones(1, 2, 1)
        mk_pad = torch.zeros(1, 20, 1)
        mk_pad[0, 18:] = 1.0
        a = m.profile(short, mk_short)
        b = m.profile(padded, mk_pad)
        req(torch.allclose(a, b, atol=1e-5),
            f"T1 {mode}: padding changes nothing (max diff "
            f"{float((a - b).abs().max()):.2e})")
        req(float(a.abs().sum()) > 0, f"T1 {mode}: the profile is not degenerate")


def test_item_bias_can_express_the_popularity_prior() -> None:
    """Every arm must be able to express an item prior, so 'beats popularity' is a
    fair test rather than a handicap."""
    for mode in ("flat", "hier", "random_hier"):
        m = MD.InterestModel(100, C.EMBED_DIM, mode, seed=1)
        req(m.bias.numel() == 100, f"T1 {mode}: one bias per embedding row")
        req(m.bias.requires_grad, f"T1 {mode}: the bias is learned")
    a = MD.InterestModel(100, C.EMBED_DIM, "flat", seed=1)
    b = MD.InterestModel(100, C.EMBED_DIM, "hier", seed=1)
    req(a.n_parameters() == b.n_parameters(),
        f"T1 the bias keeps parameter counts identical ({a.n_parameters()})")


# ------------------------------------------------------------------------- metrics
def test_evaluate_indexes_the_offset_target() -> None:
    """End-to-end: `evaluate` must reproduce an INDEPENDENT hand computation exactly.

    The logits live in the model's shifted space (raw id t is row t+1), so an
    evaluation that gathers with the raw id reads the neighbouring item's
    probability. That bug made every NLL and Recall@10 in an earlier run meaningless
    while still looking plausible.
    """
    n_items = 60
    model = MD.InterestModel(n_items + 1, C.EMBED_DIM, "flat", seed=0)
    rng = torch.Generator().manual_seed(7)
    with torch.no_grad():
        model.embed.weight.copy_(torch.randn(n_items + 1, C.EMBED_DIM, generator=rng))
        model.centroids.copy_(torch.randn(C.N_INTEREST, C.EMBED_DIM, generator=rng) * 0.3)
        model.bias.copy_(torch.randn(n_items + 1, generator=rng))
    rows = [(u, [1, 3], 2) for u in range(7)] + [(u, [4, 5, 0], 5) for u in range(7)]
    got = M.evaluate(model, rows, "cpu", topk=10)

    # independent recomputation, item by item, straight from the definition
    ref_nll = 0.0
    ref_hit = 0
    model.eval()
    with torch.no_grad():
        for _u, h, t in rows:
            H = torch.as_tensor([[x + 1 for x in h]], dtype=torch.long)
            mask = torch.ones(1, len(h), 1)
            logits = model(model.embed(H) * mask, mask)
            logp = torch.log_softmax(logits.float(), dim=-1)
            ref_nll -= float(logp[0, t + 1])
            ref_hit += int(bool((logits.topk(10, dim=-1).indices == t + 1).any()))
    req(abs(got["nll"] - ref_nll / len(rows)) < 1e-6,
        f"T1 evaluate matches the hand-computed NLL ({got['nll']:.6f} vs "
        f"{ref_nll / len(rows):.6f})")
    req(got["recall_at_10"] == ref_hit / len(rows),
        f"T1 evaluate matches the hand-computed Recall@10 "
        f"({got['recall_at_10']:.4f} vs {ref_hit / len(rows):.4f})")
    # vacuity guard: the raw-id (unshifted) gather gives a different number
    off = 0.0
    with torch.no_grad():
        for _u, h, t in rows:
            H = torch.as_tensor([[x + 1 for x in h]], dtype=torch.long)
            mask = torch.ones(1, len(h), 1)
            logits = model(model.embed(H) * mask, mask)
            off -= float(torch.log_softmax(logits.float(), dim=-1)[0, t])
    req(abs(off / len(rows) - ref_nll / len(rows)) > 1e-3,
        "T1 gathering with the raw id differs, so the test is not vacuous")

    # the padding row must never carry probability mass or a top-k slot
    logits = model(torch.randn(2, 3, C.EMBED_DIM), torch.ones(2, 3, 1))
    req(float(torch.softmax(logits, dim=-1)[:, 0].max()) == 0.0,
        "T1 the padding row gets exactly zero probability")
    req(int((logits.topk(10, dim=-1).indices == 0).sum()) == 0,
        "T1 and never occupies a top-k slot")


def test_metric_definitions() -> None:
    n = 30
    y = np.zeros(n)
    req(abs(M.bootstrap_ci(y, np.mean, 50, 0)["point"]) < 1e-12, "T1 mean of zeros is 0")
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 0.5, 5000)
    c1 = M.bootstrap_ci(v, np.mean, 200, 42)
    c2 = M.bootstrap_ci(v, np.mean, 200, 42)
    req(c1 == c2, "T1 the bootstrap is reproducible under the same seed")
    c3 = M.bootstrap_ci(v, np.mean, 200, 43)
    req(c3["lo"] != c1["lo"], "T1 and varies with the seed")
    req(c1["lo"] < c1["point"] < c1["hi"], f"T1 the interval brackets the point ({c1})")

    a = rng.normal(0.0, 1.0, 4000)
    b = rng.normal(0.6, 1.0, 4000)
    d = M.paired_bootstrap_delta(a, b, np.mean, 400, 42)
    req(abs(d["delta"] - (np.mean(a) - np.mean(b))) < 1e-12,
        f"T1 the paired delta is exactly the observed mean difference ({d['delta']:.4f})")
    req(abs(d["delta"] + 0.6) < 0.06, f"T1 and it recovers the population gap of -0.6 ({d['delta']:.4f})")
    req(d["excludes_zero"], "T1 and a real shift excludes zero")
    same = M.paired_bootstrap_delta(a, a, np.mean, 400, 42)
    req(abs(same["delta"]) < 1e-12 and not same["excludes_zero"],
        "T1 while identical samples give a zero delta that does not exclude zero")


def test_partition_agreement() -> None:
    p = np.array([0, 0, 1, 1])
    req(M.partition_agreement(p, p, 2) == 1.0, "T1 a partition agrees with itself completely")
    req(M.partition_agreement(p, p[::-1], 2) == 1.0,
        "T1 relabelling the groups does not change agreement")
    q = np.array([0, 1, 0, 1])
    req(M.partition_agreement(p, q, 2) < 1.0, "T1 a different partition agrees on fewer pairs")


def test_learned_assignment_and_transitions() -> None:
    torch.manual_seed(0)
    n_items = 50
    m = MD.InterestModel(n_items + 1, C.EMBED_DIM, "hier", seed=42)   # +1 padding row
    a = M.learned_assignment(m)
    req(a.shape == (n_items,),
        f"T1 one group per REAL item, padding row dropped ({a.shape})")
    req(a.min() >= 0 and a.max() < C.N_COARSE,
        f"T1 the assignment lands in the 8 parents ({a.min()}..{a.max()})")
    req(MD.encode([0, 5]) == [1, 6] and 0 not in MD.encode([0, 5]),
        "T1 raw id 0 maps to a real embedding row, never the padding slot")
    rows = [(0, [1, 2], 3)] * 50
    tm = M.transition_mass(rows, lambda i: int(a[i]))
    req(0.0 <= tm["within_group_transition_rate"] <= 1.0, f"T1 rate in range ({tm})")
    req(tm["n_pairs"] == 50, f"T1 every non-empty row counted ({tm['n_pairs']})")


def main() -> int:
    print("=== data: sequences, splits, leakage, controls ===", flush=True)
    test_sequence_recovery()
    test_evaluation_history_is_checked()
    test_shuffled_control_preserves_frequencies()
    test_no_leakage_of_evaluation_targets()
    print("\n=== models: matched capacity and grouping semantics ===", flush=True)
    test_parameter_parity()
    test_grouping_semantics()
    test_score_is_dot_with_profile()
    test_profile_depends_on_the_bottleneck()
    test_arm_to_mode_mapping()
    test_padding_does_not_dilute_the_profile()
    test_item_bias_can_express_the_popularity_prior()
    print("\n=== metrics ===", flush=True)
    test_evaluate_indexes_the_offset_target()
    test_metric_definitions()
    test_partition_agreement()
    test_learned_assignment_and_transitions()
    print("\n" + "=" * 72)
    print("ALL STUDY TESTS PASS" if _ok[0] else "STUDY TESTS FAILED")
    print("=" * 72)
    return 0 if _ok[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())