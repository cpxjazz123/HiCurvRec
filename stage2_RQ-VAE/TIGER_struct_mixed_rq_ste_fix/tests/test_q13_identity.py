"""Ordered identity, immutable runs, and the consumer-side checkpoint refusal."""
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import probe_q9


def identity_matches(report, triples):
    want = probe_q9.triples_digest([(c, a, b, k) for c, a, b, k in triples])
    return report["identity"]["triples_sha256"] == want


def test_same_count_different_items_is_rejected():
    triples = [(1, 2, 3, "triangle"), (4, 5, 6, "wedge")]
    digest = probe_q9.triples_digest(triples)
    report = {"identity": {"triples_sha256": digest}}
    assert identity_matches(report, triples)
    other = [(1, 2, 3, "triangle"), (4, 5, 7, "wedge")]
    assert not identity_matches(report, other)


def test_runs_are_immutable_and_the_real_consumer_accepts_the_identity():
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        first = probe_q9.run(root)
        second = probe_q9.run(root)
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        assert len(dirs) == 2, "a repeated run must not overwrite the first record"
        for report in (first, second):
            ident = report["identity"]
            assert ident["z_struct_rows"] == 3 * ident["triples_accepted"]
            assert ident["z_struct_rows"] == 3 * len(ident["triples"])
            assert len(ident["triples"]) == ident["triples_accepted"]
            assert identity_matches(report, [tuple(t) for t in ident["triples"]])
            assert probe_q9.validate_identity(report)


def test_a_truncated_checkpoint_is_refused_by_the_consumer():
    ckpt = Path("/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_struct_mixed_rq/rqvae_best.pth")
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    truncated = {k: v for i, (k, v) in enumerate(state.items()) if i > 2}
    with tempfile.TemporaryDirectory() as work:
        bad = Path(work) / "truncated.pth"
        torch.save(truncated, bad)
        try:
            probe_q9.run(Path(work) / "out", ckpt=bad)
        except SystemExit:
            pass
        else:
            raise AssertionError("the consumer must refuse a truncated checkpoint")
        assert not list((Path(work) / "out").glob("run-*/factors.json")), \
            "a refused run must not leave a success report"


def _forged(report, mutate):
    import copy
    forged = copy.deepcopy(report)
    mutate(forged["identity"])
    return forged


def test_the_consumer_rejects_forged_identities():
    base = {"identity": {
        "z_struct_rows": 135, "triples_accepted": 45,
        "triples": [[i, i + 1, i + 2, "triangle" if i % 2 else "wedge"] for i in range(45)],
    }}
    base["identity"]["triples_sha256"] = probe_q9.triples_digest(
        [tuple(t) for t in base["identity"]["triples"]])
    assert probe_q9.validate_identity(base)

    def rows(ident): ident["z_struct_rows"] = 99
    def item(ident): ident["triples"][0] = [0, 1, 4, ident["triples"][0][3]]
    def kind(ident): ident["triples"][0] = [0, 1, 2, "swapped"]
    def order(ident): ident["triples"] = list(reversed(ident["triples"]))

    for mutate in (rows, item, kind, order):
        forged = _forged(base, mutate)
        try:
            probe_q9.validate_identity(forged)
        except ValueError:
            continue
        raise AssertionError(("the consumer accepted a forged identity", mutate.__name__))


if __name__ == "__main__":
    test_same_count_different_items_is_rejected()
    test_runs_are_immutable_and_the_real_consumer_accepts_the_identity()
    test_a_truncated_checkpoint_is_refused_by_the_consumer()
    test_the_consumer_rejects_forged_identities()
    print("ok")
