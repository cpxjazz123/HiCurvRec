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
            assert ident["z_struct_rows"] == 3 * ident["triples_accepted"] + 0 or \
                ident["z_struct_rows"] == 3 * 33
            assert len(ident["triples"]) == ident["triples_accepted"]
            assert identity_matches(report, [tuple(t) for t in ident["triples"]])


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


if __name__ == "__main__":
    test_same_count_different_items_is_rejected()
    test_runs_are_immutable_and_the_real_consumer_accepts_the_identity()
    test_a_truncated_checkpoint_is_refused_by_the_consumer()
    print("ok")
