"""A same-count rotation at the consumption entry must be rejected before the
report is written, because the gathered item order differs from the sample."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_q9 as P
import training_hooks as H


def test_a_rotated_consumption_is_rejected_without_a_report():
    original = H.structure_term

    def rotated(model, embeddings, graph, rng, **kw):
        triples = list(kw["triples"])
        kw["triples"] = triples[1:] + triples[:1]
        return original(model, embeddings, graph, rng, **kw)

    H.structure_term = rotated
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                P.run(Path(d))
            except RuntimeError as exc:
                assert "differ" in str(exc)
            else:
                raise AssertionError("a rotated consumption was accepted")
            assert not list(Path(d).rglob("factors.json"))
    finally:
        H.structure_term = original


if __name__ == "__main__":
    test_a_rotated_consumption_is_rejected_without_a_report()
    print("ok")
