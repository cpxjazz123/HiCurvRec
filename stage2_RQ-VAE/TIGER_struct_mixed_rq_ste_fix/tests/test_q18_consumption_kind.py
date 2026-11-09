"""A same-item change of the behaviour kind must be rejected before a report,
because the actual h then differs from the reported behaviour."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_q9 as P
import training_hooks as H


def test_a_kind_only_consumption_change_is_rejected_without_a_report():
    original = H.structure_term

    def flipped(model, embeddings, graph, rng, **kw):
        triples = list(kw["triples"])
        centre, a, b, kind = triples[0]
        triples[0] = (centre, a, b, "wedge" if kind == "triangle" else "triangle")
        kw["triples"] = triples
        return original(model, embeddings, graph, rng, **kw)

    H.structure_term = flipped
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                P.run(Path(d))
            except RuntimeError as exc:
                assert "differ" in str(exc)
            else:
                raise AssertionError("a kind-only consumption change was accepted")
            assert not list(Path(d).rglob("factors.json"))
    finally:
        H.structure_term = original


if __name__ == "__main__":
    test_a_kind_only_consumption_change_is_rejected_without_a_report()
    print("ok")
