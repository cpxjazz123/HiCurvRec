"""The producer must know, before the forward, whether the branch should run,
and must reject a legal sample whose structure graph silently disappeared."""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_q9 as P
import structure as S
import train_rqvae as T


def _no_report(d):
    return not list(Path(d).rglob("factors.json"))


def test_a_skipped_forward_on_a_legal_sample_is_rejected():
    saved = S.sample_triples
    original = __import__("training_hooks").structure_term

    def skipped(model, embeddings, graph, rng, **kw):
        return torch.zeros((), device=embeddings.device, requires_grad=True)

    import training_hooks as H
    H.structure_term = skipped
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                P.run(Path(d))
            except RuntimeError as exc:
                assert "did not run" in str(exc)
            else:
                raise AssertionError("a skipped structure forward was accepted")
            assert _no_report(d)
    finally:
        H.structure_term = original
        S.sample_triples = saved


def test_a_forward_that_runs_although_the_branch_is_off_is_rejected():
    import training_hooks as H
    original = H.structure_term

    def forced(model, embeddings, graph, rng, **kw):
        kw["behaviour_on"] = True
        return original(model, embeddings, graph, rng, **kw)

    saved = T.BEHAVIOUR_ON
    H.structure_term = forced
    T.BEHAVIOUR_ON = False
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                P.run(Path(d))
            except RuntimeError as exc:
                assert "although the branch is inactive" in str(exc)
            else:
                raise AssertionError("an unexpected structure forward was accepted")
            assert _no_report(d)
    finally:
        H.structure_term = original
        T.BEHAVIOUR_ON = saved


def test_the_consumed_order_is_not_taken_from_the_report_under_test():
    with tempfile.TemporaryDirectory() as d:
        report = P.run(Path(d))
    order = [tuple(r) for r in report["identity"]["triples"]]
    forged = dict(report)
    forged["identity"] = dict(report["identity"])
    forged["identity"]["triples"] = list(reversed(report["identity"]["triples"]))
    try:
        P.validate_identity(forged, consumed_triples=order)
    except ValueError:
        pass
    else:
        raise AssertionError("the report's own reordered triples were accepted")


if __name__ == "__main__":
    test_a_skipped_forward_on_a_legal_sample_is_rejected()
    test_a_forward_that_runs_although_the_branch_is_off_is_rejected()
    test_the_consumed_order_is_not_taken_from_the_report_under_test()
    print("ok")
