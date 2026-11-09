"""The real report path must survive the switch and an empty sample, and must not
pretend a broken graph is NOT_APPLICABLE when the structure graph is active."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_q9 as P
import structure as S
import train_rqvae as T


def _run():
    with tempfile.TemporaryDirectory() as d:
        report = P.run(Path(d))
        ok = P.validate_identity(report, consumed_triples=[
            tuple(r) for r in report["identity"]["triples"]])
        return report, ok


def test_behaviour_off_reports_an_inactive_graph():
    saved = T.BEHAVIOUR_ON
    T.BEHAVIOUR_ON = False
    try:
        report, ok = _run()
    finally:
        T.BEHAVIOUR_ON = saved
    ident = report["identity"]
    assert ok
    assert ident["structure_forward"] is False
    assert ident["z_struct_rows"] == 0
    assert ident["structure_zero_reason"] == "behaviour_off"
    assert report["latent"]["structure"]["z_struct"] == "NOT_APPLICABLE"


def test_an_empty_sample_is_reported_as_zero_accepted_not_as_a_graph():
    saved = S.sample_triples
    S.sample_triples = lambda graph, size, rng: []
    try:
        report, ok = _run()
    finally:
        S.sample_triples = saved
    ident = report["identity"]
    assert ok
    assert ident["structure_forward"] is False
    assert ident["triples_accepted"] == 0
    assert ident["z_struct_rows"] == 0
    assert ident["structure_zero_reason"] == "no_accepted_triples"


def test_the_normal_path_still_carries_a_graph_and_a_bound_identity():
    report, ok = _run()
    ident = report["identity"]
    assert ok
    assert ident["structure_forward"] is True
    assert ident["z_struct_rows"] == 3 * ident["triples_accepted"]
    assert ident["triples_sha256"] == P.triples_digest(
        [tuple(r) for r in ident["triples"]])


def test_a_consumer_side_sample_mismatch_is_rejected():
    report, _ = _run()
    swapped = [tuple(r) for r in report["identity"]["triples"]]
    swapped[0] = (swapped[0][0], swapped[0][1], swapped[0][2] + 1, swapped[0][3])
    try:
        P.validate_identity(report, consumed_triples=swapped)
    except ValueError:
        pass
    else:
        raise AssertionError("a different consumed sample was accepted")


def test_an_active_graph_without_a_gradient_raises():
    report, _ = _run()
    forged = dict(report["identity"])
    forged = dict(report)
    forged["identity"] = dict(report["identity"])
    forged["identity"]["structure_forward"] = True
    forged["latent"] = {"structure": {"z_struct": None}}
    try:
        P.validate_identity(forged)
    except ValueError:
        pass
    else:
        raise AssertionError("an active graph with no structure gradient was accepted")


if __name__ == "__main__":
    test_behaviour_off_reports_an_inactive_graph()
    test_an_empty_sample_is_reported_as_zero_accepted_not_as_a_graph()
    test_the_normal_path_still_carries_a_graph_and_a_bound_identity()
    test_a_consumer_side_sample_mismatch_is_rejected()
    test_an_active_graph_without_a_gradient_raises()
    print("ok")
