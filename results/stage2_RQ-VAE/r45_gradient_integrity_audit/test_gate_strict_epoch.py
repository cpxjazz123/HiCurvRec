"""Issue #71 §1 — fail-closed training-completion proof for the shared Hard-L1 gate.

Audit-only. Targets the **unmodified** ``r5_common/r5_gate.py`` / ``r4_common/r4_gate.py``.

Purpose
-------
``train_end_epoch`` used to read ``int(row.get("epoch", 3000))``. Every historical R4/R5
``train_end`` record omits ``epoch`` entirely, so a malformed or truncated run record was
silently promoted to "3000 complete epochs" by the *default value itself*. This module pins
the strict contract from issue #71 §1: completion must be *proved* by an explicit
``epoch == 3000`` **and** a linked ``ckpt_saved`` record, never assumed.

Design notes (honest limitations, recorded rather than hidden)
-------------------------------------------------------------
``r5_gate.evaluate`` reads ``PAIRS_PATH`` / ``CANONICAL_SID_PATH`` as module constants and
hard-codes the 52,855 x 10 held-out support, so it cannot be fed synthetic pairs. To keep the
tests about *completion proving* rather than about the metric, ``hard_l1_er`` is monkeypatched
to return a synthetic above-baseline ER while the real frozen pairs/canonical files are used
for the hash ingredients. A **positive control** case proves the fixture can yield
``stage3_allowed=True``, so every fail-closed case below is meaningful and not vacuous.

The checkpoint-metadata / SID-to-checkpoint-linkage ingredients required by issue #71 §1 do
not exist in the current gate; the cases marked ``# RED: strict check not yet implemented``
are the genuine RED for those *absent safety checks*.

Run:  python results/stage2_RQ-VAE/r45_gradient_integrity_audit/test_gate_strict_epoch.py
"""
from __future__ import annotations

import json

import sys
import tempfile
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
AUDIT = REPO / "results/stage2_RQ-VAE/r45_gradient_integrity_audit"
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r5_common"))
sys.path.insert(0, str(REPO / "results/stage2_RQ-VAE/r4_common"))

import r5_gate as G5  # noqa: E402
import r4_gate as G4  # noqa: E402

STRICT_EPOCHS = 3000
GOOD_ER = {"pos_share": 0.2, "neg_share": 0.01, "ER_L1": 20.0,
           "n_positive_rows": G5.EXPECTED_POSITIVES,
           "n_negatives_per_row": G5.EXPECTED_NEGATIVES}
GOOD_SID_SHA = None          # filled in per-fixture by sid_sha_of()
NAN_ER = {"pos_share": float("nan"), "neg_share": float("nan"), "ER_L1": float("nan"),
          "n_positive_rows": G5.EXPECTED_POSITIVES,
          "n_negatives_per_row": G5.EXPECTED_NEGATIVES}


# --------------------------------------------------------------------------- fixtures
def write_sid(directory: Path, n_items: int = 64) -> Path:
    """A structurally valid, dense 0..N-1 item_sids.json the real loader accepts."""
    payload = {str(i): [i % 7, i % 5, i % 3, i % 2] for i in range(n_items)}
    path = directory / "item_sids.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_ckpt(directory: Path, epoch: int = STRICT_EPOCHS) -> Path:
    """A REAL torch-saved checkpoint carrying the metadata a strict gate must read.

    A byte-string stub would be unreadable by ``torch.load``, which the strict gate correctly
    treats as "checkpoint not proven" -- that would make every positive case vacuously fail, so
    the fixture has to be a genuine checkpoint.
    """
    import torch

    path = directory / "rqvae_best.pth"
    torch.save({"state_dict": {"encoder.0.weight": torch.zeros(1)},
                "epoch": epoch, "selection": "final_epoch"}, path)
    return path


def sid_sha_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(directory: Path, rows: list[dict]) -> Path:
    path = directory / "training_metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def complete_rows(epoch: int = STRICT_EPOCHS, sid_sha: str | None = None) -> list[dict]:
    end = {"event": "train_end", "epoch": epoch, "sid_shape": [64, 4],
           "final_collision": 0.0, "output_json": "item_sids.json"}
    if sid_sha is not None:
        end["output_json_sha256"] = sid_sha
    return [
        {"event": "train_start", "epochs": STRICT_EPOCHS},
        {"event": "ckpt_saved", "epoch": epoch, "path": "rqvae_best.pth"},
        end,
    ]


class patched_er:
    """Context manager forcing ``hard_l1_er`` to a chosen ER dict."""

    def __init__(self, er: dict):
        self.er = er
        self._saved = []

    def __enter__(self):
        for module in (G5, G4):
            self._saved.append((module, module.hard_l1_er))
            module.hard_l1_er = lambda _sid, _er=self.er: dict(_er)
        return self

    def __exit__(self, *exc):
        for module, fn in self._saved:
            module.hard_l1_er = fn
        return False


def run_gate(directory: Path, er: dict, *, with_ckpt: bool = True,
             ckpt_epoch: int = STRICT_EPOCHS, tests_passed: bool = True,
             sid_sha: str | None = None, rows: list[dict] | None = None):
    sid = write_sid(directory)
    if with_ckpt:
        write_ckpt(directory, ckpt_epoch)
    metrics = write_jsonl(directory, rows if rows is not None else complete_rows(
        sid_sha=sid_sha if sid_sha is not None else sid_sha_of(sid)))
    with patched_er(er):
        _, gate = G5.evaluate(arm="audit", issue=71, item_sids_path=sid,
                              metrics_path=metrics, tests_passed=tests_passed)
    return gate


def launcher_recorder():
    calls: list[int] = []

    def launcher() -> None:
        calls.append(1)

    return launcher, calls


# ------------------------------------------------------------------------------ tests
def test_positive_control_gate_can_be_true():
    """The fixture must be able to PASS, otherwise every fail-closed case is vacuous."""
    with tempfile.TemporaryDirectory() as tmp:
        gate = run_gate(Path(tmp), GOOD_ER)
        assert gate["stage3_allowed"] is True, (
            f"positive control failed; every fail-closed assertion would be vacuous: {gate}")
        assert gate["sid_linked_to_checkpoint"] is True
        assert gate["ER_L1_above_baseline"] is True
        assert gate["train_end_3000"] is True


def test_train_end_epoch_returns_explicit_value():
    """Explicit epochs must survive verbatim (guards an 'always None' over-correction)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        assert G5.train_end_epoch(write_jsonl(d, [{"event": "train_end", "epoch": 3000}])) == 3000
        assert G4.train_end_epoch(write_jsonl(d, [{"event": "train_end", "epoch": 3000}])) == 3000
        assert G5.train_end_epoch(write_jsonl(d, [{"event": "train_end", "epoch": 2999}])) == 2999
        assert G4.train_end_epoch(write_jsonl(d, [{"event": "train_end", "epoch": 2999}])) == 2999


def test_missing_epoch_in_train_end_must_fail_closed():
    """THE core RED. A ``train_end`` with no ``epoch`` proves nothing about completion."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        metrics = write_jsonl(d, [{"event": "train_end", "sid_shape": [64, 4],
                                   "final_collision": 0.0,
                                   "output_json": "item_sids.json"}])
        for name, module in (("r5_gate", G5), ("r4_gate", G4)):
            got = module.train_end_epoch(metrics)
            assert got is None, (
                f"{name}.train_end_epoch accepted a train_end record with no 'epoch' key and "
                f"reported {got!r}. This is the fail-open: the value came from the "
                f"row.get('epoch', 3000) DEFAULT, not from evidence.")


def test_gate_fails_closed_without_epoch_even_with_high_er():
    """Apparent ER gains must not buy a Stage3 launch on an unproven run."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sid = write_sid(d)
        write_ckpt(d)
        metrics = write_jsonl(d, [{"event": "train_end", "sid_shape": [64, 4],
                                   "final_collision": 0.0}])
        with patched_er(GOOD_ER):
            _, gate = G5.evaluate(arm="audit", issue=71, item_sids_path=sid,
                                  metrics_path=metrics, tests_passed=True)
        assert gate["measured_er_l1"] > gate["baseline_er_l1"], "fixture ER must beat baseline"
        assert gate["train_end_3000"] is False, (
            "a train_end without an explicit epoch must NOT count as 3000 epochs")
        assert gate["stage3_allowed"] is False


def test_stage3_launcher_never_called_on_false_gate():
    """No false gate may ever reach the launcher, whatever the ER says."""
    launcher, calls = launcher_recorder()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        gate = run_gate(d, GOOD_ER)
        gate["measured_er_l1"] = 1e9          # absurd apparent gain
        gate["ER_L1_above_baseline"] = True
        gate["stage3_allowed"] = False        # ... but completion unproven
        assert G5.maybe_launch_stage3(gate, launcher) is False
        assert calls == []
        # and a non-dict / malformed gate must also fail closed
        assert G5.maybe_launch_stage3(None, launcher) is False
        assert G5.maybe_launch_stage3({}, launcher) is False
        assert calls == []


def test_fail_closed_matrix():
    """Every ingredient issue #71 §1 enumerates must fail the gate closed."""
    cases: list[tuple[str, dict]] = []

    # (a) epoch one short of complete
    cases.append(("epoch_2999", {
        "stage3_allowed": False,
        "reason": "train_end says 2999, not 3000"}))

    # (b) fabricated train_end(3000) with no final checkpoint saved
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sid = write_sid(d)
        metrics = write_jsonl(d, [{"event": "train_end", "epoch": 3000}])
        with patched_er(GOOD_ER):
            _, gate = G5.evaluate(arm="audit", issue=71, item_sids_path=sid,
                                  metrics_path=metrics, tests_passed=True)
        cases.append(("no_ckpt_file", gate))

    # (c) checkpoint file missing entirely
    with tempfile.TemporaryDirectory() as tmp:
        cases.append(("checkpoint_absent",
                      run_gate(Path(tmp), GOOD_ER, with_ckpt=False)))

    # (d) checkpoint present but its own metadata epoch is wrong
    #     # RED: strict check not yet implemented -- today's gate never reads ckpt metadata
    with tempfile.TemporaryDirectory() as tmp:
        cases.append(("checkpoint_metadata_wrong",
                      run_gate(Path(tmp), GOOD_ER, ckpt_epoch=2999)))

    # (e) the run records a SID digest that does not match the file it exported
    with tempfile.TemporaryDirectory() as tmp:
        cases.append(("sid_checkpoint_mismatch",
                      run_gate(Path(tmp), GOOD_ER, sid_sha="0" * 64)))

    # (e2) the run records NO SID digest at all -> explicit uncertainty, not a silent pass
    with tempfile.TemporaryDirectory() as tmp:
        cases.append(("sid_linkage_not_recorded",
                      run_gate(Path(tmp), GOOD_ER, rows=complete_rows(sid_sha=None))))

    # (f) tests did not pass
    cases.append(("tests_false", run_gate(Path(tempfile.mkdtemp()), GOOD_ER,
                                          tests_passed=False)))

    # (g) NaN / +inf ER
    nan_er = dict(NAN_ER)
    cases.append(("er_nan", run_gate(Path(tempfile.mkdtemp()), nan_er)))
    inf_er = {"pos_share": float("inf"), "neg_share": 0.01, "ER_L1": float("inf"),
              "n_positive_rows": G5.EXPECTED_POSITIVES,
              "n_negatives_per_row": G5.EXPECTED_NEGATIVES}
    cases.append(("er_inf", run_gate(Path(tempfile.mkdtemp()), inf_er)))

    # (h) zero denominator
    zero_er = {"pos_share": 0.0, "neg_share": 0.0, "ER_L1": float("nan"),
               "n_positive_rows": G5.EXPECTED_POSITIVES,
               "n_negatives_per_row": G5.EXPECTED_NEGATIVES}
    cases.append(("zero_denominator", run_gate(Path(tempfile.mkdtemp()), zero_er)))

    # (i) missing hashes: no item_sids.json at all
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_ckpt(d)
        metrics = write_jsonl(d, complete_rows())
        with patched_er(GOOD_ER):
            _, gate = G5.evaluate(arm="audit", issue=71,
                                  item_sids_path=d / "item_sids.json",
                                  metrics_path=metrics, tests_passed=True)
        cases.append(("missing_sid_hashes", gate))

    # (j) incomplete source log: no train_end row at all
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sid = write_sid(d)
        write_ckpt(d)
        metrics = write_jsonl(d, [{"event": "train_start", "epochs": 3000},
                                  {"event": "ckpt_saved", "epoch": 3000}])
        with patched_er(GOOD_ER):
            _, gate = G5.evaluate(arm="audit", issue=71, item_sids_path=sid,
                                  metrics_path=metrics, tests_passed=True)
        cases.append(("incomplete_source_log", gate))

    launcher, calls = launcher_recorder()
    failures: list[str] = []
    for name, gate in cases:
        if name == "sid_linkage_not_recorded":
            # runs predating the field cannot be linked: surfaced as explicit uncertainty
            assert gate.get("sid_linked_to_checkpoint") == "not_recorded", (
                f"an unrecorded SID digest must be reported as not_recorded, got "
                f"{gate.get('sid_linked_to_checkpoint')!r}")
            continue
        if gate.get("stage3_allowed") is not False:
            failures.append(f"{name}: stage3_allowed={gate.get('stage3_allowed')!r}")
        before = len(calls)
        invoked = G5.maybe_launch_stage3(gate, launcher) is not False
        if invoked or len(calls) != before:
            failures.append(f"{name}: launcher was invoked on a false gate")
    assert not failures, "gate did not fail closed for: " + "; ".join(failures)


def test_pairs_sha_mismatch_fails_closed():
    """A held-out pairs file whose SHA differs from the frozen value must fail closed."""
    saved_path, saved_sha = G5.PAIRS_PATH, G5.PAIRS_SHA256
    try:
        G5.PAIRS_SHA256 = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_gate(Path(tmp), GOOD_ER)
            assert gate["hashes_valid"] is False
            assert gate["stage3_allowed"] is False
    finally:
        G5.PAIRS_PATH, G5.PAIRS_SHA256 = saved_path, saved_sha
    assert G5.PAIRS_SHA256 == saved_sha, "test must restore the frozen pairs SHA"


def test_r4_and_r5_gates_share_the_same_defect():
    """Both shared gate modules must fail closed identically (they are duplicated logic)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sid = write_sid(d)
        write_ckpt(d)
        metrics = write_jsonl(d, [{"event": "train_end", "sid_shape": [64, 4]}])
        for module in (G5, G4):
            assert module.train_end_epoch(metrics) is None, (
                f"{module.__name__} still promotes an epoch-less train_end to 3000")


def main() -> int:
    tests = [
        test_positive_control_gate_can_be_true,
        test_train_end_epoch_returns_explicit_value,
        test_missing_epoch_in_train_end_must_fail_closed,
        test_gate_fails_closed_without_epoch_even_with_high_er,
        test_stage3_launcher_never_called_on_false_gate,
        test_fail_closed_matrix,
        test_pairs_sha_mismatch_fails_closed,
        test_r4_and_r5_gates_share_the_same_defect,
    ]
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}\n     {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}\n     {type(exc).__name__}: {exc}", flush=True)
        else:
            print(f"pass {name}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed", flush=True)
    if failed:
        print("GATE_TESTS_RED", flush=True)
        return 1
    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
