"""P0 acceptance tests for the GeneRec Harness² integration (issue #101 §D/P0).

They run with the framework's interpreter, because the package needs ``harness2`` and
Python 3.12:

    /home/wlia0047/ar57_scratch/wenyu/harness2/.venv/bin/python -m pytest \
        tools/harness2_genrec/tests -q

Zero GPU, zero network, zero credentials.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness2.core.spec import ChannelSpec  # noqa: E402

from tools.harness2_genrec import cli, tasks  # noqa: E402
from tools.harness2_genrec.adapter import GenrecAdapter, repo_surface_tree  # noqa: E402
from tools.harness2_genrec.proposer import (  # noqa: E402
    _CORRECT_VERIFIER,
    _HARDCODED_VERIFIER,
    ScriptedProposer,
)

OFFLINE_CHANNEL = ChannelSpec("opencode", "local/deterministic")
TASK_A = "verify_masked_sum"
TASK_B = "verify_masked_sum_b"


def _adapter(tmp_path: Path) -> GenrecAdapter:
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    return GenrecAdapter(REPO_ROOT, runs_root, channel=OFFLINE_CHANNEL)


def _sweep(tmp_path: Path):
    adapter = _adapter(tmp_path)
    prop = ScriptedProposer(OFFLINE_CHANNEL)
    report = cli.run_sweep(adapter, prop, adapter.runs_root, k=1, modes=("agg",))
    return adapter, report, adapter.runs_root


def _task_dir(runs_root: Path, task_id: str) -> Path:
    hits = sorted(runs_root.glob(f"adapt/*/{task_id}"))
    assert hits, f"no task directory for {task_id} under {runs_root}"
    return hits[0]


def _harness(tmp_path: Path, body: str) -> Path:
    harness = tmp_path / "harness"
    (harness / "scripts").mkdir(parents=True, exist_ok=True)
    (harness / "scripts" / "verify_masked_sum.py").write_text(body, encoding="utf-8")
    return harness


# -- the pipeline ---------------------------------------------------------


def test_offline_sweep_writes_evidence_selection_and_library(tmp_path: Path) -> None:
    adapter, report, runs_root = _sweep(tmp_path)

    done = sum(int(d.get("done", 0)) for d in report["domains"])
    failed = sum(int(d.get("failed", 0)) for d in report["domains"])
    assert (done, failed) == (len(tasks.all_tasks()), 0)

    task_dir = _task_dir(runs_root, TASK_A)
    summary = json.loads((task_dir / "summary.json").read_text())
    assert summary["status"] == "ok"

    selection = json.loads((task_dir / "selection" / "agg1" / "selection.json").read_text())
    assert selection["ok"] is True
    assert selection["chosen_label"] == "agg1"

    # the evidence tree the proposer was given
    context = task_dir / "context" / "inv1"
    assert (context / "task.md").is_file()
    assert (context / "harness" / "CLAUDE.md").is_file()
    assert (context / "rollouts").is_dir()
    assert (context / "deliverables").is_dir()

    # the library, one entry per task with the reason it was adopted
    library = list(runs_root.glob("library/genrec/k1-local/agg/mechanism_verification"))
    assert library, "no library directory was written"
    index = (library[0] / "INDEX.md").read_text()
    assert TASK_A in index and TASK_B in index
    experience = (library[0] / "harnesses" / TASK_A / "EXPERIENCE.md").read_text()
    assert "lesson" in experience and "evidence" in experience


def test_second_task_sees_the_first_task_lessons(tmp_path: Path) -> None:
    """The improvement harness has to travel: the second task's evidence carries the first."""
    _adapter_, _report, runs_root = _sweep(tmp_path)
    second = _task_dir(runs_root, TASK_B) / "context" / "inv1" / "library"
    assert second.is_dir(), "the second task had no library in its evidence"
    entries = sorted(p.name for p in (second / "harnesses").iterdir())
    assert TASK_A in entries
    assert (second / "harnesses" / TASK_A / "EXPERIENCE.md").is_file()


def test_the_task_pool_has_a_development_and_a_held_back_split() -> None:
    """A transfer claim needs tasks the harness never saw while it was being edited."""
    development = tasks.tasks_for_split("train")
    held_back = tasks.tasks_for_split("test")
    assert development and held_back
    assert not set(development) & set(held_back)
    assert set(tasks.tasks_for_split("all")) == set(development) | set(held_back)
    served = {
        task_id: tasks.served_fixtures(task_id)[0].read_text(encoding="utf-8")
        for task_id in development + held_back
    }
    for task_id in held_back:
        for other in development:
            assert served[task_id] != served[other], (
                "a held-back task must not share its fixture with a development one"
            )


def test_sweep_is_deterministic(tmp_path: Path) -> None:
    def chosen(runs_root: Path) -> list[str]:
        out = []
        for summary_path in sorted(runs_root.glob("adapt/*/*/summary.json")):
            doc = json.loads(summary_path.read_text())
            for mode, block in doc["modes"].items():
                out.append(f"{doc['task_id']}:{mode}:{block['finals']}")
        return out

    _, _, first_root = _sweep(tmp_path / "one")
    _, _, second_root = _sweep(tmp_path / "two")
    assert chosen(first_root) == chosen(second_root)


# -- the gates ------------------------------------------------------------


def test_restating_candidate_is_rejected_before_it_is_rolled(tmp_path: Path) -> None:
    """The static gate must throw out a candidate that restates a declared value."""
    adapter, _report, runs_root = _sweep(tmp_path)
    summary = json.loads((_task_dir(runs_root, TASK_A) / "summary.json").read_text())
    stages = {s["label"]: s for s in summary["modes"]["agg"]["stages"]}

    assert stages["dom1"]["fell_back"] is True
    assert "literal" in stages["dom1"]["error"]
    assert stages["inv1"]["fell_back"] is False

    # the rejected candidate fell back to the base tree, which has no verifier at all
    verdict = json.loads(
        (runs_root / "executions").glob(f"*/{TASK_A}/dom1/verdict.json").__next__().read_text()
    )
    assert verdict["verifier_present"] is False


def test_static_gate_rejects_a_declared_literal(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    tree = repo_surface_tree(REPO_ROOT)
    tree["scripts/verify_masked_sum.py"] = _HARDCODED_VERIFIER
    try:
        adapter.validate_tree(tree)
    except ValueError as exc:
        assert "literal" in str(exc)
    else:  # pragma: no cover - the gate must fire
        raise AssertionError("the static gate accepted a restating verifier")


def test_static_gate_rejects_a_changed_frozen_threshold(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    tree = repo_surface_tree(REPO_ROOT)
    target = "scripts/history_hierarchy_features.py"
    assert "16.38953942232631" in tree[target], "fixture changed: no frozen threshold to edit"
    tree[target] = tree[target].replace("16.38953942232631", "16.5")
    try:
        adapter.validate_tree(tree)
    except ValueError as exc:
        assert "frozen threshold" in str(exc)
    else:  # pragma: no cover - the gate must fire
        raise AssertionError("the static gate accepted an edited frozen threshold")


def test_restating_verifier_is_caught_at_runtime(tmp_path: Path) -> None:
    """Even if the static gate is bypassed, the independent reference catches the lie."""
    adapter = _adapter(tmp_path)
    harness = _harness(tmp_path, _HARDCODED_VERIFIER)
    run_dir = adapter.execute(TASK_A, harness, "run-hardcoded")

    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["correct_pass"] is True, "it did reproduce the served fixture"
    assert verdict["mutation_detected"] == 0, "it claimed success on every held-out fixture"
    assert verdict["false_pass_count"] == len(tasks.heldout_files(TASK_A))
    assert adapter.delivered(run_dir) is False, "a false success must not be a deliverable"


def test_computing_verifier_is_deliverable_and_catches_every_mutant(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    harness = _harness(tmp_path, _CORRECT_VERIFIER)
    run_dir = adapter.execute(TASK_A, harness, "run-correct")

    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["correct_pass"] is True
    assert verdict["mutation_detected"] == verdict["mutation_total"] > 0
    assert verdict["false_pass_count"] == 0
    assert adapter.delivered(run_dir) is True

    score = adapter.score(run_dir, TASK_A)
    assert score is not None
    assert score["correctness_pass_rate"] == 1.0
    assert score["mutation_detection"] == 1.0
    assert score["false_pass_count"] == 0


def test_a_missing_verifier_is_not_a_deliverable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    harness = tmp_path / "empty-harness"
    harness.mkdir()
    run_dir = adapter.execute(TASK_A, harness, "run-empty")
    assert adapter.delivered(run_dir) is False
    score = adapter.score(run_dir, TASK_A)
    assert score is not None and score["correctness_pass_rate"] == 0.0
