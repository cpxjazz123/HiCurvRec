"""The P0 task pool: frozen correctness tasks with a served spec and a private held-out set.

Each task ships two things:

* ``fixtures/<task>/task/`` -- the frozen spec the candidate may read (the input and
  the expected value of one fixture),
* ``fixtures/<task>/heldout/`` -- the private set. It never enters a model context
  (the adapter keeps it out of ``task_files`` and declares it in ``leak_globs``); it is
  what makes a false PASS detectable.

The two tasks share one verifier contract so a candidate harness written for one is
exercised on the other: that is the transfer the P0 library has to demonstrate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
FIXTURES = PACKAGE_DIR / "fixtures"

VERIFIER_SCRIPT = "scripts/verify_masked_sum.py"

_BRIEF = """\
Task `{task_id}`.

Ship `{script}` inside the harness. It takes exactly one argument: the path of a
fixture JSON file. It must

1. read the fixture and compute the masked weighted sum of its `input`: the sum of
   `w * v` over the terms whose `mask` entry is truthy, where a masked-out term
   contributes NOTHING at all;
2. compare its own computed value with the fixture's `expected` field within 1e-9;
3. print exactly two lines: `STATUS: PASS` or `STATUS: FAIL`, and `RESULT: <value>`.

The verifier must COMPUTE the value from the input. Writing the expected value into
the script is a forbidden substitution: the verifier would then agree with itself
instead of with the fixture.
"""


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    split: str  # "train" (development) or "test" (held back, blind)

    @property
    def brief(self) -> str:
        return _BRIEF.format(task_id=self.task_id, script=VERIFIER_SCRIPT)


TASKS: Tuple[Task, ...] = (
    Task("verify_masked_sum", "mechanism_verification", "train"),
    Task("verify_masked_sum_b", "mechanism_verification", "train"),
    # The held-back tasks. The harness is developed on the two above and never sees these
    # while it is being edited, so a sweep that transfers to them has learned the rule
    # rather than the fixture. Their numbers, shapes and codebook geometry are all
    # different from the development ones.
    Task("verify_masked_sum_c", "mechanism_verification", "test"),
    Task("verify_masked_sum_d", "mechanism_verification", "test"),
)

_BY_ID = {t.task_id: t for t in TASKS}


def all_tasks() -> list[str]:
    return [t.task_id for t in TASKS]


def tasks_for_split(split: str = "all") -> list[str]:
    """The tasks of one split.

    ``train`` is the development split: the harness is edited against it. ``test`` is the
    held-back split: the harness never sees those tasks while it is being edited, so a
    sweep that transfers to them has learned the rule rather than the fixture.
    """
    if split == "all":
        return all_tasks()
    if split in ("train", "dev"):
        return [t.task_id for t in TASKS if t.split == "train"]
    if split in ("test", "heldout"):
        return [t.task_id for t in TASKS if t.split == "test"]
    raise ValueError(f"unknown split {split!r}; use all | train | test")


def task(task_id: str) -> Task:
    try:
        return _BY_ID[task_id]
    except KeyError:
        raise ValueError(f"unknown task {task_id!r}; known: {sorted(_BY_ID)}") from None


def task_dir(task_id: str) -> Path:
    return FIXTURES / task_id


def served_dir(task_id: str) -> Path:
    """The frozen spec the candidate may read."""
    return task_dir(task_id) / "task"


def heldout_files(task_id: str) -> list[Path]:
    """The private fixtures. Never served to a candidate."""
    d = task_dir(task_id) / "heldout"
    return sorted(d.glob("*.json")) if d.is_dir() else []


def served_fixtures(task_id: str) -> list[Path]:
    d = served_dir(task_id)
    return sorted(d.glob("*.json")) if d.is_dir() else []


def expected_values() -> dict[str, float]:
    """Every expected value in the pool, task -> value (the served fixture's value)."""
    out: dict[str, float] = {}
    for t in TASKS:
        for f in served_fixtures(t.task_id):
            doc = json.loads(f.read_text(encoding="utf-8"))
            out[f"{t.task_id}/{f.name}"] = float(doc["expected"])
    return out


__all__ = [
    "FIXTURES",
    "PACKAGE_DIR",
    "TASKS",
    "Task",
    "VERIFIER_SCRIPT",
    "all_tasks",
    "expected_values",
    "heldout_files",
    "served_dir",
    "served_fixtures",
    "task",
    "task_dir",
    "tasks_for_split",
]
