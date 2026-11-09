"""A cancelled task must be claimable again once the run is resumed."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research.core.dag import TaskStore


def test_reopen_makes_a_cancelled_task_claimable():
    with tempfile.TemporaryDirectory() as tmp:
        dag = TaskStore(Path(tmp))
        dag.add("run1__literature", "literature")
        dag.cancel_prefix("run1__")
        assert dag.snapshot()["tasks"]["run1__literature"]["status"] == "cancelled"
        assert dag.claim("w2", lease_seconds=60, prefix="run1__") is None
        dag.reopen("run1__literature", "resume after cooperative cancel")
        again = dag.claim("w2", lease_seconds=60, prefix="run1__")
        assert again and again[0] == "run1__literature"


def test_a_cancelling_task_is_not_reopenable():
    # A worker that is still stopping has not released the task; reopening it here
    # would let a second agent run the same step.
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        dag = TaskStore(Path(tmp))
        dag.add("run1__literature", "literature")
        dag.claim("w1", lease_seconds=60, prefix="run1__")
        dag.cancel_prefix("run1__")
        assert dag.snapshot()["tasks"]["run1__literature"]["status"] == "cancelling"
        with pytest.raises(Exception):
            dag.reopen("run1__literature", "resume")


if __name__ == "__main__":
    test_reopen_makes_a_cancelled_task_claimable()
    print("ok")
