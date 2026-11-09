"""Run paths that do not depend on the DAG scheduler."""
from pathlib import Path

from .. import settings


def run_dir(run_id):
    if not isinstance(run_id, str) or len(run_id) != 20 or not run_id.startswith("run_"):
        raise ValueError("unknown research run identifier")
    if not all(c in "0123456789abcdef" for c in run_id[4:]):
        raise ValueError("unsafe research run identifier")
    return settings.STATE_ROOT / "runs" / run_id
