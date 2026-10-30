"""The cell-removal construction alone, for a bench run of its own."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_r10", HERE / "metrics_round10.py")
_r10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r10)

METRICS = {"hbaa_offcell_expected_rank": _r10.hbaa_offcell_expected_rank}
FREE_CONSTANTS = {"hbaa_offcell_expected_rank": 0}
