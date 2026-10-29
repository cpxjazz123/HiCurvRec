"""The hyp_bucket_reach_lift_l1 construction alone, for a bench run of its own."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_all", HERE / "metrics_all.py")
_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_all)

METRICS = {"hyp_bucket_reach_lift_l1": _all.METRICS["hyp_bucket_reach_lift_l1"]}
FREE_CONSTANTS = {"hyp_bucket_reach_lift_l1": _all.FREE_CONSTANTS.get("hyp_bucket_reach_lift_l1", 0)}
