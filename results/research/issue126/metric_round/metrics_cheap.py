"""The two cheap metrics of the round, scored together so a result exists now.

`hsmre` is not here: the formula as proposed costs 156 s per table, i.e. 4.2 h for
the 97 tables, and it is scored separately after its null-barycentre loop is
batched. A metric is never scored on a different sample than its competitors.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location("metrics_v1", Path(__file__).with_name("metrics_v1.py"))
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

METRICS = {
    "hyp_voronoi_margin_l1": _module.hyp_voronoi_margin_l1,
    "hyp_adj_r2_l1": _module.hyp_adj_r2_l1,
}
FREE_CONSTANTS = {"hyp_voronoi_margin_l1": 1, "hyp_adj_r2_l1": 2}
