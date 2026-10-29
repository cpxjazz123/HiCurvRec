"""HSMRE alone, so it can be scored on all 97 pairs after its null loop was batched."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location("metrics_v1", Path(__file__).with_name("metrics_v1.py"))
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

METRICS = {"hsmre": _module.hsmre}
FREE_CONSTANTS = {"hsmre": 2}
