"""taskB option_D wrapper (issue23_option_d 架构): BoundedWeightedMixedCurvatureConditionerPerLayer.

薄转发: exec 归档在 taskB/stage3/_archive/ 的原始训练脚本, 调 load_wrapper_cls() 取类.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_ARCHIVE = _ROOT / "taskB/stage3/_archive/taskB_stage3_issue23_option_d.py"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "HG-Rec/model"))
sys.path.insert(0, str(_ROOT / "HG-Rec/data"))

_spec = importlib.util.spec_from_file_location("wrap_taskB_option_d", str(_ARCHIVE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def load_wrapper_cls():
    return _mod.load_wrapper_cls()
