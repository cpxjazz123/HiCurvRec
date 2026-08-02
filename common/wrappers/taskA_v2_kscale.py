"""taskA v2 wrapper (kappa_scale_recontinue 架构): HG_Rec_with_BoundedAdapter (无 kappa_logits, 旧).

薄转发: exec 归档在 taskA/stage3/_archive/ 的原始训练脚本, 导出 wrapper 类.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_ARCHIVE = _ROOT / "taskA/stage3/_archive/taskA_stage3_kappa_scale_recontinue.py"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "HG-Rec/model"))
sys.path.insert(0, str(_ROOT / "HG-Rec/data"))

_spec = importlib.util.spec_from_file_location("wrap_taskA_v2_kscale", str(_ARCHIVE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HG_Rec_with_BoundedAdapter = _mod.HG_Rec_with_BoundedAdapter
BoundedKappaScaleConditioner = _mod.BoundedKappaScaleConditioner
get_t5_config = _mod.get_t5_config


def load_wrapper_cls():
    return HG_Rec_with_BoundedAdapter, get_t5_config
