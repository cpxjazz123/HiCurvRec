"""taskB v2 wrapper (mixed_curv_recontinue 架构): HG_Rec_with_BoundedWeightedMixedAdapter (旧, 单头 conditioner).

薄转发: exec 归档在 taskB/stage3/_archive/ 的原始训练脚本, 导出 wrapper 类.
注意: 该脚本当前已演化为 23-param (kappa_logits+mixing_logits) 架构, 与 issue193 9-param ckpt 不匹配 —
      issue193 ckpt 已无法加载 (见 issue30 重测), 此模块仅供架构对照/历史追溯.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_ARCHIVE = _ROOT / "taskB/stage3/_archive/taskB_stage3_mixed_curv_recontinue.py"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "HG-Rec/model"))
sys.path.insert(0, str(_ROOT / "HG-Rec/data"))

_spec = importlib.util.spec_from_file_location("wrap_taskB_v2_mixed", str(_ARCHIVE))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HG_Rec_with_BoundedWeightedMixedAdapter = _mod.HG_Rec_with_BoundedWeightedMixedAdapter
BoundedWeightedMixedCurvatureConditioner = _mod.BoundedWeightedMixedCurvatureConditioner
get_t5_config = _mod.get_t5_config


def load_wrapper_cls():
    return HG_Rec_with_BoundedWeightedMixedAdapter, get_t5_config
