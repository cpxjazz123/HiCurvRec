# 本地 evaluate 子包 (避免与 HuggingFace site-packages/evaluate 冲突)
from .metrics import TopKAccumulator

__all__ = ["TopKAccumulator"]