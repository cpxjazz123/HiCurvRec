from collections import defaultdict
from einops import rearrange
import torch
from torch import Tensor


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        match_found, rank = pos_match.all(axis=-1).max(axis=-1)
        matched_rank = rank[match_found]
        ndcg = 1.0 / torch.log2(matched_rank.float() + 2.0)
        # 总体 NDCG (实际等于 NDCG@top_k_size, 因为 rank 上限是 top_k 的最大 beam 数)
        self.metrics["ndcg"] += ndcg.sum().item()
        # 显式 NDCG@K (HG-Rec 风格, 用于 valid NDCG@20 选 best ckpt)
        for k in self.ks:
            # NDCG@K: 只看前 K 个 beam 内的命中, log2(rank+2) 的 rank 限制在 < K
            in_top_k_mask = matched_rank < k
            ndcg_at_k_sum = ndcg[in_top_k_mask].sum().item()
            self.metrics[f"ndcg@{k}"] += ndcg_at_k_sum
            self.metrics[f"h@{k}"] += len(matched_rank[in_top_k_mask])
        self.total += B

    def reduce(self) -> dict:
        # 单卡/无 DDP: 直接除以 self.total
        return {k: v / self.total for k, v in self.metrics.items()}

    def get_sums_and_total(self) -> dict:
        """DDP: 返回每个 rank 本地的累加 sum + total, 不做除法.
        调用方拿到 4 个 rank 的 sums 后, 自己 all_reduce SUM 再除以全局 total.
        """
        out = dict(self.metrics)
        out["total"] = self.total
        return out
