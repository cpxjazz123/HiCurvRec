"""DECOR PromptFormer 移植版 — 完整 candidate bins + alpha gate.

vs DECOR 原版 (DECOR/models/DECOR/layers.py:PromptFormer):
  我们的简化:
    - 不引入 multimodal (HG-Rec 文本模态)
    - 不引入 ortho_weight / collab_layernorm (HG-Rec 已有 B_geo / HAB / GEO)
    - 只保留 PromptFormer 核心: bos_queries + candidate bins + alpha gate
    - attention pool 用 Euclidean (Path A 实验证明 Poincaré 单点改造无效)

设计:
  输入: e_fused (B, L, D) T5 standard lookup embedding
  1. bos_vec = AttentionPool(fused_embeds, mask)
     e_ctx = mean(fused_embeds)   # 简化版: 不引入 2 层 Transformer
     scores = e_ctx @ bos_queries.T
     probs = softmax(scores)
     bos_vec = probs @ bos_queries
  2. candidate bins: 每个 token 位置从 bin_id = token_id // 256 选 256 候选 SID embedding
  3. attn = softmax(q_ctx(bos_vec) @ k_candidates(candidates).T)  # context-aware
  4. e_soft = (attn * candidates).sum(dim=2)
  5. e_final = α * e_soft + (1-α) * e_fused  (α=0.35 init)

参考:
  - DECOR SIGIR 2026 paper
  - DECOR/models/DECOR/layers.py (yiuaa repo)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5


class DecorPromptFormer(nn.Module):
    """DECOR PromptFormer: candidate bins + alpha gate.

    Args:
        d_model: T5 d_model (128 for HG-Rec).
        vocab_size: SID vocabulary size (1024 for HG-Rec: 4 bins × 256 codes).
        num_bins: number of bins (4 for HG-Rec).
        codes_per_bin: codes per bin (256 for HG-Rec).
        num_bos_queries: learnable bos_queries count (DECOR default 64).
        alpha_init: initial alpha (0.35 from DECOR paper).
        e_fused_embedding: nn.Embedding(1024, 128) — T5 shared, used as source of candidate embeddings.
    """

    def __init__(self, d_model=128, vocab_size=1024, num_bins=4, codes_per_bin=256,
                 num_bos_queries=64, alpha_init=0.35,
                 alpha_warmup_steps=200, alpha_penalty_weight=0.01,
                 bos_diversity_weight=0.01, attn_entropy_weight=0.001):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_bins = int(num_bins)
        self.codes_per_bin = int(codes_per_bin)
        self.num_bos_queries = int(num_bos_queries)
        # Issue #68 抗 self-reinforcing trap 配置
        self.alpha_warmup_steps = int(alpha_warmup_steps)
        self.alpha_penalty_weight = float(alpha_penalty_weight)
        self.bos_diversity_weight = float(bos_diversity_weight)
        self.attn_entropy_weight = float(attn_entropy_weight)
        self._step_counter = 0  # warmup 步数跟踪

        # bos_queries: 64 learnable vectors (DECOR default)
        # Issue #68: 加大 init std 防同质化 (原 xavier_uniform_ → normal(0, 0.5))
        bos_queries = torch.zeros(num_bos_queries, d_model)
        nn.init.normal_(bos_queries, mean=0.0, std=0.5)
        self.bos_queries = nn.Parameter(bos_queries, requires_grad=True)

        # alpha gate: sigmoid(alpha_raw) = alpha (init=0.35)
        # logit(0.35) ≈ -0.619
        alpha_logit = torch.log(torch.tensor(alpha_init / (1.0 - alpha_init + _EPS)))
        self.alpha_raw = nn.Parameter(alpha_logit, requires_grad=True)

        # candidate bin projection: q_ctx (bos_vec → query), k_candidates (candidate → key)
        # DECOR 用 Linear(latent_size, latent_size, bias=False)
        self.q_ctx = nn.Linear(d_model, d_model, bias=False)
        self.k_candidates = nn.Linear(d_model, d_model, bias=False)

        # candidate bins embedding (4 bins × 256 codes × 128 dim)
        # registered as buffer (frozen) — 实际值在 forward 时从 e_fused_embedding 取
        # 这里只注册 ID 表
        bin_offsets = torch.arange(num_bins) * codes_per_bin  # [0, 256, 512, 768]
        self.register_buffer("bin_offsets", bin_offsets)

    def _attention_pool(self, fused_embeds, mask):
        """DECOR 简化版 attention pooling → bos_vec (B, D).

        用欧氏点积 + softmax over num_bos_queries.
        不用 2 层 Transformer fusion_layers (DECOR 复杂但对 HG-Rec 可能过参数化).
        """
        B, L, D = fused_embeds.shape
        # mask 屏蔽 PAD
        mask_f = mask.float().unsqueeze(-1) if mask is not None else torch.ones(B, L, 1, device=fused_embeds.device)
        # mean pool fused_embeds → e_ctx
        e_ctx = (fused_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (B, D)

        # bos_queries attention
        scores = e_ctx @ self.bos_queries.T                              # (B, num_bos_queries)
        probs = F.softmax(scores, dim=-1)                                 # (B, num_bos_queries)
        bos_vec = probs @ self.bos_queries                                # (B, D)
        return bos_vec

    def _candidate_attention(self, bos_vec, candidate_embeds):
        """Context-aware soft attention over candidate bins.

        Args:
            bos_vec: (B, D) — attention pool 出来的 context 向量
            candidate_embeds: (B, L, codes_per_bin, D) — 每个位置的 256 候选 SID embedding

        Returns:
            e_soft: (B, L, D) — context-aware 加权的 decoder embedding
            attn: (B, L, K) — attention 权重 (for entropy reg)
        """
        B, L, K, D = candidate_embeds.shape

        # q_ctx(bos_vec) → (B, D) → broadcast (B, L, D)
        q = self.q_ctx(bos_vec).unsqueeze(1).expand(-1, L, -1)             # (B, L, D)
        # k_candidates(candidates) → (B, L, K, D)
        k = self.k_candidates(candidate_embeds)                            # (B, L, K, D)

        # attention scores: (B, L, K)
        attn_scores = (q.unsqueeze(2) * k).sum(dim=-1)                     # (B, L, K)
        attn = F.softmax(attn_scores, dim=-1)                              # (B, L, K)

        # weighted sum: (B, L, D)
        e_soft = (attn.unsqueeze(-1) * candidate_embeds).sum(dim=2)
        return e_soft, attn

    def forward(self, e_fused, input_ids, attention_mask=None,
                e_fused_embedding=None):
        """Compute e_final = alpha * e_soft + (1-alpha) * e_fused.

        Issue #68 抗 self-reinforcing trap:
          - alpha_warmup: 前 K 步冻结 alpha 不动 (T5 学稳定 baseline)
          - bos_diversity_loss: 鼓励 bos_queries 间有差异 (防同质化)
          - attn_entropy_loss: 鼓励 attn 分布尖锐 (防均匀)
        Returns:
            e_final, aux, reg_losses
        """
        B, L, D = e_fused.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=e_fused.device)

        # 1. attention pool fused_embeds → bos_vec (B, D)
        bos_vec = self._attention_pool(e_fused, attention_mask)            # (B, D)

        # 2. candidate bins: 每个位置选 256 候选
        bin_ids = torch.clamp(input_ids // self.codes_per_bin, max=self.num_bins - 1)
        bin_code_ids = self.bin_offsets[bin_ids].unsqueeze(-1) + torch.arange(
            self.codes_per_bin, device=e_fused.device
        )                                                                  # (B, L, 256)

        # 3. lookup candidates from e_fused_embedding
        if e_fused_embedding is None:
            raise ValueError("e_fused_embedding must be provided (T5 shared nn.Embedding)")
        candidate_embeds = e_fused_embedding(bin_code_ids)                 # (B, L, 256, D)

        # 4. context-aware soft attention
        e_soft, attn = self._candidate_attention(bos_vec, candidate_embeds)  # (B, L, D), (B, L, K)

        # 5. alpha gate: e_final = alpha * e_soft + (1-alpha) * e_fused
        alpha = torch.sigmoid(self.alpha_raw)                              # (1,) ∈ (0, 1)
        e_final = alpha * e_soft + (1.0 - alpha) * e_fused

        # === Issue #68 抗 self-reinforcing trap reg losses ===
        reg_losses = {}

        # (a) alpha warmup penalty: 前 alpha_warmup_steps 步, 强制 alpha 接近 0 (只用 T5 embedding)
        # 惩罚 = max(0, alpha - 0.1)  (alpha>0.1 时有梯度, 但线性增长不大)
        if self._step_counter < self.alpha_warmup_steps:
            # warmup 期间, alpha 应该接近 0 (不引入 PromptFormer)
            # penalty = relu(alpha - 0.1) 鼓励 alpha 小
            alpha_penalty = F.relu(alpha - 0.1) ** 2
            reg_losses["alpha_penalty"] = self.alpha_penalty_weight * alpha_penalty
        else:
            reg_losses["alpha_penalty"] = torch.tensor(0.0, device=e_fused.device)

        # (b) bos diversity loss: 鼓励 bos_queries 各维度有差异 (防同质化)
        # loss = -std(bos_queries, dim=0).mean()  鼓励 std 大 (维度间差异)
        bos_std = self.bos_queries.std(dim=0).mean()  # 平均 std per-dim
        bos_diversity_loss = -bos_std  # 负 std = 鼓励 std 增大
        reg_losses["bos_diversity"] = self.bos_diversity_weight * bos_diversity_loss

        # (c) attn entropy loss: 鼓励 attn 分布尖锐 (防均匀)
        # entropy = -sum(p * log p), 鼓励 entropy 小 (sharp) → loss = entropy.mean()
        attn_log = torch.log(attn + 1e-9)  # (B, L, K)
        attn_entropy = -(attn * attn_log).sum(dim=-1).mean()  # (scalar)
        reg_losses["attn_entropy"] = self.attn_entropy_weight * attn_entropy

        # 步数计数器 +1
        self._step_counter += 1

        aux = {
            "alpha": alpha.detach().item(),
            "bos_vec_norm": bos_vec.norm(dim=-1).mean().item(),
            "e_soft_norm": e_soft.norm(dim=-1).mean().item(),
            "bos_std_per_dim": bos_std.item(),
            "attn_entropy": attn_entropy.item(),
        }
        return e_final, aux, reg_losses