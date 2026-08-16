import gin
import math
import torch

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.hyperbolic import C_MAX as _C_MAX
from modules.hyperbolic import C_MIN as _C_MIN
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple, Optional
from torch import nn
from torch import Tensor
from torch.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor
    margin: Tensor  # C5: per-item top2-top1 距离 margin (d2 - d1), 用于 margin 正则


class PrefixRouter(nn.Module):
    """Issue #154 迁移: prefix-conditioned per-item 曲率 router.

    g_l(prefix_emb) → delta_l ∈ (-DELTA_MAX, +DELTA_MAX),
    c_l,i = C_MIN + (C_MAX-C_MIN)*sigmoid(theta_l + delta_l,i).
    fc1 Xavier 非零初始化 (LayerNorm 输入) + fc2 零初始化 → 初始 delta=0 (与 baseline 严格等价),
    且梯度通路形成 (Issue #154 ROUTER_COLLAPSE 修复).
    """

    G_HIDDEN = 16
    DELTA_MAX = 1.0

    def __init__(self, in_dim, hidden=G_HIDDEN, delta_max=DELTA_MAX):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.delta_max = float(delta_max)

    def forward(self, prefix_emb):
        # Issue #154 修复: LayerNorm 规范化输入尺度 → fc1 梯度通路有效 (codeword 范数仅 ~0.01-0.04)
        prefix_emb = F.layer_norm(prefix_emb, prefix_emb.shape[-1:])
        h = F.relu(self.fc1(prefix_emb))
        delta = self.delta_max * torch.tanh(self.fc2(h).squeeze(-1))
        return delta  # (B,)


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, "b d -> b 1 d")
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    ).squeeze()


class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,  # https://arxiv.org/pdf/2411.02038
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2,
        hyperbolic_distance: bool = False,  # HG-Rec 机制: Poincaré 距离 argmin (缓解 collapse)
        sk_eps: float = 0.0,                # HG-Rec 机制: Sinkhorn 均衡温度 (0=关闭)
        sk_iters: int = 3,                  # Sinkhorn 迭代数
        prefix_routing: bool = False,       # Issue #154: prefix-conditioned per-item 曲率 (L1/L2)
        in_dim_router: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.hyperbolic_distance = hyperbolic_distance
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.prefix_routing = prefix_routing
        self._last_delta = None

        # M2/M3 (Issue #166 treatment 移植): 每层可学习曲率 θ_l → c_l = C_MIN + (C_MAX-C_MIN)*sigmoid(θ)
        # 初始化 c=1.0 (HG-Rec c=1.0 对齐): θ = log((1.0-C_MIN)/(C_MAX-1.0)) = log(0.5) ≈ -0.6931
        self.theta = nn.Parameter(torch.tensor(math.log((1.0 - _C_MIN) / (_C_MAX - 1.0)), dtype=torch.float32))

        # Issue #154: prefix router (L1/L2) — per-item delta, fc2 零初始化 → 初始 delta=0
        if prefix_routing:
            if in_dim_router is None:
                raise ValueError("prefix_routing=True requires in_dim_router")
            self.router = PrefixRouter(in_dim_router)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity(),
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    def get_c(self) -> Tensor:
        """M2/M3: 该层全局曲率 c_l (1,) — C_MIN + (C_MAX-C_MIN)*sigmoid(theta_l)."""
        return _C_MIN + (_C_MAX - _C_MIN) * torch.sigmoid(self.theta)

    def get_c_per_item(self, prefix_emb: Optional[Tensor] = None) -> Tensor:
        """Issue #154: per-item 曲率 c_l,i (B,) — prefix router 输出 delta 调制 θ.

        无 prefix_routing / 无 prefix_emb → 退化为全局 c_l (1,).
        fc2 零初始化 → 初始 delta=0 → 与 baseline 严格等价.
        """
        c_global = self.get_c()
        if not self.prefix_routing or prefix_emb is None:
            return c_global
        # prefix_emb 必须 detach (spec: 防止 router loss 反向改写 codebook 路径)
        delta = self.router(prefix_emb.detach())  # (B,)
        self._last_delta = delta.detach()
        theta_eff = self.theta + delta  # (B,) broadcast
        return _C_MIN + (_C_MAX - _C_MIN) * torch.sigmoid(theta_eff)  # (B,)

    @staticmethod
    def _center_distance_for_constraint(distances: Tensor) -> Tensor:
        """HG-Rec: 距离归一化到 [-1, 1] 附近 (sinkhorn 前处理)."""
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-10
        assert amplitude > 0, "Amplitude must be positive"
        return (distances - middle) / amplitude

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)

    @torch.no_grad
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature, prefix_emb=None) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)

        if self.distance_mode == QuantizeDistance.L2:
            if self.hyperbolic_distance:
                # HG-Rec 机制复制: Poincaré 距离 argmin (双曲量化, 缓解 codebook collapse)
                # latent/codebook expmap 到该层曲率空间后算双曲距离 (per-item c 广播)
                from modules.hyperbolic import _expmap0_t, _poincare_distance_t
                c = self.get_c_per_item(prefix_emb)  # (1,) 或 (B,)
                c_exp = c.view(-1, 1, 1)             # (1,1,1) 或 (B,1,1)
                B = x.shape[0]
                K = codebook.shape[0]
                latent_h = _expmap0_t(x.unsqueeze(1), c_exp)                    # (B,1,D)
                cb_exp0 = codebook.unsqueeze(0).expand(B, K, -1)               # (B,K,D)
                codebook_h = _expmap0_t(cb_exp0, c_exp)                        # (B,K,D)
                dist = _poincare_distance_t(
                    latent_h.expand(B, K, -1), codebook_h, c_exp
                ).squeeze(-1)  # (B, K)
            else:
                dist = (
                    (x**2).sum(axis=1, keepdim=True)
                    + (codebook.T**2).sum(axis=0, keepdim=True)
                    - 2 * x @ codebook.T
                )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                x
                / x.norm(dim=1, keepdim=True)
                @ (codebook.T)
                / codebook.T.norm(dim=0, keepdim=True)
            )
        else:
            raise Exception("Unsupported Quantize distance mode.")

        _, ids = (dist.detach()).min(axis=1)

        # C5: per-item top2-top1 距离 margin (d2 - d1)
        # 注意: 不 detach — 让 margin 正则梯度经 dist 流向 codebook/encoder (拉大 top1/top2 距离)
        dist_sorted, _ = dist.sort(dim=-1)
        margin = dist_sorted[:, 1] - dist_sorted[:, 0]  # (B,)

        if self.sk_eps > 0 and self.hyperbolic_distance:
            # HG-Rec 防坍缩: Sinkhorn-Knopp 均衡分配 (balanced assignment, 强制 code 均衡使用)
            from modules.hyperbolic import _sinkhorn_algorithm
            d_centered = self._center_distance_for_constraint(dist.detach())
            Q = _sinkhorn_algorithm(d_centered.double(), self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn algorithm produced NaN or Inf values.")
            ids = torch.argmax(Q, dim=-1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = weights @ codebook
                emb_out = emb
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8),
                    x,
                )
                emb_out = (
                    emb_out
                    * (
                        torch.norm(emb, dim=1, keepdim=True)
                        / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                    ).detach()
                )
            else:
                raise Exception("Unsupported Quantize forward mode.")

            loss = self.quantize_loss(query=x, value=emb)

        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(embeddings=emb_out, ids=ids, loss=loss, margin=margin)
