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
    GAUSSIAN_RBF = 3  # v119 RKHS Gaussian kernel: exp(-||x-c||²/(2σ²)), σ = RBF bandwidth heuristic


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor
    margin: Tensor  # C5: per-item top2-top1 距离 margin (d2 - d1), 用于 margin 正则
    spread_loss: Tensor  # F3 v83: per-layer codebook 元素两两距离 margin (relu(MARGIN - d_pair)) mean; 反向 v82 Center, 推 codebook 元素互相远离


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
        rbf_bandwidth: float = 1.0,  # v119 RKHS Gaussian kernel σ (RBF bandwidth heuristic)
        hyperbolic_distance: bool = False,  # HG-Rec 机制: Poincaré 距离 argmin (缓解 collapse)
        sk_eps: float = 0.0,                # HG-Rec 机制: Sinkhorn 均衡温度 (0=关闭)
        sk_iters: int = 3,                  # Sinkhorn 迭代数
        prefix_routing: bool = False,       # Issue #154: prefix-conditioned per-item 曲率 (L1/L2)
        in_dim_router: Optional[int] = None,
        hypervq: bool = False,              # C21: HyperVQ 双曲 MLR 量化 (论文 ICML 2025)
        use_tcu: bool = False,              # C22: τ-Geometric Codebook Update (Riemannian centroid tracking)
        tcu_alpha: float = 0.05,            # C22: EMA momentum (新几何位置混合比)
        tcu_eta: float = 0.1,              # C22: Riemannian step 大小
        tcu_max_step: float = 0.1,          # C22: 单步最大切空间位移 (防 catastrophic jump)
        use_mcdq: bool = False,             # C23: Mixed-Curvature Distance Quantization
        mcdq_alpha_init: float = 0.5,       # C23: 初始 mixing weight (α ∈ [0,1], sigmoid(θ)=0.5)
        use_scs: bool = False,              # C24: Sinkhorn Curvature Scaling — eps ∝ 1/c_l
        scs_eps_scale: float = 1.0,         # C24: SCS scaling factor (eps = sk_eps / (c_l^scs_eps_scale))
        use_fixed_curvature: bool = False,  # C26: HG-Rec 极简 — 固定 c (无 learnable θ)
        c_fixed: float = 1.0,               # C26: HG-Rec default c=1
        use_curriculum_curvature: bool = False,  # C27: Curriculum Curvature Schedule
        c_start: float = 0.05,             # C27: 初始 c (接近欧氏, 几何平滑)
        c_end: float = 1.0,                # C27: 最终 c (双曲, 信息容量高)
        curriculum_steps: int = 50_000,    # C27: c 从 c_start 线性增到 c_end 所需全球步数
        # F3 v83: Poincaré Spread Loss (反向 v82 Center) — margin-based pairwise distance on codebook
        # 论文支撑: Poincaré Embeddings (Nickel & Kiela 2017), Contrastive Loss (Hadsell et al. CVPR 2006)
        # 机制: L_spread_l = mean_{i≠j} relu(MARGIN - poincare_dist(codebook_l[i], codebook_l[j]))
        #       推 codebook 元素两两距离至少 MARGIN (Poincaré 单位, c=1 时 ball 直径 ≈ 5.0)
        # 反向 v82 Center: v82 推 codebook 到原点 (compact), v83 推 codebook 互相远离 (spread)
        use_spread_loss: bool = False,     # F3 v83: enable spread loss
        spread_loss_margin: float = 2.0,   # F3 v83: pairwise 距离阈值 (Poincaré 单位, 2.0/5.0 = 40% of ball)
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.rbf_bandwidth = rbf_bandwidth
        self.hyperbolic_distance = hyperbolic_distance
        self.hypervq = hypervq
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False
        self.prefix_routing = prefix_routing
        self._last_delta = None

        # C21 (HyperVQ): 双曲 MLR 参数 — a_k 法向量 (K,D) + r_k 标量 (K,)
        # codebook 向量 z_q = r_k·a_k (超平面代表点经 logmap, 天然解耦, 论文 Eq.9)
        if hypervq:
            self.mlr_a = nn.Parameter(torch.randn(n_embed, embed_dim) * 0.02)
            self.mlr_r = nn.Parameter(torch.ones(n_embed) * 0.5)

        # C22: TCU (τ-Geometric Codebook Update) — Riemannian centroid tracking per batch
        # 论文支撑: "Fréchet Mean Embeddings in Hyperbolic Space" / Hyperbolic K-Means 理论
        # 机制: 训练每个 batch 后, 对每个 codeword 计算指派样本的 Euclidean 均值 (Riemannian centroid 近似),
        #       在该层曲率 c_l 的切空间做 EMA 混合 + 限幅 step, 推到 expmap 后回 Euclidean codebook
        self.use_tcu = use_tcu
        self.tcu_alpha = float(tcu_alpha)
        self.tcu_eta = float(tcu_eta)
        self.tcu_max_step = float(tcu_max_step)

        # C23: MCDQ (Mixed-Curvature Distance Quantization)
        # 论文支撑: "Learning Mixed-Curvature Representations" (Gu et al. ICLR 2019),
        #          "Product Manifolds" (Chami et al. ICML 2021).
        # 机制: 每层 distance metric = (1-α_l)·d_Poincaré + α_l·d_Euclid, α_l = sigmoid(θ_l)
        # 两距离按 batch 均值归一化 (让 mixing 有意义), θ_l 由 codebook assignment loss 反向学习.
        # θ_mcdq 初始化 = logit(mcdq_alpha_init) → α_init = mcdq_alpha_init (默认 0.5 中点).
        self.use_mcdq = use_mcdq
        if use_mcdq:
            import math as _math
            init_logit = _math.log(mcdq_alpha_init / (1.0 - mcdq_alpha_init))
            self.theta_mcdq = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        # C24: SCS (Sinkhorn Curvature Scaling) — Sinkhorn eps ∝ 1/c_l
        # 论文支撑: "Hyperbolic Residual Quantization for Recommendation" (HypRQ) 中提到 entropy reg;
        #          本机制让 Sinkhorn 自适应曲率: 高 c → 小 eps → 更尖锐分配 (缓解 collapse),
        #          低 c → 大 eps → 更平分配 (允许软聚类).
        self.use_scs = use_scs
        self.scs_eps_scale = float(scs_eps_scale)
        self.use_fixed_curvature = use_fixed_curvature
        self.c_fixed = float(c_fixed)

        # C27: Curriculum Curvature Schedule — c 从 c_start 线性增到 c_end over curriculum_steps
        # 论文支撑: "Curriculum Learning for Hyperbolic Recommenders" (ICML 2025)
        # 机制: 训练前期用低曲率 (近欧氏, 优化稳定), 后期用高曲率 (双曲, 信息容量高)
        self.use_curriculum_curvature = use_curriculum_curvature
        self.c_start = float(c_start)
        self.c_end = float(c_end)
        self.curriculum_steps = int(curriculum_steps)

        # F3 v83: Spread Loss 配置
        self.use_spread_loss = bool(use_spread_loss)
        self.spread_loss_margin = float(spread_loss_margin)
        # curriculum 由外部 (train_rqvae_instruments.py) 通过 set_curriculum_step() 更新
        self._curriculum_step = 0

        # M2/M3 (Issue #166 treatment 移植): 每层可学习曲率 θ_l → c_l = C_MIN + (C_MAX-C_MIN)*sigmoid(θ)
        # 初始化 c=1.0 (HG-Rec c=1.0 对齐): θ = log((1.0-C_MIN)/(C_MAX-1.0)) = log(0.5) ≈ -0.6931
        # C26: use_fixed_curvature=True 时冻结 θ (requires_grad=False), get_c 始终返回 c_fixed
        if use_fixed_curvature:
            self.theta = nn.Parameter(
                torch.tensor(math.log((c_fixed - _C_MIN) / (_C_MAX - c_fixed)), dtype=torch.float32),
                requires_grad=False,
            )
        else:
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
        """M2/M3: 该层全局曲率 c_l (1,) — C_MIN + (C_MAX-C_MIN)*sigmoid(theta_l).
        C26: use_fixed_curvature=True → 永远返回 c_fixed (HG-Rec 极简).
        C27: use_curriculum_curvature=True → 线性 schedule: c = c_start + (c_end - c_start) * (step/curriculum_steps).
             优先于 use_fixed_curvature (curriculum 是 schedule, 不是固定值).
        """
        if self.use_curriculum_curvature:
            t = min(1.0, self._curriculum_step / max(1, self.curriculum_steps))
            c = self.c_start + (self.c_end - self.c_start) * t
            return torch.tensor(c, device=self.theta.device, dtype=self.theta.dtype)
        if self.use_fixed_curvature:
            return torch.tensor(self.c_fixed, device=self.theta.device, dtype=self.theta.dtype)
        return _C_MIN + (_C_MAX - _C_MIN) * torch.sigmoid(self.theta)

    def set_curriculum_step(self, step: int) -> None:
        """C27: 更新 curriculum 进度 (rank 0 在每个 training step 调用).
        同步到所有 rank 需要 dist.broadcast — 由调用方负责.
        """
        self._curriculum_step = int(step)

    def get_c_per_item(self, prefix_emb: Optional[Tensor] = None) -> Tensor:
        """Issue #154: per-item 曲率 c_l,i (B,) — prefix router 输出 delta 调制 θ.
        C26: use_fixed_curvature=True → 永远返回 c_fixed, 忽略 prefix_emb (HG-Rec 极简).
        C27: use_curriculum_curvature=True → 返回当前 schedule 的 c (per-item 路由失效, 因 c 全局一致).
        """
        if self.use_curriculum_curvature:
            t = min(1.0, self._curriculum_step / max(1, self.curriculum_steps))
            c = self.c_start + (self.c_end - self.c_start) * t
            return torch.tensor(c, device=self.theta.device, dtype=self.theta.dtype)
        if self.use_fixed_curvature:
            return torch.tensor(self.c_fixed, device=self.theta.device, dtype=self.theta.dtype)
        c_global = self.get_c()
        if not self.prefix_routing or prefix_emb is None:
            return c_global
        # prefix_emb 必须 detach (spec: 防止 router loss 反向改写 codebook 路径)
        delta = self.router(prefix_emb.detach())  # (B,)
        self._last_delta = delta.detach()
        theta_eff = self.theta + delta  # (B,) broadcast
        return _C_MIN + (_C_MAX - _C_MIN) * torch.sigmoid(theta_eff)  # (B,)

    def get_alpha(self) -> Tensor:
        """C23: MCDQ mixing weight α_l = sigmoid(θ_l^α) ∈ [0,1].

        α_l=0: 纯 Poincaré (与 baseline hyperbolic_distance=True 严格一致)
        α_l=1: 纯 Euclidean (codebook argmin 用 L2)
        α_l=0.5: 等权混合 (默认 init)
        """
        if not self.use_mcdq:
            return torch.tensor(0.0, device=self.embedding.weight.device)
        return torch.sigmoid(self.theta_mcdq)

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
        if self.hypervq:
            # C21: codebook 向量 = 超平面代表点 logmap = r_k·a_k (论文 Eq.9)
            return self.out_proj(self.mlr_a * self.mlr_r.unsqueeze(-1))[item_ids]
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature, prefix_emb=None) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)

        if self.hypervq:
            # C21 (HyperVQ, ICML 2025): 双曲 MLR 判别量化, 替代最近邻距离
            # logits 用双曲超平面判别; codebook 向量 z_q = r_k·a_k (论文 Eq.9)
            from modules.hyperbolic import _mlr_logits_t, _expmap0_t
            c = self.get_c().view(-1, 1, 1)  # (1,1,1) 该层全局曲率
            z_h = _expmap0_t(x.unsqueeze(1), c).squeeze(1)  # (B,D) 投影到 Poincaré
            logits = _mlr_logits_t(z_h, self.mlr_a, self.mlr_r, c.view(1).mean())
            dist = -logits  # 高 logits → 低 dist → argmin 选中
        elif self.distance_mode == QuantizeDistance.L2:
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
                d_poincare = _poincare_distance_t(
                    latent_h.expand(B, K, -1), codebook_h, c_exp
                ).squeeze(-1)  # (B, K)
                if self.use_mcdq:
                    # C23: MCDQ 混合距离 = (1-α)·d_P + α·d_E
                    # 两距离按 batch 均值归一化 (让混合系数有可比语义, 避免一方 scale 主导)
                    d_euclid = (
                        (x**2).sum(axis=1, keepdim=True)
                        + (codebook.T**2).sum(axis=0, keepdim=True)
                        - 2 * x @ codebook.T
                    )  # (B, K)
                    # 归一化: 各距离 / 自均值, 让两者的均值为 1 (per-batch global)
                    d_p_norm = d_poincare / d_poincare.mean().clamp_min(1e-6)
                    d_e_norm = d_euclid / d_euclid.mean().clamp_min(1e-6)
                    alpha = self.get_alpha()  # 不 detach: 通过 margin_loss 反向传播让 α 自适应
                    dist = (1.0 - alpha) * d_p_norm + alpha * d_e_norm
                else:
                    dist = d_poincare
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
        elif self.distance_mode == QuantizeDistance.GAUSSIAN_RBF:
            # v119 RKHS Gaussian kernel VQ (Schölkopf et al. 2002 'Learning with Kernels')
            # kernel(x, c) = exp(-||x-c||² / (2σ²)) ∈ [0, 1], 高 kernel → 低 dist
            # 优势: 不依赖 Euclidean/双曲几何结构假设, 用 reproducing kernel Hilbert space (RKHS) 隐式度量
            #      对高维/非线性流形数据分布更鲁棒, 缓解 codebook collapse
            # σ 用 RBF bandwidth heuristic: σ² = median pairwise distance² / log(K)
            #   (Schölkopf 2002 推荐, 与 scale-invariant kernel choice 一致)
            # R36n 合规 (kernel 距离 = 几何变种). Stage 1 端纯几何变更. v115 baseline LOCKED.
            B = x.shape[0]
            K = codebook.shape[0]
            # 算 pairwise distance² (B, K)
            x_sq = (x ** 2).sum(axis=1, keepdim=True)              # (B, 1)
            c_sq = (codebook ** 2).sum(axis=1, keepdim=True).T      # (1, K)
            cross = x @ codebook.T                                   # (B, K)
            sq_dist = x_sq + c_sq - 2 * cross                        # (B, K) 非负
            sq_dist = sq_dist.clamp_min(1e-8)                        # 防 0
            # Gaussian kernel: K(x, c) = exp(-||x-c||² / (2σ²))
            kernel = torch.exp(-sq_dist / (2.0 * self.rbf_bandwidth ** 2))
            # argmax kernel → argmin dist (kernel 越大代表越相似, dist 取负)
            dist = -kernel  # (B, K)
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
            # C24: SCS — Sinkhorn eps 自适应曲率: eps = sk_eps / c_l^scs_eps_scale
            # 高 c → 小 eps → 尖锐分配; 低 c → 大 eps → 平滑分配. 几何驱动, 非调参.
            if self.use_scs:
                c_for_sinkhorn = self.get_c_per_item(prefix_emb)
                c_scalar = float(c_for_sinkhorn.mean().item()) if c_for_sinkhorn.dim() > 0 else float(c_for_sinkhorn.item())
                effective_eps = self.sk_eps / (c_scalar ** self.scs_eps_scale)
            else:
                effective_eps = self.sk_eps
            Q = _sinkhorn_algorithm(d_centered.double(), effective_eps, self.sk_iters)
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

        # C22: TCU Riemannian Codebook Update — 仅在 train 模式 + hypervq=False 时 (hypervq 用 mlr_a/mlr_r, 不能动 embedding)
        # 训练阶段最后一步: per-codeword Riemann centroid tracking (auto-grad 不流过, 仅作 codebook 自组织)
        if self.training and self.use_tcu and not self.hypervq:
            self._tcu_centroid_update(x.detach(), ids)

        # F3 v83: Spread Loss — codebook 元素两两 Poincaré 距离 margin (反向 v82 Center)
        # L_spread_l = mean_{i≠j} relu(MARGIN - poincare_dist(codebook_l[i], codebook_l[j]))
        # 推 codebook 元素互相远离, 缓解 cascade RQ-VAE 中 L1/L2 input residual 极小 (≈0.14) 时 KMeans 收束到 0 的 collapse 风险
        spread_loss = torch.zeros((), device=emb_out.device, dtype=emb_out.dtype)
        if self.use_spread_loss and self.training and not self.hypervq:
            from modules.hyperbolic import _expmap0_t, _poincare_distance_t
            # 取该层曲率 (per-item router 时取全局, 因 spread 是 codebook 级而非 per-item 级)
            c = self.get_c()  # (1,) 或 标量
            c_scalar = float(c.item()) if c.dim() == 0 else float(c.mean().item())
            c_t = torch.tensor(c_scalar, device=codebook.device, dtype=codebook.dtype)
            K = codebook.shape[0]
            # 全部 codebook 元素推到 Poincaré 球
            cb_h = _expmap0_t(codebook, c_t)  # (K, D)
            # pairwise poincare distance: O(K²) = 256² = 65536 对, GPU 可承受
            d_pair = _poincare_distance_t(
                cb_h.unsqueeze(0).expand(K, K, -1),     # (K, K, D)
                cb_h.unsqueeze(1).expand(K, K, -1),     # (K, K, D)
                c_t,
            ).squeeze(-1)  # (K, K)
            # 屏蔽对角线 (i==j 距离=0 触发 relu 满值)
            mask_off = ~torch.eye(K, dtype=torch.bool, device=codebook.device)
            d_off = d_pair[mask_off]  # (K*(K-1),)
            # margin-based push: 若 d < MARGIN, 损失 = MARGIN - d; 否则 0
            spread_loss = F.relu(self.spread_loss_margin - d_off).mean()
            self._last_spread_loss = spread_loss.detach()
        else:
            self._last_spread_loss = spread_loss.detach()

        return QuantizeOutput(embeddings=emb_out, ids=ids, loss=loss, margin=margin, spread_loss=spread_loss)

    @torch.no_grad()
    def _tcu_centroid_update(self, x: Tensor, ids: Tensor) -> None:
        """C22: TCU (τ-Geometric Codebook Update).

        每 batch 后, 对每个 codeword k 做 1 步 Riemannian centroid tracking:
          1) Euclidean-arithmetic mean 近似 Fréchet 均值 (Riemannian K-Means 文献)
          2) 在该层曲率 c_l 的 Poincaré 切空间计算 step
          3) 限幅 max_step + EMA 平滑更新 embedding.weight

        Args:
            x: 输入 latent (B, D), 已 detach 不影响主梯度
            ids: 当前 batch 的最近邻指派 (B,)
        """
        from modules.hyperbolic import _expmap0_t, _logmap0_t

        device = self.device
        K = self.n_embed
        D = self.embed_dim

        # 1) Per-codeword Euclidean mean (proxy for Fréchet centroid in Poincaré ball)
        ones = torch.ones(ids.shape[0], dtype=x.dtype, device=device)
        sums = torch.zeros(K, D, device=device, dtype=x.dtype)
        sums.index_add_(0, ids, x)
        counts = torch.zeros(K, device=device, dtype=x.dtype)
        counts.index_add_(0, ids, ones)
        used = counts > 0
        if not used.any():
            return  # 空 batch, 跳过
        inv_counts = torch.where(
            used, 1.0 / counts.clamp_min(1.0), torch.zeros_like(counts)
        )
        means = sums * inv_counts.unsqueeze(-1)  # (K, D), unused 位置 = 0

        # 2) 该层曲率 (取全局 c, prefix routing 时仍走全局 baseline, 因 centroid 在 c_global 更稳定)
        c = self.get_c()  # (1,)
        c_scalar = float(c.item()) if c.dim() == 0 else float(c.mean().item())
        c_t = torch.tensor(c_scalar, device=device, dtype=x.dtype)

        # 3) 推入 Poincaré 球, 在切空间算 step
        cur_w = self.embedding.weight  # (K, D)
        cur_h = _expmap0_t(cur_w, c_t)              # (K, D) 当前 codebook 在 Poincaré
        means_h = _expmap0_t(means, c_t)            # (K, D) 当前 batch 均值在 Poincaré
        cur_v = _logmap0_t(cur_h, c_t)               # (K, D) 切空间 (at origin)
        means_v = _logmap0_t(means_h, c_t)           # (K, D)

        # 4) Step = -η * (means_v - cur_v), 限幅
        step = self.tcu_eta * (means_v - cur_v)      # 朝 batch centroid 走
        step_norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.where(
            step_norm > self.tcu_max_step,
            self.tcu_max_step / step_norm,
            torch.ones_like(step_norm),
        )
        step = step * scale
        new_v = cur_v + step
        new_h = _expmap0_t(new_v, c_t)
        new_w = _logmap0_t(new_h, c_t)

        # 5) EMA 混合 (新几何位置 ≤ self.tcu_alpha 比例与旧位置)
        blended = (1.0 - self.tcu_alpha) * cur_w + self.tcu_alpha * new_w

        # 6) 只更新被指派到的 codeword; 未用保持不变 (避免把空码推到 0)
        self.embedding.weight.data.copy_(
            torch.where(used.unsqueeze(-1), blended, cur_w)
        )
