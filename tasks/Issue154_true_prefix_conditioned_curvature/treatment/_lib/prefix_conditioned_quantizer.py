"""Issue #147 — Prefix-conditioned Curvature Routing (QINCo 风格).

A (control): 三层共享 c_l = C_MIN + (C_MAX-C_MIN) * sigmoid(theta_l)
B (treatment):
  L0 全局共享 c_0 = C_MIN + (C_MAX-C_MIN) * sigmoid(theta_0)
  L1 per-item c_1,i = C_MIN + (C_MAX-C_MIN) * sigmoid(theta_1 + delta_1,i)
  L2 per-item c_2,i = C_MIN + (C_MAX-C_MIN) * sigmoid(theta_2 + delta_2,i)

  delta_1,i = delta_max * tanh(g_1(stop_grad(e_0,i)))
  delta_2,i = delta_max * tanh(g_2(stop_grad([e_0,i, e_1,i])))
  g_1/g_2 = 零初始化小 MLP → 训练开始 delta=0 → 与 A 严格等价.

关键约束 (spec):
- 同一 item 的同一层所有候选 codeword 必须用同一个 c_l,i (per-item, 非 per-candidate)
- prefix codeword 表示 stop-gradient (防止 router loss 改写 codebook)
- L_route = lambda_delta * mean(delta^2) + lambda_mean * (mean(log c) - log c)^2
- 禁止 occupancy / 均衡损失 / 强制曲率分化项
- Stage3/4 与 A 一致, 只消费 SID
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import proj_to_ball, expmap0, logmap0, poincare_distance, kmeans, MLP, _eps

C_MIN = 0.5
C_MAX = 2.0
DELTA_MAX = 1.5  # spec 上界 tanh 范围
G_HIDDEN = 32  # MLP 隐藏层宽度
# Issue #154 fix: 正则分工. 根因诊断: (a) 量化 loss 对 c 的梯度极弱 (8e-11),
# 原 0.01/0.01 使 LAMBDA_DELTA 正则梯度 (1.7e-8) 主导 → delta 压回 0 → ROUTER_COLLAPSE;
# (b) 双正则全降 1e-6 → delta 整体漂移饱和到 -1.5 边界 (L1 退化常数).
# 修复: LAMBDA_DELTA=1e-6 (仅防极端漂移, 允许 per-item 分化), LAMBDA_MEAN=0.01
# (锚定 per-item log c 均值 ≈ 全局, 防整体漂移到 C_MIN/C_MAX 边界).
LAMBDA_DELTA = 1e-6  # 防极端漂移 (不主导学习, 允许 delta 分化)
LAMBDA_MEAN = 0.01  # 锚定 per-item log c 均值到全局, 防曲率塌缩边界
GATE1_DIST_TOL = 1e-5  # 距离容差
GATE1_ARG_TOL = 1e-7  # argmin 容差


# ── 几何工具 (tensor-safe, 与 Issue #145 复用 _proj_to_ball_t / _expmap0_t) ──
def _proj_to_ball_t(x, c, eps=1e-6):
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def _expmap0_t(u, c):
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    return _proj_to_ball_t(factor * u, c)


def _logmap0_t(x, c):
    sqrt_c = c ** 0.5
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    factor = torch.atanh((sqrt_c * norm_x).clamp(max=1 - 1e-5)) / (sqrt_c * norm_x)
    return factor * x


def _artanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def _sigmoid(x):
    return torch.sigmoid(x)


# ── Prefix Router: g_l(stop_grad(prefix_emb)) ──
class PrefixRouter(nn.Module):
    """g_l(prefix_emb) → delta_l ∈ (-DELTA_MAX, +DELTA_MAX).
    Issue #154 修复 (常数退化根因): fc1 Xavier 非零初始化 + bias=0; fc2 零初始化.
    旧版 (Issue #147) fc1 全零 → relu(0)=0 → fc2.weight 梯度恒 0 → 路由器退化为
    常数偏置 (所有商品 delta 相同), 而非 per-prefix 动态曲率.
    #154: fc1 非零 → h≠0 → fc2.weight 梯度通路形成 → 真正逐 prefix 分化.
    同时保持 fc2.weight/bias=0 → 初始 delta=0, 与 Control 起点严格等价.
    """

    def __init__(self, in_dim, hidden=G_HIDDEN, delta_max=DELTA_MAX):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # Issue #154 要求 1: fc1.weight Xavier 非零初始化, fc1.bias=0
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Issue #154 要求 2: fc2.weight=0, fc2.bias=0 → 初始 delta=0 (与 A 严格等价)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.delta_max = float(delta_max)

    def forward(self, prefix_emb):
        # 修复 (Issue #154): codeword 切空间范数仅 ~0.01-0.04, fc1 Xavier 适配 O(1) 输入
        # → h = relu(fc1(prefix)) 均值仅 0.003 (fc2 梯度 ∝ h, 比 theta 梯度小 300 倍),
        #   路由器因此学不到 per-prefix 分化 (ROUTER_COLLAPSE 根因).
        # LayerNorm 规范化输入尺度 → fc1 梯度通路有效.
        prefix_emb = F.layer_norm(prefix_emb, prefix_emb.shape[-1:])
        h = F.relu(self.fc1(prefix_emb))
        delta = self.delta_max * torch.tanh(self.fc2(h).squeeze(-1))
        return delta  # (B,)


# ── 单层量化器: prefix-conditioned curvature (per-item, NOT per-candidate) ──
class PrefixConditionedVQ(nn.Module):
    """B 路径单层量化器.
    对每 item, 同一层全部候选 codeword 共享同一个 c_l,i (per-item).
    A 路径 (prefix_routing=False) 直接退化为全局共享 c_l (与基线 KappaAwareVectorQuantization 一致).
    """

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 sk_eps=0.0, sk_iters=3, layer_idx=0, prefix_routing=False,
                 in_dim_router=None):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.layer_idx = layer_idx
        self.prefix_routing = prefix_routing
        # 全局曲率参数 (B 路径也是入口, 与 A 共享 theta_l)
        # 初始化使 sigmoid(theta) → 0.5 (c_l = (C_MIN+C_MAX)/2 = 1.25, 与 baseline init 一致)
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # prefix router (仅 L1/L2 启用, L0 仍全局)
        if prefix_routing:
            if in_dim_router is None:
                raise ValueError("prefix_routing=True requires in_dim_router")
            self.router = PrefixRouter(in_dim_router)
            self._last_delta = None  # 监控
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    def get_c_global(self) -> torch.Tensor:
        return C_MIN + (C_MAX - C_MIN) * _sigmoid(self.theta)

    def get_c_per_item(self, prefix_emb=None) -> torch.Tensor:
        """返回 c_l: (1,) for A / prefix_routing=False, (B,) for B (per-item).
        同 item 同一层所有候选共用同一个 c_l,i (严格遵守 spec: 同一 item c_l,i 不变).
        """
        c_global = self.get_c_global()
        if not self.prefix_routing:
            return c_global
        if prefix_emb is None:
            raise ValueError("B 路径必须传 prefix_emb")
        # prefix_emb 必须 detach (spec: 防止 router loss 反向改写 codebook)
        delta = self.router(prefix_emb.detach())  # (B,)
        # Issue #154 要求 4: 正则必须用未 detach 的 delta (另存 detached 副本仅用于日志)
        self._last_delta_live = delta
        self._last_delta = delta.detach()
        theta_eff = self.theta + delta  # 广播到 (B,)
        c_per = C_MIN + (C_MAX - C_MIN) * _sigmoid(theta_eff)
        return c_per  # (B,)

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, prefix_emb=None, use_sk=True):
        latent = x.view(-1, self.e_dim)
        B = latent.shape[0]
        if not self.initted and self.training:
            self.init_emb(latent)
        # 曲率: per-item 形状 (B,) 广播到 (B,1,1) 用于 (B,K,D) 距离
        c = self.get_c_per_item(prefix_emb)  # (1,) or (B,)
        c_exp = c.view(-1, 1, 1)  # (B,1,1)
        # 投影 latent / codeword 到 c 球
        latent_exp = latent.unsqueeze(1)  # (B,1,D)
        # codeword 投影: 用 latent i 对应的 c_l,i (每个 i 自己的 c 投影每个 codeword 一次)
        # 严格来说, 全部 K 个 codeword 对所有 B 个 item 都用各自的 c 投影 (per-item)
        # 但 codeword 是共享的, 每个 item i 看到的 codeword j 应当按 item i 的 c 投影
        # → (B,K,D) 形状, 每 (i,j) 位置用 c_l,i 投影 codeword[j]
        # 实现: 把 codebook 扩展到 (B,K,D), 每行用 c_l,i 投影
        cb_exp0 = self.embeddings.weight.unsqueeze(0).expand(B, self.n_e, -1)  # (B,K,D)
        latent_h = _proj_to_ball_t(_expmap0_t(latent_exp, c_exp), c_exp)  # (B,1,D)
        codebook_h = _proj_to_ball_t(_expmap0_t(cb_exp0, c_exp), c_exp)  # (B,K,D)
        x_exp = latent_h.expand(B, self.n_e, -1)
        cb_exp = codebook_h
        # 距离: poincare_distance 支持 (B,K,D) × (B,K,D) + c (B,1,1)
        d = poincare_distance(x_exp, cb_exp, c_exp).squeeze(-1)  # (B,K)
        # argmin / sinkhorn
        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = (d - d.mean(dim=-1, keepdim=True)) / (d.std(dim=-1, keepdim=True) + 1e-8)
            d_centered = d_centered.double()
            from utils import sinkhorn_algorithm
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn NaN/Inf")
            indices = torch.argmax(Q, dim=-1)
        # 量化 loss: commitment + codebook (与 baseline 一致)
        x_q = self.embeddings.weight.index_select(0, indices)  # (B,D)
        # per-item c 取回 (B,1)
        c_sel = c.view(-1, 1)
        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, c_sel) ** 2)
        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), c_sel) ** 2)
        loss = commitment_loss + self.beta * codebook_loss
        # 切空间回投影 (对齐 logmap0)
        x_q_safe = _proj_to_ball_t(x_q, c_sel)
        latent_safe = _proj_to_ball_t(latent, c_sel)
        x_q = _logmap0_t(x_q_safe, c_sel)
        latent = _logmap0_t(latent_safe, c_sel)
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


# ── 完整 HRQVAE (encoder + 三层 RQ + decoder) ──
class PrefixConditionedHRQVAE(nn.Module):
    def __init__(self, in_dim=768, num_emb_list=(64, 128, 256), e_dim=32,
                 layers=(512, 256, 128, 64), beta=0.25, kmeans_init=True,
                 kmeans_iters=10, prefix_routing=False, prefix_hidden=16):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.e_dim = e_dim
        self.layers = layers
        self.prefix_routing = prefix_routing
        encode_layer_dims = [in_dim] + list(layers) + [e_dim]
        self.encoder = MLP(layers=encode_layer_dims, dropout=0.0, use_bn=False)
        decode_layer_dims = encode_layer_dims[::-1]
        self.decoder = MLP(layers=decode_layer_dims, dropout=0.0, use_bn=False)
        # 路由器输入维度: L1 router 看 L0 codeword (e_dim), L2 router 看 [L0, L1] (2*e_dim)
        # L0 不启用 prefix_routing (spec: L0 全局共享)
        self.vq_layers = nn.ModuleList()
        for i, n_e in enumerate(num_emb_list):
            is_prefix = prefix_routing and i > 0  # 仅 L1/L2 启用
            if i == 1:
                in_dim_router = e_dim  # L0 codeword
            elif i == 2:
                in_dim_router = 2 * e_dim  # L0 + L1 codeword
            else:
                in_dim_router = None  # L0 不需要
            self.vq_layers.append(PrefixConditionedVQ(
                n_e, e_dim, beta=beta, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                layer_idx=i, prefix_routing=is_prefix, in_dim_router=in_dim_router,
            ))

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self._rq_forward(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices, z_q, z

    def _rq_forward(self, x, use_sk=True):
        all_losses, all_indices = [], []
        x_q = 0
        residual = x
        prefix_codes = []  # 累积已选 codeword (供 router 输入)
        for q in self.vq_layers:
            # router 输入: 该 item 之前所有已选 codeword 拼起来
            prefix_emb = None
            if q.prefix_routing:
                # 拼接所有已选 codeword 向量 (B, n_prefix*e_dim)
                prefix_emb = torch.cat(prefix_codes, dim=-1) if prefix_codes else None
            x_res, loss, idx = q(residual, prefix_emb=prefix_emb, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
            # 保存本层 codeword 向量 (供下层 router 用)
            cb_emb = q.embeddings.weight.index_select(0, idx)  # (B, e_dim)
            prefix_codes.append(cb_emb)
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices

    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        _, _, indices = self._rq_forward(z, use_sk=use_sk)
        return indices

    def compute_route_reg(self):
        """L_route = lambda_delta * mean(delta^2) + lambda_mean * (mean(log c) - log c)^2.
        仅 B 路径 (prefix_routing=True) 的 L1/L2 路由器有 delta.
        Issue #154 要求 4: 使用未 detach 的 delta (_last_delta_live) 计算训练损失,
        detached 副本 (_last_delta) 仅用于日志, 禁止进入损失.
        """
        delta_loss = torch.zeros((), device=next(self.parameters()).device)
        mean_loss = torch.zeros((), device=next(self.parameters()).device)
        for q in self.vq_layers:
            if q.prefix_routing and q._last_delta_live is not None:
                delta_loss = delta_loss + q._last_delta_live.pow(2).mean()
                c_per = C_MIN + (C_MAX - C_MIN) * _sigmoid(q.theta + q._last_delta_live)
                log_c_per = torch.log(c_per)
                log_c_global = torch.log(q.get_c_global())
                mean_loss = mean_loss + (log_c_per.mean() - log_c_global).pow(2)
        return LAMBDA_DELTA * delta_loss, LAMBDA_MEAN * mean_loss