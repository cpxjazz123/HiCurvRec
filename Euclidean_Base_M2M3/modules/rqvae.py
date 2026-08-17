import math
import torch

from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.hyperbolic import _expmap0_t, _logmap0_t, _mobius_add_t, _transport_between_t
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List, Optional
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F

torch.set_float32_matmul_precision("high")


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor
    margins: List[Tensor]  # C5: 各层 per-item margin (d2-d1), list of (B,)


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    per_layer_usage: Tensor  # codebook 健康检查: 每层量化 ids 的 unique code 数 (n_layers,)
    margin_loss: Tensor  # C5: margin 正则项值 (0 关闭时 = 0)
    per_layer_margin: Tensor  # C5: 各层 mean margin (n_layers,) 供日志监控


class RqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
        # M2/M3 (Issue #166 treatment 移植): gate_M2_intrinsic (Möbius 内在减法)
        # + gate_M3_transport (跨层曲率传输)
        gate_M2_intrinsic: bool = True,
        gate_M3_transport: bool = True,
        hyperbolic_distance: bool = True,  # HG-Rec 机制: Poincaré 距离 argmin (缓解 codebook collapse)
        sk_eps: float = 0.05,              # HG-Rec 机制: Sinkhorn 均衡温度 (0=关闭; 0.003 太陡致约束失效, 0.05 实测均衡)
        sk_iters: int = 3,                 # Sinkhorn 迭代数 (HG-Rec 论文值)
        hypervq: bool = False,             # C21: HyperVQ 双曲 MLR 量化 (ICML 2025)
        prefix_router_layers: Optional[List[bool]] = None,  # Issue #154: 各层是否 prefix-routing (默认 L0 关, L1/L2 开)
        # C5: 曲率 margin 正则 (新曲率正则项, R36) — 0=关闭
        margin_reg_weight: float = 0.0,
        margin_target: float = 0.05,
        use_tcu: bool = False,              # C22: τ-Geometric Codebook Update (Riemannian centroid tracking)
        tcu_alpha: float = 0.05,            # C22: EMA momentum
        tcu_eta: float = 0.1,              # C22: Riemannian step 大小
        use_mcdq: bool = False,             # C23: Mixed-Curvature Distance Quantization (per-layer α)
        mcdq_alpha_init: float = 0.5,       # C23: 初始 mixing weight (sigmoid⁻¹(mcdq_alpha_init))
        use_scs: bool = False,              # C24: Sinkhorn Curvature Scaling — eps ∝ 1/c_l
        scs_eps_scale: float = 1.0,         # C24: SCS scaling 指数
        use_fixed_curvature: bool = False,  # C26: HG-Rec 极简 — 固定曲率 (无 learnable θ)
        c_fixed: float = 1.0,               # C26: HG-Rec default c=1
    ) -> None:
        self._config = locals()

        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features
        self.gate_M2_intrinsic = gate_M2_intrinsic
        self.gate_M3_transport = gate_M3_transport
        self.hyperbolic_distance = hyperbolic_distance
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.hypervq = hypervq
        self.margin_reg_weight = margin_reg_weight
        self.margin_target = margin_target
        self.use_tcu = use_tcu
        self.tcu_alpha = float(tcu_alpha)
        self.tcu_eta = float(tcu_eta)
        self.use_mcdq = use_mcdq
        self.mcdq_alpha_init = float(mcdq_alpha_init)
        self.use_scs = use_scs
        self.scs_eps_scale = float(scs_eps_scale)
        self.use_fixed_curvature = use_fixed_curvature
        self.c_fixed = float(c_fixed)
        # Issue #154: 默认 L0 全局曲率, L1/L2 prefix-conditioned per-item 曲率
        if prefix_router_layers is None:
            prefix_router_layers = [False] + [True] * (n_layers - 1)
        assert len(prefix_router_layers) == n_layers
        self.prefix_router_layers = prefix_router_layers

        self.layers = nn.ModuleList(
            modules=[
                Quantize(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    forward_mode=codebook_mode,
                    do_kmeans_init=codebook_kmeans_init,
                    codebook_normalize=i == 0 and codebook_normalize,
                    sim_vq=codebook_sim_vq,
                    commitment_weight=commitment_weight,
                    hyperbolic_distance=hyperbolic_distance,
                    sk_eps=sk_eps,
                    sk_iters=sk_iters,
                    # Issue #154: L1/L2 prefix-conditioned per-item 曲率 (L0 全局)
                    # router 输入 = 已选 codeword (i*D 维) + M3 曲率信号 (1 维)
                    prefix_routing=prefix_router_layers[i],
                    in_dim_router=(i * embed_dim + 1) if prefix_router_layers[i] else None,
                    hypervq=hypervq,  # C21: HyperVQ 双曲 MLR 量化
                    use_tcu=use_tcu,  # C22: Riemannian centroid update
                    tcu_alpha=tcu_alpha,
                    tcu_eta=tcu_eta,
                    use_mcdq=use_mcdq,  # C23: Mixed-Curvature Distance Quantization
                    mcdq_alpha_init=mcdq_alpha_init,
                    use_scs=use_scs,  # C24: Sinkhorn Curvature Scaling
                    scs_eps_scale=scs_eps_scale,
                    use_fixed_curvature=use_fixed_curvature,  # C26: HG-Rec 极简
                    c_fixed=c_fixed,
                )
                for i in range(n_layers)
            ]
        )

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=False,
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features)
            if n_cat_features != 0
            else ReconstructionLoss()
        )

    @cached_property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        # C21 防泄露校验: ckpt 用 hypervq 训练 (含 mlr_a/mlr_r), 但当前模型 hypervq=False →
        # strict=False 会静默丢弃 mlr 参数, tokenizer 用未训练 embedding.weight → SID 塌缩 + 指标虚高
        # (实测: 9922 items 只剩 251 唯一 SID, test R@10 假象 0.27). 必须显式报错阻止.
        ckpt_has_mlr = any("mlr_a" in k or "mlr_r" in k for k in state["model"])
        if ckpt_has_mlr and not self.hypervq:
            raise ValueError(
                "C21 MISMATCH: ckpt 由 hypervq=True 训练 (含 mlr_a/mlr_r 参数), 但当前 RqVae "
                "hypervq=False (用未训练 embedding.weight). 这会导致 codebook SID 塌缩与指标虚高 (数据泄露). "
                "请传入 hypervq=True 保持一致."
            )
        self.load_state_dict(state["model"], strict=False)
        # 兼容多种 ckpt key 命名 (原项目 'iter' / 我们的 'global_step')
        iter_val = state.get("iter", state.get("global_step", "unknown"))
        print(f"---Loaded RQVAE Iter {iter_val}---")

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(self, x: Tensor, gumbel_t: float = 0.001) -> RqVaeOutput:
        x = x.to(next(self.encoder.parameters()).dtype)
        res = self.encode(x)

        quantize_loss = 0
        margins = []  # C5: 各层 per-item margin (d2-d1)
        embs, residuals, sem_ids = [], [], []
        prefix_codes = []  # Issue #154: 累积已选 codeword (router 输入)
        prev_c = None      # 上一层 per-item 有效曲率 (M3 transport c_sig)

        for li, layer in enumerate(self.layers):
            residuals.append(res)
            # Issue #154: prefix_emb = 已选 codeword 拼接 (+ M3 曲率信号)
            prefix_emb = None
            if layer.prefix_routing:
                parts = list(prefix_codes)
                if self.gate_M3_transport and prev_c is not None:
                    from modules.hyperbolic import C_MAX as _C_MAX
                    c_sig = torch.log(prev_c.clamp(min=1e-6)) / math.log(_C_MAX)  # (B,1)
                    parts.append(c_sig)
                prefix_emb = torch.cat(parts, dim=-1) if parts else None
            quantized = layer(res, temperature=gumbel_t, prefix_emb=prefix_emb)
            quantize_loss += quantized.loss
            margins.append(quantized.margin)  # C5
            emb, id = quantized.embeddings, quantized.ids
            if self.gate_M2_intrinsic:
                # M2 (Issue #157): Möbius 内在减法 — Poincaré 球上 residual ⊖ codeword
                # 注意: 不可 detach — 否则 θ 梯度截断, 曲率退化为固定 1.25 (M2/M3 失效)
                c_per = layer.get_c_per_item(prefix_emb)  # (1,) 或 (B,) per-item 曲率
                c_per = c_per.view(-1, 1)                 # (1,1) 或 (B,1)
                h_r = _expmap0_t(res, c_per)
                h_e = _expmap0_t(emb, c_per)
                h_next = _mobius_add_t(-h_e, h_r, c_per)
                res = _logmap0_t(h_next, c_per)
            else:
                res = res - emb
            if self.gate_M3_transport and li < len(self.layers) - 1:
                # M3 (Issue #158): 跨层曲率传输 (fused 径向闭式, 等价 Exp_0^{c_next}(Log_0^{c_l}(res)))
                c_l = layer.get_c_per_item(prefix_emb).view(-1, 1)  # per-item (B,1) 或全局 (1,1)
                c_next = self.layers[li + 1].get_c().view(-1, 1)     # M3: 下一层全局曲率 (treatment 一致)
                res = _transport_between_t(res, c_l, c_next)
                prev_c = c_next.view(1, 1).expand(res.shape[0], 1)  # (B,1) c_sig 用
            sem_ids.append(id)
            embs.append(emb)
            prefix_codes.append(emb)

        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss,
            margins=margins,
        )

    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decode(embs.sum(axis=-1))
        x_hat = torch.cat(
            [l2norm(x_hat[..., : -self.n_cat_feats]), x_hat[..., -self.n_cat_feats :]],
            axis=-1,
        )

        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        # C5: 曲率 margin 正则 — 惩罚 top2-top1 距离太近 (高层量化 margin 极小 → SID 不稳定)
        # margin_l,i = d2 - d1, margin_loss = relu(M_TARGET - margin).mean()
        # 权重 MARGIN_WEIGHT 硬编码 (R30/R43); 梯度经 margin 数值流向 codebook/encoder (STE)
        margin_loss = 0.0
        if self.margin_reg_weight > 0 and quantized.margins:
            m_stack = torch.stack(quantized.margins, dim=1)  # (B, n_layers)
            margin_loss = F.relu(self.margin_target - m_stack).mean()
        loss = (reconstuction_loss + rqvae_loss).mean() + self.margin_reg_weight * margin_loss

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(quantized.sem_ids, "b d -> b 1 d")
                        == rearrange(quantized.sem_ids, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]
            # codebook 健康检查: 每层 unique code 数 (sem_ids: [B, n_layers], 按列统计)
            per_layer_usage = torch.zeros(
                self.n_layers, dtype=torch.long, device=quantized.sem_ids.device
            )
            for li in range(self.n_layers):
                per_layer_usage[li] = quantized.sem_ids[:, li].unique().numel()
            # C5: 各层 mean margin (日志监控)
            per_layer_margin = torch.zeros(
                self.n_layers, dtype=torch.float32, device=quantized.sem_ids.device
            )
            if quantized.margins:
                for li, m in enumerate(quantized.margins):
                    per_layer_margin[li] = m.mean()

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            per_layer_usage=per_layer_usage,
            margin_loss=margin_loss,
            per_layer_margin=per_layer_margin,
        )
