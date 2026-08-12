"""Issue #150 — Bilevel Task-aware Curvature Quantizer.

A (control): 单层优化 — 三层共享曲率只由 RQ-VAE 内部目标更新.
B (treatment): 双层优化 — encoder/decoder/codebook 由内层 RQ-VAE 更新;
                三层曲率 theta_l 由外层 next-item preference loss 的 hypergradient 更新.

Inner-train: 90% 用户 (固定 user hash 划分), 用于内层 RQ-VAE 训练
Meta-train: 10% 用户, 用于外层 preference loss 提供正负样本对

外层 preference loss:
  p_l(b|i) = softmax(-d_{c_l}(r_l,i, e_l,b) / tau_l)
  q_l(i) = sum_b p_l(b|i) e_l,b
  S(i,j) = -sum_l d_{c_l}(q_l(i), q_l(j))
  L_preference = -log sigmoid(S(i,j+) - S(i,j-))

hypergradient:
  phi' = phi - eta_inner * grad_phi L_RQ(inner batch)
  L_outer(theta) = L_preference(meta batch; phi', theta)
  theta <- theta - eta_curv * grad_theta L_outer(theta)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import proj_to_ball, expmap0, logmap0, poincare_distance, kmeans, MLP, _eps

C_MIN = 0.5
C_MAX = 2.0
E_DIM = 32
ETA_INNER = 1e-3
TAU_INIT = 1.0
TAU_MIN = 0.1
TAU_DECAY = 0.95


class SharedCurvatureVQ(nn.Module):
    """单层 VQ, c_l ∈ [C_MIN, C_MAX] 由 sigmoid(theta_l) 控制."""

    def __init__(self, n_e, e_dim=E_DIM, beta=0.25, kmeans_init=True,
                 kmeans_iters=10, sk_eps=0.0, sk_iters=3, layer_idx=0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.layer_idx = layer_idx
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    def get_c(self):
        return C_MIN + (C_MAX - C_MIN) * torch.sigmoid(self.theta)

    def init_emb(self, data):
        if data.shape[0] < self.n_e:
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            centers = kmeans(data, self.n_e, self.kmeans_iters)
            # 缩放到安全 ball 范围 (norm < 0.7) 防止 expmap0 后 norm → 1
            norms = centers.norm(dim=-1, keepdim=True)
            scale = torch.clamp(0.5 / (norms + 1e-8), max=1.0)
            self.embeddings.weight.data.copy_(centers * scale)
        self.initted = True

    def forward(self, x, use_sk=True, return_dist=False):
        """返回 x_q, loss, indices; 如果 return_dist=True 还返回 (B, K) 距离矩阵."""
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        B = latent.shape[0]
        K = self.n_e
        c = self.get_c()
        latent_h = proj_to_ball(expmap0(latent, c), c)
        codebook_h = proj_to_ball(expmap0(self.embeddings.weight, c), c)
        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
        d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)
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
        x_q = self.embeddings.weight.index_select(0, indices)
        x_q_proj = proj_to_ball(x_q, c)
        latent_proj = proj_to_ball(latent, c)
        commitment_loss = torch.mean(poincare_distance(x_q_proj.detach(), latent_proj, c) ** 2)
        codebook_loss = torch.mean(poincare_distance(x_q_proj, latent_proj.detach(), c) ** 2)
        loss = commitment_loss + self.beta * codebook_loss
        x_q = logmap0(x_q_proj, c)
        latent = logmap0(latent_proj, c)
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        if return_dist:
            return x_q, loss, indices, d
        return x_q, loss, indices


class BilevelHRQVAE(nn.Module):
    """HRQVAE used by both A (single-level) and B (bilevel)."""

    def __init__(self, in_dim=768, num_emb_list=(64, 128, 256), e_dim=E_DIM,
                 layers=(512, 256, 128, 64), beta=0.25, kmeans_init=True, kmeans_iters=10):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.e_dim = e_dim
        self.layers = layers
        encode_layer_dims = [in_dim] + list(layers) + [e_dim]
        self.encoder = MLP(layers=encode_layer_dims, dropout=0.0, use_bn=False)
        decode_layer_dims = encode_layer_dims[::-1]
        self.decoder = MLP(layers=decode_layer_dims, dropout=0.0, use_bn=False)
        self.vq_layers = nn.ModuleList([
            SharedCurvatureVQ(n_e, e_dim=e_dim, beta=beta,
                              kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                              layer_idx=i)
            for i, n_e in enumerate(num_emb_list)
        ])

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self._rq_forward(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices, z_q, z

    def _rq_forward(self, x, use_sk=True):
        all_losses, all_indices = [], []
        x_q = 0
        residual = x
        for q in self.vq_layers:
            x_res, loss, idx = q(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices

    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        _, _, indices = self._rq_forward(z, use_sk=use_sk)
        return indices

    def get_curvatures(self):
        return [float(q.get_c().item()) for q in self.vq_layers]


def soft_assignment_logits(r, codebook_h, c, tau):
    """Compute p_l(b|i) = softmax(-d_c(r_i, e_b) / tau) for one layer.

    Args:
        r: (B, D) latent in hyperbolic space (already projected)
        codebook_h: (K, D) codebook in hyperbolic space (already projected)
        c: scalar curvature
        tau: scalar temperature
    Returns:
        p: (B, K) soft assignment probabilities
    """
    B = r.shape[0]
    K = codebook_h.shape[0]
    r_exp = r.unsqueeze(1).expand(B, K, -1)
    cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
    d = poincare_distance(r_exp, cb_exp, c).squeeze(-1)  # (B, K)
    logits = -d / tau.clamp_min(1e-3)
    return F.softmax(logits, dim=-1)


def soft_codeword(p, embeddings, c):
    """Compute q_l(i) = sum_b p_l(b|i) * e_l,b in hyperbolic space.

    Args:
        p: (B, K) soft assignment
        embeddings: (K, D) codeword embeddings
        c: scalar curvature
    Returns:
        q: (B, D) soft codeword (in tangent space at 0)
    """
    # Logmap codewords to tangent space, weighted average, then expmap back
    emb_h = proj_to_ball(expmap0(embeddings, c), c)
    emb_tan = logmap0(emb_h, c)  # (K, D) in tangent space at 0
    q_tan = torch.matmul(p, emb_tan)  # (B, D)
    q_h = proj_to_ball(expmap0(q_tan, c), c)
    return q_h


def preference_loss(model, item_emb, item_i, item_j_pos, item_j_neg, tau):
    """Compute outer preference loss given model state.

    Args:
        model: BilevelHRQVAE (must be in eval mode for gradient flow through c only)
        item_emb: (N_items, in_dim) full item embedding table (for looking up by item id)
        item_i: (B,) anchor item indices
        item_j_pos: (B,) positive next-item indices
        item_j_neg: (B,) negative next-item indices
        tau: scalar temperature for soft assignment
    Returns:
        loss: scalar preference loss
        metrics: dict with margin, score_pos, score_neg
    """
    # Encode items (use no_grad for encoder/decoder since we only update c via outer)
    # For full differentiability through the soft path, we DO need gradient through encoder
    # but the spec says outer gradient only updates theta. We'll let gradient flow
    # through encoder too but not step on it (encoder doesn't get updated).
    x_i = item_emb[item_i]      # (B, D_in)
    x_jp = item_emb[item_j_pos]
    x_jn = item_emb[item_j_neg]
    # Encode
    z_i = model.encoder(x_i)
    z_jp = model.encoder(x_jp)
    z_jn = model.encoder(x_jn)
    # Three-layer soft codeword for each (anchor / pos / neg)
    def per_item_soft_codeword(z):
        r = z
        soft_qs = []
        for q in model.vq_layers:
            c = q.get_c()
            latent_h = proj_to_ball(expmap0(r, c), c)
            cb_h = proj_to_ball(expmap0(q.embeddings.weight, c), c)
            p = soft_assignment_logits(latent_h, cb_h, c, tau)
            qh = soft_codeword(p, q.embeddings.weight, c)
            soft_qs.append(qh)
            r = r - logmap0(q.embeddings.weight[q.initted and 0].unsqueeze(0).expand_as(r), c) if False else r
        return soft_qs
    # Compute per-layer soft codeword for anchor/pos/neg
    def all_layers_soft(z):
        soft_qs = []
        r = z
        for q in model.vq_layers:
            c = q.get_c()
            latent_h = proj_to_ball(expmap0(r, c), c)
            cb_h = proj_to_ball(expmap0(q.embeddings.weight, c), c)
            p = soft_assignment_logits(latent_h, cb_h, c, tau)
            qh = soft_codeword(p, q.embeddings.weight, c)
            soft_qs.append(qh)
            # The "r" for next layer doesn't matter for soft codeword computation
            # because we compute soft per-layer independently
        return soft_qs
    qs_i = all_layers_soft(z_i)
    qs_jp = all_layers_soft(z_jp)
    qs_jn = all_layers_soft(z_jn)
    # S(i, j) = -sum_l d_{c_l}(q_l(i), q_l(j))
    def score(qs_a, qs_b):
        s = 0.0
        for qa, qb, q in zip(qs_a, qs_b, model.vq_layers):
            c = q.get_c()
            d = poincare_distance(qa, qb, c).squeeze(-1)
            s = s - d.mean()
        return s
    s_pos = score(qs_i, qs_jp)
    s_neg = score(qs_i, qs_jn)
    margin = s_pos - s_neg
    loss = -F.logsigmoid(margin)
    return loss, {"margin": margin.detach(), "s_pos": s_pos.detach(), "s_neg": s_neg.detach()}
