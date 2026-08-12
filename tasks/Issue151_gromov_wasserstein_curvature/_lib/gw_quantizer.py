"""Issue #151 — Assignment-coupled Gromov-Wasserstein Curvature.

Shared per-layer curvature `c_l = C_MIN + (C_MAX-C_MIN) * sigmoid(theta_l)`
with alternating RQ + GW step (GW only updates theta_l).

GW step:
- D_ref,l(i,j) = ||stop_grad(r_l,i) - stop_grad(r_l,j)||_2  (curvature-independent)
- D_code,l(b,b';c_l) = Poincare distance between codeword b and b'
- P_l(i,b) = sparse softmax(-D_code,l(i,b)/tau) (top-M=8)
- s_l* = closed-form optimal positive scale
- L_GW,l = normalized weighted structural error
- Only theta_l gets gradient; everything else stop-grad.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use baseline's poincare utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "baseline" / "_lib"))
from utils import expmap0, proj_to_ball, logmap0, poincare_distance, MLP  # noqa: E402


# Hardcoded hyperparams (R30/R43)
EMB_DIM = 768
E_DIM = 32
CODEBOOK_SIZES = [64, 128, 256]
ENCODER_LAYERS = [512, 256, 128, 64]
N_LAYERS = 3
C_MIN = 0.5
C_MAX = 2.0
BATCH_SIZE = 256
N_EPOCHS = 100
SEED = 42
TAU_GW = 0.5       # GW softmax temperature
TOP_M = 8          # top-M sparse coupling
PAIR_SAMPLE = 64   # item-pair sampling count for GW (per mini-batch)
N_ITEMS = 9922
GW_LAMBDA = 1.0    # GW loss weight (1.0 — primary signal for treatment)
REC_LAMBDA = 1.0   # InfoNCE loss weight
REC_TAU = 1.0
LR_PHI = 1e-3
LR_THETA = 1e-2    # GW updates theta faster than phi

THETA_INIT = math.log((1.0 - C_MIN) / (C_MAX - C_MIN) / (1.0 - (1.0 - C_MIN) / (C_MAX - C_MIN)))


# ─────────────────────────────────────────────────────────────────
# Shared curvature VQ layer
# ─────────────────────────────────────────────────────────────────
class SharedCurvatureVQ(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 sk_eps=0.0, sk_iters=3):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.theta = nn.Parameter(torch.tensor(THETA_INIT, dtype=torch.float32))
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.1, 0.1)
            self.initted = True
        else:
            self.embeddings.weight.data.zero_()
            self.initted = False

    def get_c(self) -> torch.Tensor:
        return C_MIN + (C_MAX - C_MIN) * torch.sigmoid(self.theta)

    def init_emb(self, data):
        from utils import kmeans
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        c = self.get_c()
        cb_t = self.embeddings.weight
        cb_ball = proj_to_ball(expmap0(cb_t, c), c)
        latent_ball = proj_to_ball(expmap0(latent, c), c)
        lat_e = latent_ball.unsqueeze(1).expand(-1, self.n_e, -1)
        cb_e = cb_ball.unsqueeze(0).expand(latent.shape[0], -1, -1)
        d = poincare_distance(lat_e, cb_e, c).squeeze(-1)
        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            from utils import sinkhorn_algorithm, center_distance_for_constraint
            d_centered = center_distance_for_constraint(d).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            indices = torch.argmax(Q, dim=-1)
        x_q_t = self.embeddings(indices)
        x_q_ball = proj_to_ball(expmap0(x_q_t, c), c)
        cl = torch.mean(poincare_distance(x_q_ball.detach(), latent_ball, c) ** 2)
        ql = torch.mean(poincare_distance(x_q_ball, latent_ball.detach(), c) ** 2)
        loss = cl + self.beta * ql
        x_q_t_out = logmap0(x_q_ball, c)
        latent_t = logmap0(latent_ball, c)
        x_q_out = latent_t + (x_q_t_out - latent_t).detach()
        indices = indices.view(x.shape[:-1])
        return x_q_out, loss, indices


class GWHRQVAE(nn.Module):
    def __init__(self, in_dim=EMB_DIM, num_emb_list=CODEBOOK_SIZES, e_dim=E_DIM,
                 layers=ENCODER_LAYERS, beta=0.25, kmeans_init=True, kmeans_iters=10,
                 sk_eps=(0.0, 0.0, 0.0), sk_iters=3):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.e_dim = e_dim
        encode_layer_dims = [in_dim] + layers + [e_dim]
        self.encoder = MLP(layers=encode_layer_dims, dropout=0.0, use_bn=False)
        decode_layer_dims = encode_layer_dims[::-1]
        self.decoder = MLP(layers=decode_layer_dims, dropout=0.0, use_bn=False)
        self.vq_layers = nn.ModuleList([
            SharedCurvatureVQ(n_e, e_dim, beta=beta, kmeans_init=kmeans_init,
                              kmeans_iters=kmeans_iters, sk_eps=eps, sk_iters=sk_iters)
            for n_e, eps in zip(num_emb_list, sk_eps)
        ])

    def forward(self, x, use_sk=True):
        z = self.encoder(x)
        z_q, rq_loss, indices = self._rq_forward(z, use_sk=use_sk)
        out = self.decoder(z_q)
        return out, rq_loss, indices, z_q, z

    def _rq_forward(self, x, use_sk=True, return_residuals=False):
        all_losses, all_indices, residuals = [], [], []
        x_q = 0
        residual = x
        for q in self.vq_layers:
            x_res, loss, idx = q(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(idx)
            if return_residuals:
                residuals.append(residual.detach())  # pre-quant residual
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        if return_residuals:
            return x_q, mean_loss, all_indices, residuals
        return x_q, mean_loss, all_indices

    def get_indices(self, x, use_sk=True):
        z = self.encoder(x)
        _, _, indices = self._rq_forward(z, use_sk=use_sk)
        return indices


# ─────────────────────────────────────────────────────────────────
# Gromov-Wasserstein step (vectorized per layer)
# ─────────────────────────────────────────────────────────────────
def gromov_wasserstein_loss(model: GWHRQVAE, x, rng):
    """Compute GW loss using RQ residuals + codebook distances.

    Only theta gets gradient; everything else stop-grad.
    Returns: scalar GW loss (sum over layers).
    """
    device = next(model.parameters()).device
    z = model.encoder(x)
    _, _, _, residuals = model._rq_forward(z, use_sk=False, return_residuals=True)
    # residuals[l] is the pre-quant residual at layer l (stop-grad Euclidean anchor)

    B = x.shape[0]
    B_p = min(PAIR_SAMPLE, B)
    sample_idx = torch.randint(0, B, (B_p,), generator=rng, device=device)
    M = TOP_M

    gw_total = torch.tensor(0.0, device=device)
    for l, q in enumerate(model.vq_layers):
        c_l = q.get_c()
        r_sg = residuals[l][sample_idx].detach()  # (B_p, D) — curvature-independent

        # D_ref(i,j) = ||r_i - r_j||_2 — full (B_p, B_p) Euclidean
        diff = r_sg.unsqueeze(0) - r_sg.unsqueeze(1)
        D_ref = (diff * diff).sum(-1).sqrt().clamp(min=1e-8)  # (B_p, B_p)

        # Codebook ball coords — codeword tangent coords stop-grad, but c_l carries grad
        cw_t = q.embeddings.weight.detach()  # stop-grad coords
        cw_ball = proj_to_ball(expmap0(cw_t, c_l), c_l)  # (K, D) — grad through c_l
        K_l = q.n_e
        cb1 = cw_ball.unsqueeze(0).expand(K_l, -1, -1)
        cb2 = cw_ball.unsqueeze(1).expand(-1, K_l, -1)
        D_code = poincare_distance(cb1, cb2, c_l).squeeze(-1)  # (K, K) — grad through c_l

        # P_l(i,b): top-M sparse coupling (distances from r_ball to codewords, all stop-grad)
        with torch.no_grad():
            r_ball_sg = proj_to_ball(expmap0(r_sg, c_l.detach()), c_l.detach())
            cb_e = cw_ball.detach().unsqueeze(0).expand(B_p, -1, -1)
            r_e = r_ball_sg.unsqueeze(1).expand(-1, K_l, -1)
            d_full = poincare_distance(r_e, cb_e, c_l.detach()).squeeze(-1)  # (B_p, K)
            topk_vals, topk_idx = torch.topk(-d_full, M, dim=-1)
            logits = -d_full.gather(1, topk_idx) / TAU_GW
            soft = torch.softmax(logits, dim=-1)
            P_sparse = torch.zeros_like(d_full)
            P_sparse.scatter_(1, topk_idx, soft)
            row_sum = P_sparse.sum(1, keepdim=True).clamp(min=1e-12)
            P_norm = (P_sparse / row_sum).detach()

        # M_ref[b,b'] = sum_{i,j} P[i,b] P[j,b']
        M_ref = P_norm.t() @ P_norm  # (K, K)
        # D_ref weighted by P_norm per (b,b'): D_ref_agg[b,b'] = <D_ref, P[:,b] outer P[:,b']> / M_ref[b,b']
        w_Dref = (P_norm.t() @ D_ref @ P_norm)  # (K, K)
        M_ref_safe = M_ref.clamp(min=1e-12)
        D_ref_agg = (w_Dref / M_ref_safe).detach()  # (K, K)

        # Closed-form s_l*
        with torch.no_grad():
            num = (M_ref * D_ref_agg * D_code.detach()).sum()
            den = (M_ref * (D_code.detach() ** 2)).sum().clamp(min=1e-12)
            s_star = (num / den).clamp(min=1e-6).detach()

        # GW loss: grad only through D_code(c_l)
        diff_struct = D_ref_agg - s_star * D_code
        loss_layer = (M_ref.detach() * diff_struct * diff_struct).sum() / max(M_ref.sum().item(), 1e-12)
        gw_total = gw_total + loss_layer

    return gw_total


def poincare_recon_loss(out, target, c=1.0):
    out_b = proj_to_ball(expmap0(out, c), c)
    target_b = proj_to_ball(expmap0(target, c), c)
    return torch.mean(poincare_distance(out_b, target_b, c) ** 2)