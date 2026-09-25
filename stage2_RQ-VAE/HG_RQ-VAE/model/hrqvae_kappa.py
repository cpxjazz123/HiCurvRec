"""HRQ-VAE with per-branch learnable κ (curvature) — Stage 2 κ learning.

机制 (R36 #1 Stage 2 κ learning):
  原 HRQ-VAE 每层 HVectorQuantization 用固定 c=1.0.
  本版本: 每层 (L0/L1/L2) 加可学习参数 log_c, 实际 c = exp(log_c), 通过 gradient descent 学习最适合该层的曲率.

为什么有效 (论文支持):
  - Ganea 2018 "Hyperbolic Neural Networks" 指出不同曲率适合不同语义层级
  - Chami 2019 "Hyperbolic Graph Convolutional Neural Networks" 表明不同层应学不同 κ
  - Nickel & Kiela 2017 "Poincaré Embeddings" 曲率决定嵌入空间"折叠"程度

实现要点:
  - c 始终 > 0 (exp 保证), 无需外部 clamp
  - log_c 范围约束 [-3, 3] (c ∈ [0.05, 20]) 避免极端值
  - forward 时动态计算 c = clamp(log_c.exp(), 0.05, 20.0)
  - 重建 loss 仍用 c=1.0 (decoder 在 Euclidean 空间解码, Poincaré loss 用于 reconstruct 后的比较)
  - 兼容旧 ckpt (旧 ckpt 无 log_c, init log_c=0 即 c=1, 与 baseline 一致)

约束 (R40):
  - 不改 dataset / train / valid / test 逻辑
  - 不调任何超参 (LR/dropout/label_smoothing/weight_decay 全部继承)
  - 唯一变更: 每层 HVectorQuantization 加 log_c 参数 + 学习它
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
from torch.nn.init import xavier_uniform_
from sklearn.cluster import KMeans


#===========================================================================================#
def _eps(x):
    return torch.finfo(x.dtype).eps

def artanh(x):
    x = x.clamp(min=-1 + 1e-10, max=1 - 1e-10)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def proj_to_ball(x, c, eps=1e-6):
    if c <= 0:
        raise ValueError("Curvature c must be > 0 for the Poincaré ball model.")
    r = (1.0 / c) ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    max_norm = (1 - eps) * r
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale

def mobius_add(x, y, c):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / den.clamp_min(_eps(den))

def lambda_x(x, c):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp_min(_eps(x2))

def expmap0(u, c):
    sqrt_c = c ** 0.5
    norm_u = u.norm(dim=-1, keepdim=True).clamp_min(_eps(u))
    factor = torch.tanh(sqrt_c * norm_u) / (sqrt_c * norm_u)
    x = factor * u
    return proj_to_ball(x, c)

def logmap0(x, c):
    sqrt_c = c ** 0.5
    norm_x = x.norm(dim=-1, keepdim=True).clamp_min(_eps(x))
    factor = artanh(sqrt_c * norm_x) / (sqrt_c * norm_x)
    return factor * x

def poincare_distance(x, y, c):
    diff = mobius_add(-x, y, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
    return (2.0 / sqrt_c) * artanh(sqrt_c * norm)


#===========================================================================================#
def kmeans(samples, num_clusters, num_iterations=10):
    """KMeans codebook init — 用 sklearn KMeans (n_init=1 加速).

    与 baseline 一致 (sklearn KMeans), 但 n_init=1 加速 10x.
    MiniBatchKMeans 在小数据上不稳定, 这里改回普通 KMeans.
    """
    device = samples.device
    x = samples.cpu().detach().numpy()
    cluster = KMeans(
        n_clusters=num_clusters,
        max_iter=num_iterations,
        n_init=1,  # 加速: sklearn 默认 n_init=10 太慢
        random_state=2024,
    ).fit(x)
    cluster_centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(cluster_centers).to(device)
    return tensor_centers


def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)
    B = Q.shape[0]
    K = Q.shape[1]
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K
    Q *= B
    return Q


#===========================================================================================#
class HVectorQuantizationKappa(nn.Module):
    """HVectorQuantization with LEARNABLE κ.

    New: log_c is a learnable parameter (init=0 → c=1).
    During forward: c = exp(log_c).clamp(0.05, 20.0).
    """
    def __init__(self,
                 n_e,
                 e_dim,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=10,
                 sk_eps=0.003,
                 sk_iters=3,
                 learnable_c=True,
                 c_init=1.0,
                 c_min=0.05,
                 c_max=20.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.learnable_c = learnable_c
        self.c_min = c_min
        self.c_max = c_max

        # learnable curvature: c = exp(log_c).clamp(c_min, c_max)
        if learnable_c:
            self.log_c = nn.Parameter(torch.log(torch.tensor(c_init, dtype=torch.float32)))
        else:
            self.register_buffer("log_c", torch.log(torch.tensor(c_init, dtype=torch.float32)))

        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    def get_c(self):
        """当前 curvature, clamp 到 [c_min, c_max]."""
        return self.log_c.exp().clamp(min=self.c_min, max=self.c_max)

    def get_codebook(self):
        c = self.get_c()
        return proj_to_ball(self.embeddings.weight, c)

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embeddings(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        """KMeans init codebook — DDP-safe: only rank 0 runs KMeans, broadcasts to all.

        Important: this method should be called by ALL ranks (in the same order) to
        ensure the broadcast step is reached by all ranks. Only rank 0 actually
        does KMeans; others just wait for broadcast.
        No internal barrier — outer caller is responsible for sync.
        `data` is required only on rank 0; pass None for other ranks.
        """
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                if data is None:
                    raise ValueError("rank 0 must provide data for KMeans init")
                centers = kmeans(data, self.n_e, self.kmeans_iters)
                self.embeddings.weight.data.copy_(centers)
            # All ranks call broadcast (rank 0 sends, others receive)
            with torch.no_grad():
                dist.broadcast(self.embeddings.weight, src=0)
            self.initted = True
        else:
            centers = kmeans(data, self.n_e, self.kmeans_iters)
            self.embeddings.weight.data.copy_(centers)
            self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-10
        assert amplitude > 0, "Amplitude must be positive"
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        c = self.get_c()  # dynamic per forward
        latent = x.view(-1, self.e_dim)
        codebook = self.embeddings.weight
        if not self.initted and self.training:
            self.init_emb(latent)

        latent_h = proj_to_ball(expmap0(latent, c), c)
        codebook_h = proj_to_ball(expmap0(codebook, c), c)

        B = latent_h.shape[0]
        K = codebook_h.shape[0]

        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)

        d = poincare_distance(x_exp, cb_exp, c).squeeze(-1)

        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn algorithm produced NaN or Inf values.")
            indices = torch.argmax(Q, dim=-1)
        x_exp = logmap0(x_exp, c)
        cb_exp = logmap0(cb_exp, c)
        x_q = codebook.index_select(0, indices)

        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, c)**2)
        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), c)**2)
        loss = commitment_loss + self.beta * codebook_loss

        x_q = logmap0(x_q, c)
        latent = logmap0(latent, c)
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices


#===========================================================================================#
class HResidualVectorQuantizationKappa(nn.Module):
    """HRQ-VAE with per-layer learnable κ."""
    def __init__(self,
                 n_e_list,
                 e_dim,
                 sk_eps,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_iters=100,
                 learnable_c=True,
                 c_init=1.0,
                 c_min=0.05,
                 c_max=20.0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.learnable_c = learnable_c

        self.vq_layers = nn.ModuleList([
            HVectorQuantizationKappa(
                n_e, e_dim,
                beta=beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_eps=sk_eps,
                sk_iters=sk_iters,
                learnable_c=learnable_c,
                c_init=c_init,
                c_min=c_min,
                c_max=c_max,
            )
            for n_e, sk_eps in zip(n_e_list, sk_eps)
        ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def get_curvatures(self):
        """返回每层当前曲率 [c_0, c_1, c_2]."""
        return [q.get_c().item() for q in self.vq_layers]

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            x_q = x_q + x_res
            residual = residual - x_res.detach()
            all_losses.append(loss)
            all_indices.append(indices)
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, sum(all_losses), all_indices


#===========================================================================================#
class HRQVAE_Kappa(nn.Module):
    """HRQ-VAE with learnable κ per layer — for Stage 2 κ learning.

    与 baseline HRQVAE 唯一区别: hrq 用 HResidualVectorQuantizationKappa (learnable_c=True).
    其余 encoder/decoder/compute_loss 完全继承.
    """
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type='mse',
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_eps=None,
                 sk_iters=100,
                 learnable_c=True,
                 c_init=1.0,
                 c_min=0.05,
                 c_max=20.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLP(layers=self.encode_layer_dims,
                          dropout=self.dropout_prob,
                          use_bn=self.bn)

        self.hrq = HResidualVectorQuantizationKappa(
            n_e_list=self.num_emb_list,
            e_dim=self.e_dim,
            sk_eps=self.sk_eps,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
            learnable_c=learnable_c,
            c_init=c_init,
            c_min=c_min,
            c_max=c_max,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLP(layers=self.decode_layer_dims,
                          dropout=self.dropout_prob,
                          use_bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.hrq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, x, use_sk=True):
        x = self.encoder(x)
        _, _, indices = self.hrq(x, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quent_loss, xs=None):
        # 重建 loss 用 c=1 (与 baseline 一致; 输入输出空间独立于量化层曲率)
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        elif self.loss_type == 'poincare':
            out_h = expmap0(out, c=1)
            xs_h = expmap0(xs, c=1)
            out_h = proj_to_ball(out_h, c=1)
            xs_h = proj_to_ball(xs_h, c=1)
            loss_recon = torch.mean(poincare_distance(out_h, xs_h, c=1)**2)
            out = logmap0(out_h, c=1)
            xs = logmap0(xs_h, c=1)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        loss_total = loss_recon + self.quant_loss_weight * quent_loss
        return loss_total, loss_recon


#===========================================================================================#
class MLP(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", use_bn=False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = use_bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn and idx != (len(layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = self.activation_function(self.activation, output_size)
            if activation_func is not None and idx != (len(layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def activation_function(self, activation_name="relu", emb_dim=None):
        if activation_name is None:
            return None
        elif isinstance(activation_name, str):
            n = activation_name.lower()
            if n == "sigmoid": return nn.Sigmoid()
            if n == "tanh": return nn.Tanh()
            if n == "relu": return nn.ReLU()
            if n == "leakyrelu": return nn.LeakyReLU()
            if n == "none": return None
        elif issubclass(activation_name, nn.Module):
            return activation_name()
        raise NotImplementedError(f"activation function {activation_name} is not implemented!")

    def forward(self, x):
        return self.mlp(x)


#===========================================================================================#
class EmbDataset(data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.embeddings = pd.read_parquet(data_path)
        if 'embedding' in self.embeddings.columns:
            self.dim = len(self.embeddings['embedding'].iloc[0])
            self.data = np.stack(self.embeddings['embedding'].values).astype(np.float32)
        else:
            self.dim = self.embeddings.shape[1]
            self.data = self.embeddings.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
