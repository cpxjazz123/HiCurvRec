import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import datetime
import os

from torch.nn.init import xavier_uniform_
from sklearn.cluster import KMeans

#===========================================================================================#
def _eps(x):
    return torch.finfo(x.dtype).eps

def artanh(x):
    x = x.clamp(min=-1 + 1e-10, max=1 - 1e-10)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def proj_to_ball(x, c, eps = 1e-6):
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
class MLP(nn.Module):
    def __init__(self, 
                 layers,
                 dropout = 0.0,
                 activation = "relu",
                 use_bn = False):
        super(MLP, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = use_bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(layers[:-1], layers[1:])
            ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))

            if self.use_bn and idx != (len(layers) - 2):
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))

            activetion_func = self.activation_function(self.activation, output_size)
            if activetion_func is not None and idx != (len(layers) - 2):
                mlp_modules.append(activetion_func)
            
        self.mlp = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def activation_function(self, activation_name="relu", emb_dim=None):
        if activation_name is None:
            activation = None
        elif isinstance(activation_name, str):
            if activation_name.lower() == "sigmoid":
                activation = nn.Sigmoid()
            elif activation_name.lower() == "tanh":
                activation = nn.Tanh()
            elif activation_name.lower() == "relu":
                activation = nn.ReLU()
            elif activation_name.lower() == "leakyrelu":
                activation = nn.LeakyReLU()
            elif activation_name.lower() == "none":
                activation = None
        elif issubclass(activation_name, nn.Module):
            activation = activation_name()
        else:
            raise NotImplementedError(
                f"activation function {activation_name} is not implemented!"
            )
        return activation
    
    def forward(self, x):
        return self.mlp(x)
#===========================================================================================#

def kmeans(
        samples,
        num_clusters,
        num_iterations=10):
    device = samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(
        n_clusters = num_clusters,
        max_iter = num_iterations,
        random_state = 2024,  # R41c: 固定 KMeans 中心 → 对照实验初始 codebook 完全一致
    ).fit(x)

    cluster_centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(cluster_centers).to(device)

    return tensor_centers

def sinkhorn_algorithm(
        distances,
        epsilon,
        sinkhorn_iterations): 
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q

#===========================================================================================#
class HVectorQuantization(nn.Module):
    def __init__(self, 
                 n_e,
                 e_dim,
                 beta=0.25,
                 kmeans_init = False,
                 kmeans_iters = 10,
                 sk_eps = 0.003,
                 sk_iters = 3):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_eps = sk_eps
        self.sk_iters = sk_iters
        self.c = 1.0
        if self.c <= 0:
            raise ValueError("Curvature c must be > 0.")
        
        self.embeddings = nn.Embedding(n_e, e_dim)
        if not kmeans_init:
            self.initted = True
            with torch.no_grad():
                self.embeddings.weight.data.uniform_(-0.01, 0.01)
        else:
            self.initted = False
            self.embeddings.weight.data.zero_()

    
    def get_codebook(self):
        return proj_to_ball(self.embeddings.weight, self.c)
    
    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embeddings(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(data, 
                         self.n_e, 
                         self.kmeans_iters)
        self.embeddings.weight.data.copy_(centers)
        self.initted = True
    
    @staticmethod
    def center_distance_for_constraint(distances):
        
        max_distance = distances.max()
        min_distance = distances.min()
        #print(max_distance, min_distance)

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-10
        assert amplitude > 0, "Amplitude must be positive"
        centered_distances = (distances - middle) / amplitude

        return centered_distances
    
    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)
        codebook = self.embeddings.weight  #(codebook_size, e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        
        latent_h = proj_to_ball(expmap0(latent, self.c), self.c)   #(Batch_size, e_dim)
        codebook_h = proj_to_ball(expmap0(codebook, self.c), self.c)  #(codebook_size, e_dim)

        B = latent_h.shape[0]
        K = codebook_h.shape[0]

        x_exp = latent_h.unsqueeze(1).expand(B, K, -1)      # (B, K, D+1)
        cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)
        
        d = poincare_distance(x_exp, cb_exp, self.c).squeeze(-1) # (B,K)
        #d = torch.nn.functional.normalize(d, p=2, dim=-1)
        
        if not use_sk or self.sk_eps <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(
                d_centered, 
                self.sk_eps, 
                self.sk_iters
            )

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise ValueError("Sinkhorn algorithm produced NaN or Inf values.")
            indices = torch.argmax(Q, dim=-1)
        x_exp = logmap0(x_exp, self.c)
        cb_exp = logmap0(cb_exp, self.c)
        x_q = codebook.index_select(0, indices)
        
        commitment_loss = torch.mean(poincare_distance(x_q.detach(), latent, self.c)**2)
        codebook_loss = torch.mean(poincare_distance(x_q, latent.detach(), self.c)**2)
        
        loss = commitment_loss + self.beta * codebook_loss
        
        x_q = logmap0(x_q, self.c)
        latent = logmap0(latent, self.c)
        x_q = x + (x_q - x).detach()
        '''
        x_q = self.embeddings(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q.detach(), latent)
        codebook_loss = F.mse_loss(x_q, latent.detach())
        loss = commitment_loss + self.beta * codebook_loss
        x_q = x + (x_q - x).detach()
        '''
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices

#===========================================================================================#
class HResidualVectorQuantization(nn.Module):
    def __init__(self, 
                 n_e_list,
                 e_dim,
                 sk_eps,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_iters=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.sk_eps = sk_eps
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList([HVectorQuantization(n_e, 
                                                           e_dim, 
                                                           beta = self.beta, 
                                                           kmeans_init = self.kmeans_init,
                                                           kmeans_iters = self.kmeans_iters, 
                                                           sk_eps = sk_eps,
                                                           sk_iters = self.sk_iters)
                                                           for n_e, sk_eps in zip(n_e_list, sk_eps)])
        
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook() # type: ignore
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
        
        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_loss, all_indices

#===========================================================================================#
class EmbDataset(data.Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data_path = data_path
        self.embeddings = pd.read_parquet(data_path)['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0) # type: ignore
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb
    
    def __len__(self):
        return len(self.embeddings)
    
#===========================================================================================#
def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
