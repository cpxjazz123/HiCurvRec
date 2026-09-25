import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from model.utils import *

class HRQVAE(nn.Module):
    def __init__(self,
                 in_dim = 768,
                 num_emb_list = None,
                 e_dim = 64,
                 layers = None,
                 dropout_prob = 0.0,
                 bn = False,
                 loss_type = 'mse',
                 quant_loss_weight = 1.0,
                 beta = 0.25,
                 kmeans_init = False,
                 kmeans_iters = 100,
                 sk_eps = None,
                 sk_iters = 100):
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

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim] # type: ignore
        self.encoder = MLP(layers=self.encode_layer_dims,
                          dropout=self.dropout_prob,
                          use_bn=self.bn)
        
        self.hrq = HResidualVectorQuantization(n_e_list=self.num_emb_list,
                                             e_dim=self.e_dim,
                                             sk_eps=self.sk_eps,
                                             beta=self.beta,
                                             kmeans_init=self.kmeans_init,
                                             kmeans_iters=self.kmeans_iters,
                                             sk_iters=self.sk_iters)
        
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
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean') # type: ignore
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean') # type: ignore
        elif self.loss_type == 'poincare':
            out = expmap0(out, c=1)
            xs = expmap0(xs, c=1)
            out = proj_to_ball(out, c=1)
            xs = proj_to_ball(xs, c=1)
            loss_recon = torch.mean(poincare_distance(out, xs, c=1)**2)
            out = logmap0(out, c=1)
            xs = logmap0(xs, c=1)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        loss_total = loss_recon + self.quant_loss_weight * quent_loss
        return loss_total, loss_recon
