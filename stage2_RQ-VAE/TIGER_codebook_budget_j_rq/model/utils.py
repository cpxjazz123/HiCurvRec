import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import datetime
import os

from torch.nn.init import xavier_uniform_
from sklearn.cluster import KMeans

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


#===========================================================================================#


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
