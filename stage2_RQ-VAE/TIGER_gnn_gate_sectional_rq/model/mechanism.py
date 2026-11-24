"""GNN gates enter only codeword costs; fixed curvature never reads gate weights."""
from pathlib import Path
import sys
import numpy as np
import torch
from .model import RQVAE as OriginalRQVAE
from torch.utils.checkpoint import checkpoint
from .geometry import mixed_cost, fixed_squared
from .selection import select
from losses.commitment import commitment_loss
from losses.codebook import codebook_loss


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'stage2_RQ-VAE').is_dir():
            return parent
    raise RuntimeError('missing repository marker')


ROOT = repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.graph_infra.ego_gnn import EgoStore
from .structure import DirectedGate
from .curvature import FrozenCurvature


class RQVAE(OriginalRQVAE):
    def __init__(self, config, *, in_dim, distance=None):
        super().__init__(config,in_dim=in_dim,distance=distance)
        self.structure = DirectedGate()
        self.store = EgoStore(ROOT,ROOT/'assets')
        self.targets = FrozenCurvature(ROOT/'assets')
        self.register_buffer('tangent_scale',torch.tensor(1.))
        self.register_buffer('curvature_weight',torch.tensor(float('nan')))
        self.register_buffer('target_scale',torch.tensor(self.targets.scale))
        self.behavior_on = True
        self.curvature_on = True
        self.cut_selection_cost = False
        self.last_q = None
        self.last_hard = None
        self.last_costs = []
        self.last_weights = None

    def _identities(self, embeddings, item_ids):
        if item_ids is None:
            if len(embeddings) != self.store.n:
                raise ValueError('item_ids required outside an ordered full-catalogue forward')
            item_ids = torch.arange(self.store.n,device=embeddings.device)
        ids = torch.as_tensor(item_ids,device=embeddings.device,dtype=torch.long)
        if ids.shape != (len(embeddings),) or bool((ids < 0).any()) or bool((ids >= self.store.n).any()):
            raise ValueError('item identity outside the global catalogue')
        return ids

    def assignment_cost(self, residual, codewords, weights):
        return mixed_cost(residual,codewords,weights,scale=self.tangent_scale,
                          curvature_on=self.curvature_on)

    def quantize(self, encoded, weights, *, balanced):
        residual=encoded
        quantized=torch.zeros_like(encoded)
        hard_sum=torch.zeros_like(encoded)
        tokens=[];parts=[];costs=[];unused=0
        for layer in self.rq.vq_layers:
            cost=(checkpoint(self.assignment_cost,residual,layer.embed.weight,weights,use_reentrant=False)
                  if torch.is_grad_enabled() else self.assignment_cost(residual,layer.embed.weight,weights))
            costs.append(cost)
            # The framework may install its assignment guard on VQLayer;
            # always use that real hard rule, never a parallel implementation.
            was_training=layer.training
            layer.training=balanced
            try:
                with torch.no_grad():
                    hard_ids=layer._indices(cost,False)
            finally:
                layer.training=was_training
            selected=select(cost.detach() if self.cut_selection_cost else cost,layer.embed.weight,
                            balanced=layer.use_sk,epsilon=layer.sk_epsilon,
                            iterations=layer.sk_iters,training=balanced,hard_ids=hard_ids)
            hard=selected['hard'];q=selected['q']
            parts.append((commitment_loss(residual,hard),codebook_loss(residual,hard)))
            quantized=quantized+q
            hard_sum=hard_sum+hard
            residual=residual-q
            tokens.append(selected['ids'])
            unused+=layer.n_embed-int(torch.unique(selected['ids']).numel())
        self.last_q=quantized
        self.last_hard=hard_sum
        self.last_costs=costs
        self.last_tokens=torch.stack(tokens,dim=1)
        return quantized,parts,unused,self.last_tokens

    def forward(self, embeddings, item_ids=None, *, infer_use_sk=False,
                catalogue_embeddings=None, curvature_centres=None):
        ids=self._identities(embeddings,item_ids)
        _,weights=self.structure.gates_for_items(self.store,ids,embeddings.device,
                                                behavior_on=self.behavior_on)
        self.last_weights=weights
        encoded=self.encoder(embeddings)
        quantized,parts,unused,tokens=self.quantize(encoded,weights,
                                                   balanced=self.training or infer_use_sk)
        reconstructed=self.decoder(quantized)
        result=(reconstructed,parts,unused,tokens)
        if catalogue_embeddings is not None:
            global_q=self.catalogue_q(catalogue_embeddings)
            curvature=(global_q.sum()*0 if curvature_centres is not None and len(curvature_centres)==0
                       else self.curvature_loss(global_q,curvature_centres))
            return (*result,curvature)
        return result

    @torch.no_grad()
    def get_indices(self, embeddings, *, infer_use_sk=False, item_ids=None):
        ids=self._identities(embeddings,item_ids)
        catalogue=torch.as_tensor(np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy'),
                                  device=embeddings.device,dtype=embeddings.dtype)
        if not torch.equal(embeddings,catalogue[ids]):
            raise ValueError('export embeddings must match their global catalogue IDs')
        was_training=self.training
        self.eval()
        try:
            self.catalogue_q(catalogue,infer_use_sk=infer_use_sk)
            return self.last_tokens[ids]
        finally:
            self.train(was_training)

    def catalogue_q(self, embeddings, chunk=32, *, infer_use_sk=False):
        """One global quantization; chunk only structural messages, never assignment."""
        def weight_forward(batch, ids):
            return self.structure.gates_for_items(self.store,ids,batch.device)[1]
        weights=[]
        for start in range(0,len(embeddings),chunk):
            ids=torch.arange(start,min(start+chunk,len(embeddings)),device=embeddings.device)
            if torch.is_grad_enabled():
                weight=checkpoint(weight_forward,embeddings[ids],ids,use_reentrant=False)
            else:
                weight=weight_forward(embeddings[ids],ids)
            weights.append(weight)
        encoded=self.encoder(embeddings)
        q,_,_,_=self.quantize(encoded,torch.cat(weights),balanced=infer_use_sk)
        return q

    def curvature_loss(self, catalogue_q, centres=None):
        return self.targets.loss(catalogue_q,
            lambda x,y:fixed_squared(x,y,scale=self.tangent_scale),centres=centres,
            paired_squared=lambda x,y:fixed_squared(x[:,None,:],y[:,None,:],
                                                    scale=self.tangent_scale).reshape(-1))

    @torch.no_grad()
    def freeze_tangent_scale(self, catalogue_embeddings):
        z=self.encoder(catalogue_embeddings)
        scale=z.square().sum(-1).mean().sqrt()
        if not bool(torch.isfinite(scale)) or float(scale)<=0:
            raise ValueError('nonfinite or zero frozen tangent scale')
        self.tangent_scale.copy_(scale)

    def calibrate_curvature_weight(self, embeddings, item_ids, catalogue_embeddings, centres):
        """One initial, real-loss encoder gradient ratio; no optimizer step."""
        reconstructed,parts,_,_=self(embeddings,item_ids)
        _,reconstruction=self.compute_loss(embeddings,reconstructed,parts)
        q=self.catalogue_q(catalogue_embeddings)
        curvature=self.curvature_loss(q,centres)
        parameters=tuple(self.encoder.parameters())
        rg=torch.autograd.grad(reconstruction,parameters,retain_graph=True,allow_unused=True)
        cg=torch.autograd.grad(curvature,parameters,allow_unused=True)
        rn=torch.stack([g.square().sum() for g in rg if g is not None]).sum().sqrt()
        cn=torch.stack([g.square().sum() for g in cg if g is not None]).sum().sqrt()
        if not bool(torch.isfinite(rn+cn)) or float(cn)<=0 or float(rn)<=0:
            raise RuntimeError('initial real-loss gradient calibration is degenerate')
        self.curvature_weight.copy_((rn/cn).detach())
        return {'reconstruction_gradient':float(rn),'curvature_gradient':float(cn),
                'lambda':float(self.curvature_weight)}

    def combined_loss(self, original, curvature):
        if not bool(torch.isfinite(self.curvature_weight)) or float(self.curvature_weight)<=0:
            raise RuntimeError('gradient-norm calibration must be frozen before training')
        return original+self.curvature_weight*curvature
