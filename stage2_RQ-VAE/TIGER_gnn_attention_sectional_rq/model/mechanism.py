"""GNN gates enter only codeword costs; fixed curvature never reads gate weights."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from .model import RQVAE as OriginalRQVAE
from .geometry import mixed_cost, fixed_squared
from .selection import select
from .estimator import StructuralSampler, sampled_values
from losses.commitment import commitment_loss
from losses.codebook import codebook_loss


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if ((parent / 'stage2_RQ-VAE').is_dir()
                and (parent / 'tools/research/stage2/contract.py').is_file()):
            return parent
    raise RuntimeError('missing repository marker')


ROOT = repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.graph_infra.ego_gnn import EgoStore
from .structure_gate import TopologyAttention
from .curvature import FrozenCurvature
from . import layers as layer_module
from .provenance import binding,sha256
from tools.issue252.archive_audit import imported_repo_sources
from tools.issue252.host_transfer import host_int64


class RQVAE(OriginalRQVAE):
    def __init__(self, config, *, in_dim, distance=None):
        super().__init__(config,in_dim=in_dim,distance=distance)
        self.structure = TopologyAttention()
        self.store = EgoStore(ROOT,ROOT/'assets')
        self.targets = FrozenCurvature(ROOT/'assets')
        self.register_buffer('tangent_scale',torch.tensor(1.))
        self.register_buffer('curvature_weight',torch.tensor(float('nan')))
        self.register_buffer('target_scale',torch.tensor(self.targets.scale))
        # Host mirrors of the frozen scales: the geometry helpers take the
        # scale as a number, and reading a device buffer for it would stall
        # the step on every cost.
        self._tangent_scale_value = 1.0
        self._target_scale_value = float(self.target_scale)
        self.behavior_on = True
        self.curvature_on = True
        self.cut_selection_cost = False
        self.rank_local_checks = False
        self.last_q = None
        self.last_hard = None
        self.last_costs = []
        self.last_weights = None
        self._structural_sampler = None
        self._catalogue_cache = None

    def _identities(self, embeddings, item_ids):
        """Validate item identity on the host.

        The structure gates need the identity order and the ego sizes, both of
        which are host-side facts; keeping them there lets every later gather
        state its output size instead of reading one back from the device.
        """
        if item_ids is None:
            if len(embeddings) != self.store.n:
                raise ValueError('item_ids required outside an ordered full-catalogue forward')
            host = np.arange(self.store.n, dtype=np.int64)
        elif torch.is_tensor(item_ids):
            host = item_ids.detach().cpu().numpy()
        else:
            host = np.asarray(item_ids)
        host = np.asarray(host, dtype=np.int64).reshape(-1)
        if host.shape != (len(embeddings),) or ((host < 0) | (host >= self.store.n)).any():
            raise ValueError('item identity outside the global catalogue')
        return host

    def assignment_cost(self, residual, codewords, weights):
        return mixed_cost(residual,codewords,weights,scale=self._tangent_scale_value,
                          curvature_on=self.curvature_on)

    def quantize(self, encoded, weights, *, balanced):
        residual=encoded
        quantized=torch.zeros_like(encoded)
        hard_sum=torch.zeros_like(encoded)
        tokens=[];parts=[];costs=[];unused=0
        for layer in self.rq.vq_layers:
            cost=self.assignment_cost(residual,layer.embed.weight,weights)
            costs.append(cost)
            guarded=hasattr(layer_module,'ASSIGNMENT_FLOOR')
            rank_local=not balanced or self.rank_local_checks
            if guarded:
                layer_module.verify_codebook(layer.embed.weight,distance=False)
                layer_module.verify_separation(layer.embed.weight,cost,key=id(layer),rank_local=rank_local)
                layer_module.verify_margins(cost,rank_local=rank_local)
            # The framework may install its assignment guard on VQLayer;
            # always use that real hard rule, never a parallel implementation.
            was_training=layer.training
            layer.training=balanced
            try:
                with torch.no_grad():
                    hard_ids=layer._indices(cost,False)
            finally:
                layer.training=was_training
            if guarded:
                layer_module.verify_assignment(hard_ids,layer.n_embed,layer_module.ASSIGNMENT_FLOOR)
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
                catalogue_embeddings=None, curvature_centres=None, structural_sample=None):
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
            if structural_sample is not None:
                curvature=self.sampled_curvature(catalogue_embeddings,structural_sample)
            else:
                global_q=self.catalogue_q(catalogue_embeddings)
                curvature=(global_q.sum()*0 if curvature_centres is not None and len(curvature_centres)==0
                           else self.curvature_loss(global_q,curvature_centres))
            return (*result,curvature)
        return result

    @torch.no_grad()
    def get_indices(self, embeddings, *, infer_use_sk=False, item_ids=None):
        ids=self._identities(embeddings,item_ids)
        index=host_int64(ids,embeddings.device)
        key=(embeddings.device,embeddings.dtype)
        if self._catalogue_cache is None or self._catalogue_cache[0]!=key:
            catalogue=torch.as_tensor(np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy'),
                                      device=embeddings.device,dtype=embeddings.dtype)
            self._catalogue_cache=(key,catalogue)
        catalogue=self._catalogue_cache[1]
        if not torch.equal(embeddings,catalogue[index]):
            raise ValueError('export embeddings must match their global catalogue IDs')
        was_training=self.training
        self.eval()
        try:
            self.catalogue_q(catalogue,infer_use_sk=infer_use_sk)
            return self.last_tokens[index]
        finally:
            self.train(was_training)

    def catalogue_q(self, embeddings, chunk=1024, *, infer_use_sk=False):
        """Complete global quantization; structural chunks never split assignment."""
        ids=np.arange(len(embeddings),dtype=np.int64)
        _,weights=self.structure.gates_for_items(self.store,ids,embeddings.device,
                                                chunk=chunk,behavior_on=self.behavior_on)
        encoded=self.encoder(embeddings)
        self.last_weights=weights
        self.catalogue_attention=self.structure.last_attention.detach()
        self.last_token_counts=self.structure.last_token_mask[:,:-1].sum(-1)
        q,_,_,_=self.quantize(encoded,weights,balanced=infer_use_sk)
        return q

    def curvature_loss(self, catalogue_q, centres=None):
        scale=self._tangent_scale_value
        return self.targets.loss(catalogue_q,
            lambda x,y:fixed_squared(x,y,scale=scale),centres=centres,
            paired_squared=lambda x,y:fixed_squared(x[:,None,:],y[:,None,:],
                                                    scale=scale).reshape(-1))

    def structural_sampler(self):
        if self._structural_sampler is None:
            root=ROOT/'assets'/'v8_runtime_layout'
            table=np.load(root/'targets.npy',mmap_mode='r',allow_pickle=False)
            offsets=np.load(root/'target_offsets.npy',mmap_mode='r',allow_pickle=False)
            self._structural_sampler=StructuralSampler(table,offsets,self.targets.components)
        return self._structural_sampler

    def sampled_curvature(self, embeddings, sample):
        """Fresh actual hard-q, using the declared finite auxiliary population."""
        host=np.asarray(sample['ids'],dtype=np.int64)
        ids=host_int64(host,embeddings.device)
        _,weights=self.structure.gates_for_items(self.store,host,embeddings.device,
                                                behavior_on=self.behavior_on)
        q,_,_,_=self.quantize(self.encoder(embeddings[ids]),weights,balanced=False)
        self.last_structural_q=q
        self.last_structural_ids=ids
        self.last_structural_tokens=self.last_tokens
        scale=self._tangent_scale_value
        predicted=sampled_values(q,sample['triples'],sample['references'],
            lambda x,y:fixed_squared(x,y,scale=scale),
            paired_squared=lambda x,y:fixed_squared(x[:,None,:],y[:,None,:],scale=scale).reshape(-1))
        target=torch.as_tensor(sample['target'],device=q.device,dtype=q.dtype)
        self.last_structural_prediction=predicted
        return F.huber_loss(predicted/self._target_scale_value,target/self._target_scale_value,
                           reduction='mean',delta=1.)

    @torch.no_grad()
    def freeze_tangent_scale(self, catalogue_embeddings):
        z=self.encoder(catalogue_embeddings)
        scale=z.square().sum(-1).mean().sqrt()
        if not bool(torch.isfinite(scale)) or float(scale)<=0:
            raise ValueError('nonfinite or zero frozen tangent scale')
        self.tangent_scale.copy_(scale)
        self._tangent_scale_value = float(scale)

    def calibrate_curvature_weight(self, embeddings, item_ids, catalogue_embeddings, centres=None,
                                   *, structural_sample=None):
        """Freeze one initial real-loss encoder gradient ratio; no optimizer step."""
        previous=self.rank_local_checks
        self.rank_local_checks=True
        try:
            reconstructed,parts,_,_=self(embeddings,item_ids)
        finally:
            self.rank_local_checks=previous
        _,reconstruction=self.compute_loss(embeddings,reconstructed,parts)
        if structural_sample is None:
            q=self.catalogue_q(catalogue_embeddings)
            curvature=self.curvature_loss(q,centres)
        else:
            curvature=self.sampled_curvature(catalogue_embeddings,structural_sample)
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

    def provenance(self, *, expected=None):
        result=binding(ROOT,Path(__file__).resolve().parent.parent)
        result['mathematical_contract_sha256']=sha256(ROOT/'results/research/issue252/CONTRACT.md')
        result['infrastructure_sources']=imported_repo_sources(
            Path(__file__).resolve().parent.parent,ROOT)
        if expected is not None:
            if result!=expected:
                raise ValueError('source/assets/config changed during execution')
            return expected
        return result

    def load_checkpoint(self,path):
        blob=torch.load(path,map_location=self.tangent_scale.device,weights_only=False)
        if blob.get('provenance')!=self.provenance():
            raise ValueError('checkpoint source/asset/config provenance mismatch')
        self.load_state_dict(blob['state_dict'],strict=True)
        values=torch.stack((self.tangent_scale,self.curvature_weight,self.target_scale))
        if not bool(torch.isfinite(values).all()) or not bool((values>0).all()):
            raise ValueError('checkpoint has missing/invalid frozen calibration')
        self._tangent_scale_value=float(self.tangent_scale)
        self._target_scale_value=float(self.target_scale)
        return blob
