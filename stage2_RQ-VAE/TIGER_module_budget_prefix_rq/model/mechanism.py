"""GNN gates enter only codeword costs; fixed curvature never reads gate weights."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from .model import RQVAE as OriginalRQVAE
from .geometry import mixed_cost, fixed_squared
from .selection import select
from .calibration import gradient_norm, mean_gradients, module_norms, norm_budget_weight
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
from .layer_gate import aggregate_layer_losses, cumulative_prefixes
from . import layers as layer_module
from .provenance import binding,sha256
from tools.issue252.archive_audit import imported_repo_sources
from tools.issue252.host_transfer import host_int64
# The estimator reaches repository modules, so it is imported only after the
# repository root is on sys.path: the candidate must import from the launcher's
# result directory without any PYTHONPATH help.
from .estimator import StructuralSampler, sampled_values


class RQVAE(OriginalRQVAE):
    def __init__(self, config, *, in_dim, distance=None):
        super().__init__(config,in_dim=in_dim,distance=distance)
        self.structure = TopologyAttention()
        self.store = EgoStore(ROOT,ROOT/'assets')
        self.targets = FrozenCurvature(ROOT/'assets')
        self.register_buffer('tangent_scale',torch.tensor(1.))
        self.register_buffer('curvature_weight',torch.tensor(float('nan')))
        self.register_buffer('target_scale',torch.tensor(self.targets.scale))
        # Per-level distance scales, frozen before training from the initial
        # cumulative prefixes so an early level is not penalised merely because
        # its prefix representation has a different magnitude.
        self.register_buffer('level_scales',torch.ones(3))
        self._level_scales=(1.,1.,1.)
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
        tokens=[];parts=[];costs=[];level_hard=[]
        # The unused-code count stays on the device: reading it back would
        # synchronise every layer of every step.
        unused=torch.zeros((),dtype=torch.int64,device=residual.device)
        last_level=len(self.rq.vq_layers)-1
        for level,layer in enumerate(self.rq.vq_layers):
            # Only the last level keeps the Sinkhorn-balanced assignment; the
            # earlier levels take the floor-constrained nearest codeword.
            use_sinkhorn=bool(layer.use_sk) and level==last_level
            # Every level owns its own expert mixture; the weights are
            # normalised over the experts for that level only.
            level_weights=weights[:,level,:] if weights.ndim==3 else weights
            cost=self.assignment_cost(residual,layer.embed.weight,level_weights)
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
            was_use_sk=layer.use_sk
            layer.training=balanced and use_sinkhorn
            layer.use_sk=use_sinkhorn
            try:
                with torch.no_grad():
                    hard_ids=layer._indices(cost,False)
            finally:
                layer.training=was_training
                layer.use_sk=was_use_sk
            if guarded:
                layer_module.verify_assignment(hard_ids,layer.n_embed,layer_module.ASSIGNMENT_FLOOR)
            selected=select(cost.detach() if self.cut_selection_cost else cost,layer.embed.weight,
                            balanced=use_sinkhorn,epsilon=layer.sk_epsilon,
                            iterations=layer.sk_iters,training=balanced,hard_ids=hard_ids)
            hard=selected['hard'];q=selected['q']
            parts.append((commitment_loss(residual,hard),codebook_loss(residual,hard)))
            quantized=quantized+q
            hard_sum=hard_sum+hard
            # The cumulative prefix is built from the selected value whose
            # FORWARD is bitwise the hard codeword and whose BACKWARD is the
            # same-cost soft surrogate, exactly like the quantised path.
            level_hard.append(q)
            residual=residual-q
            tokens.append(selected['ids'])
            used=torch.zeros(layer.n_embed,dtype=torch.bool,device=selected['ids'].device)
            used.index_fill_(0,selected['ids'],True)
            unused=unused+(layer.n_embed-used.sum())
        self.last_q=quantized
        self.last_hard=hard_sum
        self.last_prefixes=cumulative_prefixes(level_hard)
        self.last_costs=costs
        self.last_tokens=torch.stack(tokens,dim=1)
        return quantized,parts,unused,self.last_tokens

    def forward(self, embeddings, item_ids=None, *, infer_use_sk=False,
                catalogue_embeddings=None, curvature_centres=None, structural_sample=None):
        ids=self._identities(embeddings,item_ids)
        auxiliary_weights=None
        if catalogue_embeddings is not None and structural_sample is not None:
            # IDs are static facts. Learned tokens/gates are fresh for this call,
            # then gathered into the two UNCHANGED quantization populations.
            auxiliary_ids=np.asarray(structural_sample['ids'],dtype=np.int64)
            if auxiliary_ids.ndim != 1:
                raise ValueError('auxiliary item identities must be one-dimensional')
            union,restore=np.unique(np.concatenate((ids,auxiliary_ids)),return_inverse=True)
            _,shared=self.structure.gates_for_items(self.store,union,embeddings.device,
                                                    behavior_on=self.behavior_on)
            restore=host_int64(restore,shared.device)
            weights=shared[restore[:len(ids)]]
            auxiliary=restore[len(ids):]
            auxiliary_weights=shared[auxiliary]
            # Preserve the old diagnostics: reconstruction weights, but the
            # last topology-token/attention view belongs to the auxiliary call.
            self.structure.last_tokens=self.structure.last_tokens[auxiliary]
            self.structure.last_attention=self.structure.last_attention[auxiliary]
            self.structure.last_token_mask=self.structure.last_token_mask[auxiliary]
        else:
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
                curvature=self.sampled_curvature(catalogue_embeddings,structural_sample,
                                                weights=auxiliary_weights)
            else:
                global_q=self.catalogue_q(catalogue_embeddings)
                curvature=(global_q.sum()*0 if curvature_centres is not None and len(curvature_centres)==0
                           else self.curvature_loss(self.last_prefixes,curvature_centres))
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

    def curvature_loss(self, prefixes, centres=None):
        """The layer-wise sectional proxy: one K_l per level, equal alpha weights.

        ``prefixes`` is the cumulative hard prefix table ``[levels, N, 32]``.  Each
        level uses its own frozen distance scale, so an early level is not
        penalised merely because its prefix representation has a different
        magnitude; the target and the Huber reduction stay the frozen ones.
        """
        if prefixes.ndim != 3 or prefixes.shape[0] != len(self._level_scales):
            raise ValueError('prefix table must be [levels, items, dim]')
        values=[]
        for level in range(prefixes.shape[0]):
            level_scale=self._tangent_scale_value*float(self._level_scales[level])
            values.append(self.targets.loss(prefixes[level],
                lambda x,y,s=level_scale: fixed_squared(x,y,scale=s),
                centres=centres,
                paired_squared=lambda x,y,s=level_scale: fixed_squared(x[:,None,:],y[:,None,:],
                                                                         scale=s).reshape(-1)))
        return aggregate_layer_losses(values)

    def structural_sampler(self):
        if self._structural_sampler is None:
            root=ROOT/'assets'/'v8_runtime_layout'
            table=np.load(root/'targets.npy',mmap_mode='r',allow_pickle=False)
            offsets=np.load(root/'target_offsets.npy',mmap_mode='r',allow_pickle=False)
            self._structural_sampler=StructuralSampler(table,offsets,self.targets.components)
        return self._structural_sampler

    def sampled_curvature(self, embeddings, sample, *, weights=None):
        """Fresh actual hard-q, using the declared finite auxiliary population."""
        host=np.asarray(sample['ids'],dtype=np.int64)
        ids=host_int64(host,embeddings.device)
        if weights is None:
            _,weights=self.structure.gates_for_items(self.store,host,embeddings.device,
                                                    behavior_on=self.behavior_on)
        q,_,_,_=self.quantize(self.encoder(embeddings[ids]),weights,balanced=False)
        self.last_structural_q=q
        self.last_structural_ids=ids
        self.last_structural_tokens=self.last_tokens
        scale=self._tangent_scale_value
        prefixes=self.last_prefixes
        target=torch.as_tensor(sample['target'],device=q.device,dtype=q.dtype)
        values=[]
        for level in range(prefixes.shape[0]):
            level_scale=scale*float(self._level_scales[level])
            predicted=sampled_values(prefixes[level],sample['triples'],sample['references'],
                lambda x,y,s=level_scale: fixed_squared(x,y,scale=s),
                paired_squared=lambda x,y,s=level_scale: fixed_squared(x[:,None,:],y[:,None,:],
                                                                         scale=s).reshape(-1))
            self.last_structural_prediction=predicted
            values.append(F.huber_loss(predicted/self._target_scale_value,
                                       target/self._target_scale_value,
                                       reduction='mean',delta=1.))
        return aggregate_layer_losses(values)

    @torch.no_grad()
    def freeze_tangent_scale(self, catalogue_embeddings):
        z=self.encoder(catalogue_embeddings)
        scale=z.square().sum(-1).mean().sqrt()
        if not bool(torch.isfinite(scale)) or float(scale)<=0:
            raise ValueError('nonfinite or zero frozen tangent scale')
        self.tangent_scale.copy_(scale)
        self._tangent_scale_value = float(scale)

    @torch.no_grad()
    def freeze_level_scales(self, catalogue_embeddings, structural_sample=None):
        """Freeze one distance scale per level from the initial cumulative prefixes.

        The scale is the RMS tangent-scaled distance between the initial
        prefixes of the sampled centres and their references.  It is computed
        once, before training, from the training-side population only.
        """
        self.catalogue_q(catalogue_embeddings)
        prefixes=self.last_prefixes
        if structural_sample is None:
            raise ValueError('a training-side structural sample is required')
        centres=np.asarray(structural_sample['triples_global'],dtype=np.int64)[:,0]
        references=np.asarray(structural_sample['references_global'],dtype=np.int64)
        centre_ids=host_int64(centres,prefixes.device)
        reference_ids=host_int64(references,prefixes.device)
        scales=[]
        for level in range(prefixes.shape[0]):
            distances=fixed_squared(prefixes[level][centre_ids],prefixes[level][reference_ids],
                                     scale=self._tangent_scale_value)
            value=float(distances.square().mean().sqrt())
            if not np.isfinite(value) or value<=0:
                raise ValueError('invalid initial per-level distance scale')
            scales.append(value)
        self.level_scales.copy_(torch.tensor(scales,dtype=self.level_scales.dtype,
                                             device=self.level_scales.device))
        self._level_scales=tuple(scales)
        return {'level_scales':list(scales),'tangent_scale':float(self._tangent_scale_value),
                'samples':int(len(centres)),'references':int(len(references))}

    def calibrate_curvature_weight(self, embeddings, item_ids, catalogue_embeddings, centres=None,
                                   *, structural_sample=None):
        """Freeze the maximal global module budget on the actual first batch."""
        previous=self.rank_local_checks
        self.rank_local_checks=True
        try:
            reconstructed,parts,_,_,curvature=self(
                embeddings,item_ids,catalogue_embeddings=catalogue_embeddings,
                curvature_centres=centres,structural_sample=structural_sample)
        finally:
            self.rank_local_checks=previous
        original,reconstruction=self.compute_loss(embeddings,reconstructed,parts)
        parameters=tuple(p for name,p in self.named_parameters() if not name.startswith('decoder.'))
        encoder=tuple(self.encoder.parameters())
        og=torch.autograd.grad(original,parameters,retain_graph=True)
        rg=torch.autograd.grad(reconstruction,encoder,retain_graph=True)
        cg=torch.autograd.grad(curvature,parameters)
        for gradients in (og,rg,cg):
            mean_gradients(gradients)
        originals=dict(zip(map(id,parameters),og))
        structurals=dict(zip(map(id,parameters),cg))
        onorms=module_norms(self,originals)
        cnorms=module_norms(self,structurals)
        budgets={name:(onorms[name],cnorms[name]) for name in onorms}
        weight=norm_budget_weight(budgets)
        self.curvature_weight.fill_(weight)
        weight=float(self.curvature_weight)
        if not np.isfinite(weight) or weight<=0:
            raise RuntimeError('calibrated coefficient is not representable as positive float32')
        on,cn=budgets['encoder']
        rn=gradient_norm(rg)
        original_aligned=sum(float((originals[id(p)].double()*structurals[id(p)].double()).sum())
                             for p in encoder)
        reconstruction_aligned=sum(float((a.double()*structurals[id(p)].double()).sum())
                                   for p,a in zip(encoder,rg))
        return {'original_gradient':on,'reconstruction_gradient':rn,'curvature_gradient':cn,
                'original_curvature_cosine':original_aligned/(on*cn) if on*cn>0 else None,
                'reconstruction_curvature_cosine':reconstruction_aligned/(rn*cn) if rn*cn>0 else None,
                'lambda':weight,'calibration_batch_items':len(embeddings),
                'calibration_world_size':(torch.distributed.get_world_size()
                    if torch.distributed.is_initialized() else 1),
                'module_budgets':{name:{'original_gradient':a,'curvature_gradient':b,
                    'weighted_ratio':weight*b/a if a>0 else 0.}
                    for name,(a,b) in budgets.items()}}

    def combined_loss(self, original, curvature):
        if not bool(torch.isfinite(self.curvature_weight)) or float(self.curvature_weight)<=0:
            raise RuntimeError('gradient-norm calibration must be frozen before training')
        return original+self.curvature_weight*curvature

    def provenance(self, *, expected=None):
        result=binding(ROOT,Path(__file__).resolve().parent.parent)
        result['mathematical_contract_path']='results/research/issue256/rounds/r01/CONTRACT.md'
        result['mathematical_contract_sha256']=sha256(ROOT/'results/research/issue256/rounds/r01/CONTRACT.md')
        result['infrastructure_sources']=imported_repo_sources(
            Path(__file__).resolve().parent.parent,ROOT)
        if expected is not None:
            if result!=expected:
                raise ValueError('source/assets/config changed during execution')
            return expected
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # Restores are cold-path operations; the training hot path uses these
        # host mirrors without synchronizing the device for every distance.
        self._tangent_scale_value = float(self.tangent_scale)
        self._target_scale_value = float(self.target_scale)
        self._level_scales = tuple(float(value) for value in self.level_scales)
        return result

    def load_checkpoint(self,path):
        blob=torch.load(path,map_location=self.tangent_scale.device,weights_only=False)
        if blob.get('provenance')!=self.provenance():
            raise ValueError('checkpoint source/asset/config provenance mismatch')
        self.load_state_dict(blob['state_dict'],strict=True)
        values=torch.stack((self.tangent_scale,self.curvature_weight,self.target_scale))
        if not bool(torch.isfinite(values).all()) or not bool((values>0).all()):
            raise ValueError('checkpoint has missing/invalid frozen calibration')
        return blob
