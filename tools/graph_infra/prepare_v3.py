"""No-argument, CPU-only issue247 build/verify entry. Frozen v2 assets are read-only.

An existing v3 manifest is verified, never silently overwritten. Remove nothing automatically.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, connected_components
from tools.graph_infra import assets, config, graph_build, graphlet, sectional, metrics
from tools.graph_infra.interfaces_v3 import verify, TrainingAssets, HeldOutEvaluation, validate_ids

REPO = config.REPO
ROOT = REPO / 'results/research/issue247'
OUT = ROOT / 'assets'
V2 = config.ASSET_ROOT
SEED = 247001
REFERENCE_SIZES = (16,64,256)
COMMAND = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.graph_infra.prepare_v3'


def dump(path, data):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data,indent=2,default=convert,allow_nan=False)+'\n')


def source_hashes():
    return {str(p.relative_to(REPO)):assets.sha256(p) for p in sorted((REPO/'tools/graph_infra').glob('*.py'))}


def audit_v2():
    frozen = json.loads((V2/'manifest.json').read_text())
    for e in frozen['assets']:
        assert assets.sha256(REPO/e['path']) == e['sha256'], ('v2 asset drift',e['name'])
    graph = assets.load_sparse(V2/'graph/visible_forward.npz')
    users = dict(np.load(V2/'graph/split_users.npz'))
    sequence = graph_build.sequences_from_parquet(config.TRAIN)
    assert set.union(*(set(v.tolist()) for v in users.values())) == set(sequence)
    for a in users:
        for b in users:
            if a < b: assert not set(users[a]) & set(users[b])
    # Independent raw transition counters, not the graph builder's counting implementation.
    rebuilt = {}
    for name, group in users.items():
        count = Counter()
        for user in group:
            s = sequence[int(user)]
            for a,b in zip(s[:-1],s[1:]):
                if a != b: count[int(a),int(b)] += 1
        actual = assets.load_sparse(V2/f'graph/{name}_forward.npz').tocoo()
        assert count == Counter({(int(a),int(b)):int(w) for a,b,w in zip(actual.row,actual.col,actual.data)})
        rebuilt[name] = {'users':len(group),'directed_edges':len(count),'weight':sum(count.values())}
    train_keys = set(); strict_keys = set()
    bank_counts = {}
    for p in sorted((V2/'pairs').glob('*.npz')):
        z = np.load(p)
        if 'pairs' not in z.files:
            # The v2 PPR cache is deliberately ignored: issue #247 forbids computing or using PPR.
            continue
        pairs=z['pairs']; keys={tuple(sorted(x)) for x in pairs.tolist()}
        assert len(pairs) == len(set(map(tuple,pairs.tolist())))
        bank_counts[p.stem] = len(pairs)
        if p.stem.startswith('train_') or p.stem == 'probe': train_keys |= keys
        if p.stem.startswith('strict_'): strict_keys |= keys
    assert not train_keys & strict_keys
    motif = dict(np.load(V2/'motifs/motif_counts.npz'))
    audit = {'v2_manifest_sha256':assets.sha256(V2/'manifest.json'), 'verified_assets':len(frozen['assets']),
             'splits':rebuilt,'banks':bank_counts,'strict_train_intersection':0,
             'actual_probe':bank_counts['probe'],'curvature_files':list(map(str,(V2/'curvature').glob('*'))),
             'motif_instances':int(motif['instances'].sum()), 'corrections':[
                 'Visible users are 45951, not 51695. Three sets: 45951/5744/5744.',
                 'Probe has 62 pairs, not 64. Strict group sizes include 2940, not maximum 2938.',
                 'No v2 curvature files exist. Ricci implementation is not sectional A.1.',
                 'Historical comments disagree with final files; final files and recomputation govern.']}
    dump(ROOT/'AUDIT_246.json',audit)
    return graph,users


def entry(name,path,use,schema,dependencies=None,coverage=None):
    path = Path(path)
    if schema == 'csr':
        z=assets.load_sparse(path); shape=list(z.shape); dtype=str(z.dtype)
    elif path.suffix == '.npz':
        z=np.load(path,allow_pickle=True); shape={k:list(z[k].shape) for k in z.files}; dtype={k:str(z[k].dtype) for k in z.files}
    elif path.suffix == '.npy':
        z=np.load(path,mmap_mode='r'); shape=list(z.shape); dtype=str(z.dtype)
    else: shape=None; dtype='json/text'
    return {'name':name,'path':str(path.relative_to(REPO)),'use':use,'schema':schema,'shape':shape,'dtype':dtype,
            'sha256':assets.sha256(path),'bytes':path.stat().st_size,'dependencies':dependencies or {},
            'coverage':coverage,'generation_command':COMMAND}


def build():
    start=time.monotonic()
    graph,users=audit_v2()
    OUT.mkdir(parents=True,exist_ok=True)
    ids=graph_build.item_index()['ids']; n=len(ids)
    validate_ids(ids,np.arange(n))
    semantic=np.load(config.EMBEDDINGS,mmap_mode='r')
    assert semantic.shape == (n,768) and np.isfinite(semantic).all()
    np.save(OUT/'item_ids.npy',ids)
    coo=graph.tocoo()
    np.savez_compressed(OUT/'directed_edges.npz',source=coo.row,target=coo.col,weight=coo.data,
                        reciprocal=np.asarray(graph[coo.col,coo.row]).ravel()>0)
    undirected=sectional.projection(graph)
    degree=np.diff(undirected.indptr).astype(np.int64)
    cc,labels=connected_components(undirected,directed=False)
    sample=sectional.sample_triples(graph,SEED)
    triples=sample['triples']
    references=np.random.default_rng(SEED+1).permutation(n)[:max(REFERENCE_SIZES)].astype(np.int64)
    ds=time.monotonic()
    distances=shortest_path(undirected,directed=False,unweighted=True,indices=references)
    distance_seconds=time.monotonic()-ds
    # Deterministic BFS outputs regenerated on validate, not another historical cache.
    np.save(OUT/'reference_distances.npy',distances)
    sensitivity=[]; final=None
    for size in REFERENCE_SIZES:
        s=time.monotonic(); result=sectional.estimate(graph,triples,references[:size],distances[:size])
        if final is not None:
            mask=final['valid'] & result['valid']
            delta=np.abs(result['value'][mask]-final['value'][mask])
            change={'common_valid':int(mask.sum()),'mean_absolute_change':float(delta.mean()) if len(delta) else None,
                    'max_absolute_change':float(delta.max()) if len(delta) else None,
                    'sign_change_fraction':float(np.mean(np.sign(result['value'][mask])!=np.sign(final['value'][mask]))) if len(delta) else None}
        else: change=None
        sensitivity.append({'references':size,'valid_triples':int(result['valid'].sum()),
                            'effective_reference_min':int(result['effective_references'].min()),
                            'seconds':time.monotonic()-s,'change_from_previous':change})
        final=result
    m,b,c=triples.T
    # Graphlet canonical ID includes directed edges; curvature itself remains undirected.
    codes=np.zeros(len(triples),dtype=np.int64)
    for left,right in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]:
        codes=(codes<<1)|np.asarray(graph[triples[:,left],triples[:,right]]>0).ravel().astype(np.int64)
    classes=graphlet.canonical_lookup()[codes]
    np.savez_compressed(OUT/'sectional_triples.npz',triples=triples,references=references,
                        value=final['value'],effective_references=final['effective_references'],valid=final['valid'],
                        invalid_reason=final['reason'],endpoint_distance=final['endpoint_distance'],
                        directed_graphlet=classes,neighbor_pair_population=sample['population'],
                        reference_contributions=final['contributions'])
    summary=sectional.summarize(n,triples,final['value'])
    np.savez_compressed(OUT/'sectional_summary.npz',**summary)
    # All items remain represented; orbit participation explicitly refers to sampled instances.
    motif=np.load(V2/'motifs/motif_counts.npz')
    counts=motif['counts']; denom=counts.sum(axis=1,keepdims=True)
    orbit3=np.load(V2/'motifs/motif_instances_visible.npz')
    orbit_keys=np.unique(np.column_stack((orbit3['graphlets'],orbit3['orbits'])),axis=0)
    orbit_map={tuple(k):i for i,k in enumerate(orbit_keys.tolist())}
    orbit_counts=np.zeros((n,len(orbit_keys)),dtype=np.int64)
    columns=np.array([orbit_map[tuple(k)] for k in np.column_stack((orbit3['graphlets'],orbit3['orbits'])).tolist()])
    np.add.at(orbit_counts,(orbit3['centers'],columns),1)
    od=orbit_counts.sum(axis=1,keepdims=True)
    np.savez_compressed(OUT/'topology_features.npz',degree_out=np.diff(graph.indptr),degree_in=np.diff(graph.T.tocsr().indptr),
                        reciprocal_degree=np.asarray(graph.multiply(graph.T).getnnz(axis=1)),
                        family_counts=counts,family_distribution=np.divide(counts,denom,out=np.zeros_like(counts,dtype=float),where=denom>0),
                        family_valid=denom[:,0]>0,sampled_orbit_counts=orbit_counts,
                        sampled_orbit_distribution=np.divide(orbit_counts,od,out=np.zeros_like(orbit_counts,dtype=float),where=od>0),
                        sampled_orbit_valid=od[:,0]>0)
    np.savez_compressed(OUT/'orbit_schema.npz',graphlet_orbit_keys=orbit_keys)
    # Sparse full directed reachability within two hops, plus explicit exact-hop separation.
    support=(graph>0).astype(np.int64)
    out2=((support@support)>0); out2.setdiag(False); out2.eliminate_zeros()
    assets.save_sparse(OUT/'directed_out_within2.npz',out2.maximum(support).astype(np.int8))
    back=support.T.tocsr()
    in2=((back@back)>0); in2.setdiag(False); in2.eliminate_zeros()
    assets.save_sparse(OUT/'directed_in_within2.npz',in2.maximum(back).astype(np.int8))
    four=np.load(V2/'motifs/motif_instances_4node_visible.npz')
    four_counts=np.bincount(four['centers'],minlength=n)
    four_population=degree*(degree-1)*(degree-2)//6
    coverage={'items':n,'components':int(cc),'isolated':int(np.sum(degree==0)),
              'triple_centers':len(triples),'triple_center_fraction':len(triples)/n,
              'eligible_pair_population':int(sample['population'].sum()),
              'sampled_pair_fraction':len(triples)/int(sample['population'].sum()),
              'valid_curvature_item_fraction':float(summary['valid'].mean()),
              'reference_sampling':'uniform without replacement global permutation, seed 247002, nested 16/64/256',
              'triple_sampling':'one uniform unordered neighbor pair per eligible center, seed 247001',
              'disconnected_estimand':'finite references in same component; no imputation; small components may have no references',
              'four_node':{'instances':len(four['centers']),'centers':int(np.count_nonzero(four_counts)),
                           'center_coverage':float(np.mean(four_counts>0)),
                           'neighbor_triple_population':int(four_population.sum()),
                           'instance_fraction':len(four['centers'])/int(four_population.sum()),
                           'sampling':'v2: 500 centers; first 12 sorted neighbors; at most 10 triples per center; ID-biased, not population uniform',
                           'uncertainty':'NA: selection has zero inclusion probability outside first12; no unbiased population estimate or confidence interval justified'},
              'orbit_scope':'center orbit participation in sampled 3-node instances, not full exact orbit totals',
              'graph_build_rules':{
                  'source':'results/stage0_build_parquet/train.parquet: one sequence per user = longest history plus its target',
                  'edges':'adjacent transitions (i -> i+1) inside each user sequence; self-loops dropped',
                  'weight':'number of users contributing the transition',
                  'visible_users':'frozen v2 split (seed 246001, ratios 0.8/0.1/0.1): 45951/5744/5744, pairwise disjoint',
                  'undirected_projection':'visible directed support symmetrised, unweighted, loop-free',
                  'curvature_inputs':'visible graph only; struct_val/struct_test edges and full-description graph never read',
                  'ppr':'not computed, not read, not referenced'},
              'ego':'exact induced radius1/2 on demand from frozen visible CSR; sampled v2 separately retained',
              'distance_seconds':distance_seconds,'sensitivity':sensitivity,'zero_tolerance':sectional.ZERO_TOLERANCE}
    dump(OUT/'coverage.json',coverage)
    # Freeze actual Stage3 SID separately from checkpoint and NPY identities.
    sidpath=REPO/'results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json'
    table=json.loads(sidpath.read_text()); assert set(table)==set(map(str,ids.tolist()))
    sid=np.asarray([table[str(i)] for i in ids],dtype=np.int64)
    assert sid.shape==(n,4) and np.all((sid[:,:3]>=0)&(sid[:,:3]<256))
    prefixes={}
    for bank in sorted((V2/'pairs').glob('*.npz')):
        blob=np.load(bank)
        if 'pairs' not in blob.files:
            continue
        pairs=blob['pairs']
        prefixes[bank.stem]={str(k):float(np.mean(np.all(sid[pairs[:,0],:k]==sid[pairs[:,1],:k],axis=1))) for k in (1,2,3)}
    provenance=json.loads((ROOT.parent/'infra246/tiger_audit.json').read_text())
    bindings=[]
    sidsha=assets.sha256(sidpath)
    for log in sorted((REPO/'results/stage3_T5Train/TIGER_RQ-VAE').rglob('HG_Rec.log')):
        text=log.read_text(errors='replace')
        if sidsha in text:
            bindings.append({'path':str(log.relative_to(REPO)),'sha256':assets.sha256(log),'sid_sha256':sidsha})
    # Configuration path proves file selection, but absent historical hash cannot prove old contents.
    baseline={'sid_path':str(sidpath.relative_to(REPO)),'sid_sha256':sidsha,
              'structure':metrics.sid_structure(sid),'codebooks':metrics.codebook_stats(sid[:,:3],unit='bits'),
              'prefix_by_structure_bank':prefixes,'hash_proven_stage3_bindings':bindings,
              'historical_path_evidence':provenance['stage3_binding'],
              'recall':None,'recall_reason':'NA: no immutable historical SID hash binding to completed recall established',
              'reconstruction_mse':None,'reconstruction_reason':'NA: frozen checkpoint belongs to another SID generation',
              'separate_checkpoint':provenance['reexport']['checkpoint'],
              'separate_npy_sha256':provenance['legacy']['sids_for_hgrec.npy']['sha256']}
    dump(OUT/'tiger.json',baseline)
    entries=[]
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            schema='csr' if path.name.startswith('directed_') and path.suffix=='.npz' and path.name!='directed_edges.npz' else path.suffix[1:]
            use='description' if path.name=='tiger.json' else 'training'
            entries.append(entry(path.stem,path,use,schema))
    for name,rel,use,schema in [
        ('visible_forward','graph/visible_forward.npz','training','csr'),
        ('visible_reverse','graph/visible_reverse.npz','training','csr'),
        ('split_users','graph/split_users.npz','evaluation','npz'),
        ('held_out_support','graph/held_out_support.npz','evaluation','csr'),
        ('ego_full_1hop','ego/ego_full_1hop_visible.npz','training','npz'),
        ('ego_sampled','ego/ego_sampled_visible.npz','training','npz'),
        ('motif_counts','motifs/motif_counts.npz','training','npz'),
        ('motif_instances3','motifs/motif_instances_visible.npz','training','npz'),
        ('motif_instances4','motifs/motif_instances_4node_visible.npz','training','npz')]:
        entries.append(entry(name,V2/rel,use,schema))
    entries += [entry('semantic',config.EMBEDDINGS,'training','npy'), entry('items_source',config.ITEMS,'training','parquet'),
                entry('train_source',config.TRAIN,'evaluation','parquet'), entry('tiger_sid',sidpath,'description','json')]
    for bank in sorted((V2/'pairs').glob('*.npz')):
        if 'pairs' not in np.load(bank).files:
            continue
        entries.append(entry(bank.stem,bank,'training' if bank.stem.startswith('train_') else 'evaluation','npz'))
    # Exact immutable dependencies and versioned source inventory shared by all entries.
    sources={str(p.relative_to(REPO)):assets.sha256(p) for p in (config.ITEMS,config.TRAIN,config.EMBEDDINGS,V2/'manifest.json')}
    manifest={'version':'v3-sectional-247','issue':247,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              'official_commit':'1948d3c5f19f7251cadd8d1118c06f0e294f96b3','source_hashes':source_hashes(),
              'inputs':sources,'seed':SEED,'reference_sizes':REFERENCE_SIZES,'graph_sha256':assets.sha256(V2/'graph/visible_forward.npz'),
              'command':COMMAND,'runtime_seconds':time.monotonic()-start,'assets':entries,
              'storage_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()),
              'reused_bytes':sum(e['bytes'] for e in entries if not e['path'].startswith(str(OUT.relative_to(REPO))))}
    dump(OUT/'manifest.json',manifest)
    print('Built v3 assets',flush=True)


def validate():
    start=time.monotonic()
    manifest=verify(OUT/'manifest.json',REPO)
    assert manifest['source_hashes'] == source_hashes(), 'production source drift'
    for path,digest in manifest['inputs'].items(): assert assets.sha256(REPO/path)==digest
    train=TrainingAssets(REPO)
    n=len(train.ids)
    g=train.forward; undirected=sectional.projection(g)
    triples=train.load('sectional_triples'); summary=train.load('sectional_summary')
    again=sectional.sample_triples(g,SEED)
    np.testing.assert_array_equal(again['triples'],triples['triples'])
    references=np.random.default_rng(SEED+1).permutation(n)[:256]
    np.testing.assert_array_equal(references,triples['references'])
    distances=shortest_path(undirected,directed=False,unweighted=True,indices=references)
    np.testing.assert_array_equal(distances,train.load('reference_distances'))
    recalculated=sectional.estimate(g,triples['triples'],references,distances)
    np.testing.assert_allclose(recalculated['value'],triples['value'],atol=1e-12,equal_nan=True)
    for k,v in sectional.summarize(n,triples['triples'],recalculated['value']).items():
        np.testing.assert_allclose(v,summary[k],equal_nan=True)
    assert np.isfinite(triples['value'][triples['valid']]).all()
    assert np.isnan(triples['value'][~triples['valid']]).all()
    topology=train.load('topology_features')
    assert all(np.isfinite(v).all() for v in topology.values())
    aligned=train.aligned(np.arange(n))
    assert aligned['semantic'].shape==(n,768)
    # Exhaustive radius1 and radius2 reachability membership; exact induced ego loader on hand/real centers.
    out2=train.load('directed_out_within2'); in2=train.load('directed_in_within2')
    support=(g>0).astype(np.int64); back=support.T.tocsr()
    two_out=((support@support)>0); two_out.setdiag(False); two_out.eliminate_zeros()
    two_in=((back@back)>0); two_in.setdiag(False); two_in.eliminate_zeros()
    np.testing.assert_array_equal(out2.toarray(),two_out.maximum(support).toarray())
    np.testing.assert_array_equal(in2.toarray(),two_in.maximum(back).toarray())
    rng=np.random.default_rng(247003); sample=rng.choice(n,2000,replace=False)
    for center in sample.tolist():
        one=set(g.indices[g.indptr[center]:g.indptr[center+1]])
        two=set(one)
        for node in one: two.update(g.indices[g.indptr[node]:g.indptr[node+1]])
        two.discard(center)
        assert two==set(out2.indices[out2.indptr[center]:out2.indptr[center+1]])
        rev=set(g.T.tocsr().indices[g.T.tocsr().indptr[center]:g.T.tocsr().indptr[center+1]])
        reach=set(rev)
        for node in rev: reach.update(g.T.tocsr().indices[g.T.tocsr().indptr[node]:g.T.tocsr().indptr[node+1]])
        reach.discard(center)
        assert reach==set(in2.indices[in2.indptr[center]:in2.indptr[center+1]])
    for center in (0,n//2,n-1):
        for radius in (1,2):
            ego=train.ego(center,radius)
            nodes=ego['nodes']; edges=ego['edges']
            assert nodes[ego['root_local']]==center and len(set(nodes))==len(nodes)
            sub=g[nodes][:,nodes].tocoo()
            assert set(map(tuple,edges.tolist()))==set(zip(sub.row.tolist(),sub.col.tolist()))
            assert ego['node_mask'].all() and ego['edge_mask'].all()
    evaluate=HeldOutEvaluation(REPO)
    strict=set(); training=set()
    for name,e in train.entries.items():
        if name.startswith('strict_'):
            pairs=evaluate.load(name)['pairs']; strict.update(tuple(sorted(x)) for x in pairs.tolist())
            try: train.load(name)
            except ValueError: pass
            else: raise AssertionError('training target leakage')
        elif name.startswith('train_') and name!='train_source':
            pairs=train.load(name)['pairs']; training.update(tuple(sorted(x)) for x in pairs.tolist())
    assert not strict & training
    # Independent tiny-graph numerical comparator runs in tests; full item metrics recomputed here.
    tiger=json.loads((OUT/'tiger.json').read_text())
    table=json.loads((REPO/tiger['sid_path']).read_text()); sid=np.array([table[str(i)] for i in range(n)])
    distinct=len(set(map(tuple,sid[:,:3].tolist())))
    assert abs(1-distinct/n-tiger['structure']['collision_prefix3'])<1e-12
    report={'status':'PASS','items':n,'manifest_assets':len(manifest['assets']),
            'finite_valid_curvature':int(triples['valid'].sum()),'missing_curvature':int((~summary['valid']).sum()),
            'strict_training_key_intersection':len(strict&training),'triples_reproduced':True,
            'reference_distances_reproduced':True,'full_directed_twohop_membership_verified':n,
            'seconds':time.monotonic()-start,'command':COMMAND,'exit_code':0,
            'production_sha256':source_hashes(),'manifest_sha256':assets.sha256(OUT/'manifest.json')}
    dump(ROOT/'ACCEPTANCE.json',report)
    print(json.dumps(report),flush=True)


def main():
    if not (OUT/'manifest.json').exists(): build()
    validate()


if __name__ == '__main__': main()
