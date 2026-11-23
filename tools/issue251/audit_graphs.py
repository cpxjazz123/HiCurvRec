"""Read-only proof of frozen-asset/canonical graph identity or incompatibility."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
PATHS={'shared_forward':'assets/train/visible_forward.npz',
       'shared_undirected':'assets/train/visible_undirected.npz',
       'canonical_directed':'results/research/behavior_graph_directed/graph.npz',
       'canonical_undirected':'results/research/behavior_graph/graph.npz'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def keys(path):
    with np.load(path) as data:
        n=int(data['shape'][0]) if 'shape' in data else int(data['nodes'])
        prefix='out_' if 'out_indptr' in data else ''
        indptr=data[prefix+'indptr'];indices=data[prefix+'indices']
        rows=np.repeat(np.arange(n,dtype=np.int64),np.diff(indptr))
        return n,np.unique(rows*n+indices.astype(np.int64))


def compare(left,right):
    n,a=left;m,b=right
    if n!=m:raise ValueError('graph node universes differ')
    common=int(len(np.intersect1d(a,b,assume_unique=True)))
    return {'shared_edges_or_arcs':len(a),'canonical_edges_or_arcs':len(b),
            'common':common,'shared_only':len(a)-common,'canonical_only':len(b)-common,
            'same_topology':bool(np.array_equal(a,b))}


def main():
    graphs={name:keys(ROOT/path) for name,path in PATHS.items()}
    ids=np.load(ROOT/'assets/train/item_ids.npy')
    if not np.array_equal(ids,np.arange(graphs['shared_forward'][0])):
        raise ValueError('shared graph item order is not the canonical stage0 row order')
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    directed=json.loads((ROOT/'results/research/behavior_graph_directed/metadata.json').read_text())
    undirected=json.loads((ROOT/'results/research/behavior_graph/metadata.json').read_text())
    train=digest(ROOT/'results/stage0_build_parquet/train.parquet')
    source_hashes=[manifest['inputs']['train'],directed['sources']['train.parquet']['sha256'],
                   undirected['sources']['results/stage0_build_parquet/train.parquet']]
    report={'training_sha256':train,'training_sources_agree':all(x==train for x in source_hashes),
        'nodes':len(ids),'graphs':{name:{'path':path,'sha256':digest(ROOT/path)} for name,path in PATHS.items()},
        'directed':compare(graphs['shared_forward'],graphs['canonical_directed']),
        'undirected':compare(graphs['shared_undirected'],graphs['canonical_undirected']),
        'audit_source_sha256':digest(Path(__file__)),
        'interpretation':'Graph-set equality is required by the canonical contract; a strict subgraph is not an equal graph. No graph or target asset is rebuilt.'}
    output=ROOT/'results/research/issue251/GRAPH_AUDIT.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':main()
