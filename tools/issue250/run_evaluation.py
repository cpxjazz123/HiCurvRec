"""CPU-only, independently decode the controlled validation checkpoint/SID."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE=ROOT/'stage2_RQ-VAE/TIGER_gnn_gate_sectional_rq'
sys.path.insert(0,str(TREE))
from model.geometry import fixed_squared
from tools.issue250.evaluation import evaluate,evaluate_heldout,compare_historical_tiger
from tools.graph_infra.heldout_eval import HeldOutEvaluation
from tools.graph_infra.assets import load_sparse
from tools.graph_infra.curvature_stream import CurvatureReader


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    runtime=ROOT/'results/research/issue250/runtime'
    checkpoint=runtime/'controlled_microstep.pth'
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=blob['state_dict']
    sid_file=runtime/'item_sids.json'
    raw=json.loads(sid_file.read_text())
    sid=np.asarray([raw[str(i)] for i in range(len(raw))],dtype=np.int64)
    books=[state[f'rq.vq_layers.{i}.embed.weight'].numpy() for i in range(3)]
    reader=CurvatureReader(ROOT,ROOT/'assets')
    centres=np.flatnonzero(reader.item_count>0)[:16]
    rows=[reader.item_rows(int(item)) for item in centres]
    triples=np.concatenate([np.column_stack((row['centre'],row['b'],row['c'])) for row in rows])
    target=np.concatenate([row['curvature'] for row in rows])
    valid=np.concatenate([row['valid'] for row in rows])
    with np.load(ROOT/'assets/node_moments.npz') as source:components=source['component']
    references={int(item):np.flatnonzero(components==components[item]) for item in centres}
    q=sum(books[i][sid[:,i]] for i in range(3))
    # Decode through the saved decoder only: no continuous encoder substitutes.
    h=torch.from_numpy(q)
    indices=sorted(int(key.split('.')[2]) for key in state if key.startswith('decoder.mlp.') and key.endswith('.weight'))
    for position,index in enumerate(indices):
        h=torch.nn.functional.linear(h,state[f'decoder.mlp.{index}.weight'],state[f'decoder.mlp.{index}.bias'])
        if position+1<len(indices):h=torch.relu(h)
    graph=load_sparse(ROOT/'assets/train/visible_undirected.npz')
    report=evaluate(sid,books,triples,{'values':target,'valid':valid},references,
                    components_or_graph_optional=graph,scale=float(state['tangent_scale']),
                    fixed_squared=fixed_squared,
                    embeddings=np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy'),
                    reconstruction=h.numpy())
    report.pop('decoded_q')
    report['strict_heldout']=evaluate_heldout(sid,books,HeldOutEvaluation(ROOT),
                       scale=float(state['tangent_scale']),fixed_squared=fixed_squared)
    report['historical_tiger']=compare_historical_tiger(ROOT)
    report['provenance']={'sid_sha256':digest(sid_file),'checkpoint_sha256':digest(checkpoint),
                         'scope':'controlled one-microstep diagnostic; not trained improvement',
                         'centres':centres.tolist(),'triples':len(triples),
                         'references':'full frozen component; all valid triples of listed centres',
                         'scale_fitted_on_evaluation':False}
    output=ROOT/'results/research/issue250/HARD_SID_EVALUATION.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output.relative_to(ROOT)),'curvature':report['curvature']},default=float))

if __name__=='__main__':main()
