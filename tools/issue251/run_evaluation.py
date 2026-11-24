"""CPU-only, independently decode the controlled validation checkpoint/SID."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE=ROOT/'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq'
sys.path.insert(0,str(TREE))
from model.geometry import fixed_squared
from model.mechanism import RQVAE
import train_rqvae as training
from tools.issue251.evaluation import (evaluate,evaluate_heldout,evaluate_continuous_heldout,
                                     evaluate_reconstruction,compare_historical_tiger)
from tools.issue251.full_evaluation import full_evaluate,fixed_squared_numpy
from tools.graph_infra.heldout_eval import HeldOutEvaluation


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    runtime=ROOT/'results/research/issue251/runtime'
    checkpoint=runtime/'controlled_microstep.pth'
    model=RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE)
    model.load_checkpoint(checkpoint)
    model.eval()
    state=model.state_dict()
    sid_file=runtime/'item_sids.json'
    raw=json.loads(sid_file.read_text())
    sid=np.asarray([raw[str(i)] for i in range(len(raw))],dtype=np.int64)
    books=[state[f'rq.vq_layers.{i}.embed.weight'].numpy() for i in range(3)]
    scale=float(state['tangent_scale'])
    with np.load(ROOT/'assets/node_moments.npz') as source:components=source['component']
    q=sum(books[i][sid[:,i]] for i in range(3))
    embeddings=np.load(ROOT/'stage1_GeneEmbedding/output/sentence_t5.npy')
    with torch.no_grad():
        reconstruction=model.decoder(torch.from_numpy(q)).numpy()
        continuous_tensor=model.encoder(torch.from_numpy(embeddings))
        continuous=continuous_tensor.numpy()
        continuous_reconstruction=model.decoder(continuous_tensor).numpy()
    empty=np.empty((0,3),dtype=np.int64)
    report=evaluate(sid,books,empty,np.empty(0),{},component_labels=components,
                    scale=scale,fixed_squared=fixed_squared,
                    embeddings=embeddings,reconstruction=reconstruction)
    report.pop('decoded_q')
    report['reconstruction'].pop('per_item_l2')
    agreement={}
    for name,coordinates in (('hard_q',q),('continuous_z',continuous)):
        left=coordinates[:64].astype(np.float64);right=coordinates[64:128].astype(np.float64)
        expected=fixed_squared(torch.from_numpy(left),torch.from_numpy(right),scale=scale).numpy()
        error=float(np.abs(expected-fixed_squared_numpy(left,right,scale=scale)).max())
        if error>1e-9:raise AssertionError('independent ambient geometry disagrees on actual '+name)
        agreement[name]=error
    report.update(full_evaluate(q,ROOT/'assets',scale=scale))
    continuous_report=full_evaluate(continuous,ROOT/'assets',scale=scale)
    continuous_report['representation']='continuous_encoder_z'
    continuous_report['reconstruction']=evaluate_reconstruction(
        embeddings,continuous_reconstruction,representation='continuous-z')
    continuous_report['reconstruction'].pop('per_item_l2')
    if report['curvature']['count']!=28391948 or continuous_report['curvature']['count']!=28391948:
        raise AssertionError('full independent evaluation did not visit every frozen valid triple')
    heldout=HeldOutEvaluation(ROOT)
    report['strict_heldout']=evaluate_heldout(sid,books,heldout,
                       scale=scale,fixed_squared=fixed_squared)
    continuous_report['strict_heldout']=evaluate_continuous_heldout(
        continuous,heldout,scale=scale,fixed_squared=fixed_squared)
    report['historical_tiger']=compare_historical_tiger(ROOT)
    report['continuous_encoder']=continuous_report
    report['provenance']={'sid_sha256':digest(sid_file),'checkpoint_sha256':digest(checkpoint),
                         'scope':'controlled one-microstep diagnostic; not trained improvement',
                         'triples':report['curvature']['count'],
                         'references':'all valid frozen triples and entire frozen components',
                         'ambient_vs_production_max_abs':agreement,
                         'evaluator_sources':{str(Path(path).relative_to(ROOT)):digest(path) for path in
                             (Path(__file__),ROOT/'tools/issue251/full_evaluation.py',
                              ROOT/'tools/issue251/evaluation.py')},
                         'scale_fitted_on_evaluation':False}
    output=ROOT/'results/research/issue251/HARD_SID_EVALUATION.json'
    output.write_text(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n')
    headline=lambda r:{key:value for key,value in r['curvature'].items() if key!='per_item'}
    summary={key:value for key,value in report.items() if key not in ('curvature','continuous_encoder')}
    summary['curvature']=headline(report)
    summary['continuous_encoder']={key:value for key,value in continuous_report.items() if key!='curvature'}
    summary['continuous_encoder']['curvature']=headline(continuous_report)
    summary['full_per_item_report']={'path':str(output.relative_to(ROOT)),'sha256':digest(output),
                                    'bytes':output.stat().st_size}
    (output.parent/'EVALUATION_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output.relative_to(ROOT)),'curvature':headline(report),
                      'continuous_curvature':headline(continuous_report)},default=float))

if __name__=='__main__':main()
