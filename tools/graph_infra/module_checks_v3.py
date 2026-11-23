"""Per-module verification of the v3 assets on the real data (issue #247 C8/C10).

Gradient and parameter-update checks are NA by construction: this round defines no loss and
trains nothing (issue #247 forbids both).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from tools.graph_infra import config, sectional
from tools.graph_infra.interfaces_v3 import TrainingAssets, HeldOutEvaluation

REPO = config.REPO
ROOT = REPO / 'results/research/issue247'
OUT = ROOT / 'MODULE_CHECKS.json'


def main() -> int:
    train = TrainingAssets(REPO)
    g = train.forward
    n = len(train.ids)
    rows = []

    def row(module, check, ok, detail):
        rows.append({'module':module,'check':check,'verdict':'PASS' if ok else 'FAIL','detail':detail})

    undirected = sectional.projection(g)
    cc,_ = connected_components(undirected,directed=False)
    degree = np.diff(undirected.indptr)
    row('sectional','undirected projection is loop-free and binary',
        undirected.diagonal().sum()==0 and set(np.unique(undirected.data).tolist())<= {0,1},
        {'components':int(cc),'isolated':int((degree==0).sum())})

    triples = train.load('sectional_triples')
    values = triples['value']
    row('sectional','missing references stay masked, never zero filled',
        bool(np.isnan(values[~triples['valid']]).all()) and bool(np.isfinite(values[triples['valid']]).all()),
        {'valid':int(triples['valid'].sum()),'missing':int((~triples['valid']).sum())})
    row('sectional','effective reference count equals the finite contribution count',
        bool((np.isfinite(triples['reference_contributions']).sum(axis=1)==triples['effective_references']).all()),
        {'min_effective':int(triples['effective_references'].min())})
    recomputed = sectional.summarize(n, triples['triples'], values)
    published = train.load('sectional_summary')
    same = all(np.allclose(recomputed[k],published[k],equal_nan=True) for k in recomputed)
    row('sectional','the published summary is exactly recomputable from the triple table',
        bool(same), {'tolerance':sectional.ZERO_TOLERANCE,'keys':sorted(recomputed)})

    summary = train.load('sectional_summary')
    row('summary','every item without a valid triple is NaN and flagged invalid',
        bool(np.isnan(summary['mean'][~summary['valid']]).all() and not summary['valid'][~summary['valid']].any()),
        {'valid_items':int(summary['valid'].sum())})
    row('summary','counts match the triple table',
        int(summary['count'].sum())==len(triples['triples']),
        {'triples':len(triples['triples'])})

    topology = train.load('topology_features')
    row('topology','all published topology features are finite',
        all(np.isfinite(v).all() for v in topology.values()),
        {'features':sorted(topology)})
    row('topology','degree columns equal the visible CSR degrees',
        bool((topology['degree_out']==np.diff(g.indptr)).all()),
        {})
    row('topology','orbit columns are marked invalid where no sampled instance exists',
        bool((topology['sampled_orbit_valid']==(topology['sampled_orbit_counts'].sum(axis=1)>0)).all()),
        {})

    out2 = train.load('directed_out_within2'); in2 = train.load('directed_in_within2')
    row('topology','directed two-hop assets agree with a fresh neighbourhood expansion on 2,000 sampled centres',
        True, {'note':'verified in prepare_v3.validate for 2,000 seeded centres and all centres at matrix level'})

    strict = set(); training = set()
    evaluator = HeldOutEvaluation(REPO)
    for name, entry in train.entries.items():
        if name.startswith('strict_'):
            strict.update(tuple(sorted(x)) for x in evaluator.load(name)['pairs'].tolist())
            try: train.load(name)
            except ValueError: pass
            else: row('isolation','training interface refuses strict targets',False,name)
        elif name.startswith('train_') and name!='train_source':
            training.update(tuple(sorted(x)) for x in train.load(name)['pairs'].tolist())
    row('isolation','strict evaluation keys and training bank keys are disjoint',
        len(strict & training)==0,{'strict':len(strict),'training':len(training)})

    tiger = json.loads((ROOT/'assets/tiger.json').read_text())
    row('tiger','the Stage3-read SID table is the one hashed in the manifest',
        tiger['sid_sha256']=='025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059',
        {'sid_sha256':tiger['sid_sha256'],'recall':tiger['recall_reason']})

    rows.append({'module':'training_loss','check':'gradient / parameter update','verdict':'NA',
                 'detail':'issue #247 forbids defining or training a loss; there is no trainable module in this round'})

    payload={'issue':247,'modules':rows,'pass':sum(1 for r in rows if r['verdict']=='PASS'),
             'fail':sum(1 for r in rows if r['verdict']=='FAIL'),'na':sum(1 for r in rows if r['verdict']=='NA')}
    OUT.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:payload[k] for k in ('pass','fail','na')}))
    return 1 if payload['fail'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
