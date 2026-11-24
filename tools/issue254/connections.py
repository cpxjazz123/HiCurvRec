"""Persist item-ID joins for wrong/target/history features and descriptive concordance."""
import json
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
OUTPUT = ROOT / 'results/research/issue254'


def sign_concordance(gates, ratios, valid):
    """Top sign labels only; this is not a test of an item's optimal geometry."""
    gates, ratios, valid = np.asarray(gates,float), np.asarray(ratios,float), np.asarray(valid,bool)
    if gates.shape != ratios.shape or gates.ndim != 2 or gates.shape[1] != 3 or valid.shape != (len(gates),):
        raise ValueError('distribution shape mismatch')
    if not np.isfinite(gates).all() or not np.allclose(gates.sum(1),1,atol=2e-6,rtol=0):
        raise ValueError('invalid gate distribution')
    if not np.isfinite(ratios[valid]).all() or not np.allclose(ratios[valid].sum(1),1,atol=1e-12,rtol=0):
        raise ValueError('invalid curvature distribution')
    result = np.full(len(gates),'unavailable',dtype='<U11')
    result[valid] = np.where(gates[valid].argmax(1)==ratios[valid].argmax(1),'same','different')
    return result


def main():
    import pandas as pd
    from tools.issue254.structure import sha256
    tables = OUTPUT / 'tables'
    paths = ['joined_item_diagnostics.csv','joined_user_diagnostics.csv','candidate_errors.csv','paired_user_ledger.csv','mechanism_arrays.npz']
    inputs = {str((tables/p).relative_to(ROOT)):sha256(tables/p) for p in paths}
    items = pd.read_csv(tables/'joined_item_diagnostics.csv').set_index('item',verify_integrity=True)
    if not np.array_equal(items.index.to_numpy(),np.arange(24587)):
        raise ValueError('unordered or missing catalogue IDs')
    users = pd.read_csv(tables/'joined_user_diagnostics.csv')
    ratios = items[['curvature_negative_ratio','curvature_zero_ratio','curvature_positive_ratio']].to_numpy()
    valid = ~items['curvature_missing'].to_numpy(bool)
    diagnostics = pd.DataFrame({'item':items.index})
    for arm in ['issue252','issue253']:
        gates = items[['gate_'+k+'_'+arm for k in ['negative','zero','positive']]].to_numpy()
        diagnostics['sign_label_concordance_'+arm] = sign_concordance(gates,ratios,valid)
        logg = np.zeros_like(gates)
        np.log(gates,out=logg,where=gates>0)
        diagnostics['three_sign_gate_entropy_'+arm] = -(gates*logg).sum(1)
    joined = users[['global_row','user','sample_sha256','target_item','group5','group10','delta_hit5','delta_hit10']].merge(
        diagnostics,left_on='target_item',right_on='item',how='left',validate='many_to_one')
    raw = pd.read_csv(tables/'paired_user_ledger.csv',usecols=['global_row','user','sample_sha256','target_item','last_history_item','topk_issue252','topk_issue253'])
    keys=['global_row','user','sample_sha256','target_item']
    if not joined[keys].equals(raw[keys]) or len(joined)!=57439:
        raise ValueError('context concordance identity mismatch')
    families = items['dominant_family'].fillna('unavailable')
    for arm in ['issue252','issue253']:
        first_wrong = [next(item for item in json.loads(seq) if item != target)
                       for seq,target in zip(raw['topk_'+arm],raw['target_item'])]
        joined['first_non_target_candidate_'+arm]=first_wrong
        joined['candidate_same_dominant_family_'+arm]=(families.loc[first_wrong].to_numpy()==families.loc[raw['target_item']].to_numpy())
    joined.to_csv(tables/'context_concordance.csv',index=False)
    details=pd.read_csv(tables/'candidate_errors.csv')
    feature_columns=['degree_in','degree_out','degree_undirected','curvature_valid','curvature_missing','curvature_negative_ratio',
                     'curvature_positive_ratio','curvature_mean','curvature_variance','dominant_family']
    for arm in ['issue252','issue253']:
        feature_columns += [stem+'_'+arm for stem in ['gate_negative','gate_zero','gate_positive','attention_1hop','attention_2hop',
                                                     'curvature_mae_q','reconstruction_mse_q','quantization_mse','collision_group_size']]
    for role,key in [('target','target_item'),('candidate','candidate_item'),('last_history','last_history_item')]:
        lookup=items[feature_columns].rename(columns={c:role+'_'+c for c in feature_columns})
        details=details.merge(lookup,left_on=key,right_index=True,how='left',validate='many_to_one')
    if len(details)!=40716 or details['candidate_degree_undirected'].isna().any() or details['target_degree_undirected'].isna().any():
        raise ValueError('candidate/target item join lost coverage')
    details.to_csv(tables/'candidate_mechanism_join.csv',index=False)
    groups={}
    feature_stems=['gate_zero','attention_1hop','attention_2hop','curvature_mae_q','reconstruction_mse_q']
    for group,subset in joined.groupby('group10'):
        selection=users[users['group10']==group]
        record={'samples':len(subset),'targets':int(subset['target_item'].nunique())}
        for arm in ['issue252','issue253']:
            record[arm]={'sign_label_concordance':subset['sign_label_concordance_'+arm].value_counts().to_dict(),
                         'mean_three_sign_gate_entropy':float(subset['three_sign_gate_entropy_'+arm].mean()),
                         'first_non_target_same_family_fraction':float(subset['candidate_same_dominant_family_'+arm].mean()),
                         'mean':{stem:float(selection[stem+'_'+arm].mean()) if stem+'_'+arm in selection else
                                 float(items.loc[subset['target_item'],stem+'_'+arm].mean()) for stem in feature_stems}}
        groups[group]=record
    by_match={}
    for arm in ['issue252','issue253']:
        by_match[arm]={}
        for label,subset in joined.groupby('sign_label_concordance_'+arm):
            by_match[arm][label]={'samples':len(subset),'net_hit_count':int(subset['delta_hit10'].sum()),
                                 'global_recall_contribution':float(subset['delta_hit10'].sum()/57439),
                                 'conditional_recall_delta':float(subset['delta_hit10'].mean())}
    for path,expected in inputs.items():
        if sha256(ROOT/path)!=expected: raise ValueError('input changed during connections: '+path)
    summary={'schema':'genrec.issue254.connections/1','purpose':'evaluation','population':len(joined),'candidate_rows':len(details),
             'inputs_sha256':inputs,'source_sha256':sha256(Path(__file__)), 'groups10':groups,'descriptive_sign_concordance':by_match,
             'definitions':{'sign_labels':'argmax of negative/zero/positive gate and GraphMoRE proxy target fractions; first sign in tie',
                            'concordance':'descriptive sign-label agreement only; not geometric optimality or graphlet-specific causal matching',
                            'family_match':'dominant exact three-node family of target versus first non-target selected candidate',
                            'entropy':'Shannon entropy over the same three aggregated sign weights, in nats'},
             'outputs':{p:{'path':str((tables/p).relative_to(ROOT)),'sha256':sha256(tables/p)} for p in ['context_concordance.csv','candidate_mechanism_join.csv']}}
    (OUTPUT/'CONNECTIONS_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'population':len(joined),'candidate_rows':len(details),'groups10':groups},indent=2))


if __name__=='__main__':
    main()
