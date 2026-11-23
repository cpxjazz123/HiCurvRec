"""Validate and compact actual four-rank output; never execute a model or train."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'assets').is_dir())
OUTPUT=ROOT/'results/research/issue251'
RUNTIME=OUTPUT/'runtime'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def main():
    reports=[json.loads((RUNTIME/('rank_%d.json'%rank)).read_text()) for rank in range(4)]
    for rank,report in enumerate(reports):
        if report['status']!='pass' or report['rank']!=rank or report['world_size']!=4:
            raise AssertionError('actual four-rank proof incomplete')
        if report['master_port']!=50202 or report['formal_training']:
            raise AssertionError('verification violated frozen dispatch protocol')
        if report['gradient_rank_max_difference']>1e-6 or report['parameter_rank_max_difference']>1e-6:
            raise AssertionError('DDP rank agreement failed')
        if report['DDP_vs_independent_rank_mean_relative_error']>1e-4:
            raise AssertionError('DDP gradient differs from independently averaged real losses')
        for path,expected in report['source_hashes'].items():
            if digest(ROOT/path)!=expected:raise AssertionError('stale executed source: '+path)
        for group in ('encoder','GNN','Q','K','V','score','codebook'):
            if report['curvature_gradients'][group]['norm']<=0:
                raise AssertionError('actual curvature gradient missing: '+group)
        if report['curvature_gradients']['decoder']['norm']!=0:
            raise AssertionError('curvature unexpectedly reached decoder')
        for group in ('GNN','Q','K','V','score'):
            if report['detached_cost_curvature_gradients'][group]['norm']!=0:
                raise AssertionError('selection-cost cut failed: '+group)
        if min(report['module_update_norms'].values())<=0:
            raise AssertionError('a required trainable module did not update')
        if min(report['query_curvature_gradient_per_expert'])<=0:
            raise AssertionError('one expert query has no actual curvature gradient')
        calibration=report['calibration']
        np.testing.assert_allclose(calibration['lambda'],calibration['reconstruction_gradient']/
                                   calibration['curvature_gradient'],rtol=2e-6,atol=0)
    first=reports[0];full=first['full_target_end_to_end']
    if full['valid_triples']!=28391948 or full['full_catalogue_items']!=24587 or full['optimizer_steps']!=0:
        raise AssertionError('full target proof was truncated or trained')
    for group in ('encoder','GNN','Q','K','V','score','codebook'):
        if full['module_gradients'][group]['norm']<=0:
            raise AssertionError('full-target curvature gradient missing: '+group)
    if not first['export']['exact_reload'] or not first['export']['rank0_inference_without_collectives']:
        raise AssertionError('checkpoint-to-SID identity proof incomplete')
    if first['hard_sid_changed_items']<=0 or min(first['gate_weight_change_norm'],first['selection_cost_change_norm'])<=0:
        raise AssertionError('controlled update did not change actual gates/costs/SIDs')
    attention_file=RUNTIME/'catalogue_attention.npz'
    with np.load(attention_file) as source:
        attention=source['token_attention'];weights=source['expert_weights'];counts=source['token_counts']
    if attention.shape!=(24587,5,3) or weights.shape!=(24587,5) or not np.all(counts==2):
        raise AssertionError('global token/attention coverage incomplete')
    if not np.isfinite(attention).all() or not np.isfinite(weights).all():
        raise AssertionError('nonfinite learned attention')
    np.testing.assert_allclose(attention.sum(-1),1.,rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(weights.sum(-1),1.,rtol=2e-6,atol=2e-7)
    np.testing.assert_array_equal(attention[:,:,-1],0.)
    bindings={'source':first['source_hashes'],'contract':digest(OUTPUT/'CONTRACT.md'),
              'asset_manifest':digest(ROOT/'assets/manifest.json'),
              'checkpoint':digest(RUNTIME/'controlled_microstep.pth'),
              'sid':digest(RUNTIME/'item_sids.json'),'attention':digest(attention_file)}
    if bindings['sid']!=first['export']['export_sha256'] or bindings['checkpoint']!=first['export']['checkpoint_sha256']:
        raise AssertionError('persisted export/checkpoint digest drift')
    if bindings['attention']!=full['attention_sha256']:
        raise AssertionError('persisted attention digest drift')
    module_rows=[]
    for rank,report in enumerate(reports):
        for group,comparison in report['module_gradient_comparison'].items():
            module_rows.append({'rank':rank,'module':group,'run_id':report['run_id'],
                'worker_pid':report['worker_pid'],'original':report['original_gradients'][group],
                'curvature':report['curvature_gradients'][group],**comparison,
                'update_norm':report['module_update_norms'][group],
                'combined_global_preclip_norm':report['preclip_norm'],
                'combined_global_postclip_norm':report['postclip_norm'],
                'clip_limit':report['clip_norm'],'full_target_rank0_curvature':full['module_gradients'][group]})
    (OUTPUT/'MODULE_CHECKS.json').write_text(json.dumps({'bindings':bindings,'modules':module_rows},indent=2)+'\n')
    summary={key:value for key,value in first.items() if key not in
             ('source_hashes','parameter_update_norms','module_gradient_comparison','original_gradients',
              'curvature_gradients','detached_cost_curvature_gradients')}
    summary['rank_checks']=[{key:report[key] for key in ('rank','worker_pid','controller_pid','run_id','status',
        'master_port','cuda_peak_bytes','wall_seconds','timings_seconds','gradient_rank_max_difference',
        'parameter_rank_max_difference','DDP_vs_independent_rank_mean_relative_error')} for report in reports]
    summary['bindings']=bindings
    summary['token_coverage']={'items':len(counts),'tokens_per_item':2,'coverage':1.,
        'scales':['full_one_hop','full_two_hop'],'attention_axes':['item','expert','token'],
        'geometry_weight_axes':['item','expert'],'null_slot_index':2,'null_active_items':0,
        'attention_max_normalization_error':float(np.abs(attention.sum(-1)-1).max()),
        'weight_max_normalization_error':float(np.abs(weights.sum(-1)-1).max())}
    (OUTPUT/'RUNTIME_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','module_rows':len(module_rows),'full_valid_triples':full['valid_triples'],
        'changed_sid_items':first['hard_sid_changed_items'],'bindings':bindings},sort_keys=True))


if __name__=='__main__':main()
