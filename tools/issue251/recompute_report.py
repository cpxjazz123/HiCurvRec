"""Reconcile REPORT with raw SIDs, logs and independently remeasured distributions.

No production model or evaluator imports. Centre-weighted and triple-weighted
headline reductions use Python fsum, independently of the streaming NumPy code.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
OUT=ROOT/'results/research/issue251'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curvature_headlines(per_item):
    rows=list(per_item.values());count=sum(r['count'] for r in rows)
    return {'count':count,'centres':len(rows),
            'macro_mae':math.fsum(r['mean_abs_error'] for r in rows)/len(rows),
            'micro_mae':math.fsum(r['count']*r['mean_abs_error'] for r in rows)/count}


def sid_headlines(rows):
    result={}
    for level in (1,2,3):
        counts=Counter(tuple(row[:level]) for row in rows)
        result['L%d'%level]={'distinct':len(counts),'collision':1-len(counts)/len(rows),
            'sharing':sum(n for n in counts.values() if n>1)/len(rows),
            'used':len({row[level-1] for row in rows})}
    return result


def close(actual,expected):
    if not math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=1e-9):
        raise AssertionError((actual,expected))


def table_rows(text):
    return [[cell.strip() for cell in line.strip().strip('|').split('|')]
            for line in text.splitlines() if line.startswith('|')]


def load_sid(path):
    raw=json.loads(path.read_text())
    assert set(raw)=={str(i) for i in range(len(raw))},'global item identities drifted'
    return [raw[str(i)] for i in range(len(raw))]


def main():
    raw=json.loads((OUT/'HARD_SID_EVALUATION.json').read_text())
    summary=json.loads((OUT/'EVALUATION_SUMMARY.json').read_text())
    source=OUT/'FINAL_SOURCE_BINDINGS.json'
    for name,expected in json.loads(source.read_text())['files'].items():
        assert sha256(ROOT/name)==expected,'executed/final source drift: '+name
    coverage=json.loads((ROOT/'assets/coverage.json').read_text())
    report=OUT/'REPORT.md';rows=table_rows(report.read_text());heads={};expected={}
    for label,result in (('hard_q',raw),('continuous_z',raw['continuous_encoder'])):
        curvature=result['curvature'];heads[label]=curvature_headlines(curvature['per_item'])
        assert heads[label]['count']==coverage['triples_expected']
        assert heads[label]['centres']==coverage['centres_with_triples']
        for name,value in heads[label].items():close(curvature[name],value)
        sign=curvature['sign']['recall']
        balanced=math.fsum(v['recall'] for v in sign.values() if v['count'])/sum(v['count']>0 for v in sign.values())
        close(curvature['sign']['balanced_accuracy'],balanced)
        expected[label]={'macro_mae':heads[label]['macro_mae'],'micro_mae':heads[label]['micro_mae'],
            'sign_balanced_accuracy':balanced,
            'distance_mean_absolute_error':result['distance_distortion']['mean_abs_error'],
            'distance_mean_relative_error':result['distance_distortion']['mean_relative_error'],
            'reconstruction_mse':result['reconstruction']['mean_squared_error'],
            'reconstruction_mean_l2':result['reconstruction']['mean_l2_error']}
    for metric in expected['hard_q']:
        found=[r for r in rows if r[0]==metric];assert len(found)==1,metric
        close(found[0][1],expected['hard_q'][metric]);close(found[0][2],expected['continuous_z'][metric])
    sid_path=OUT/'runtime/item_sids.json';discrete=sid_headlines(load_sid(sid_path))
    for level,values in discrete.items():
        found=[r for r in rows if r[0]==level];assert len(found)==1,level
        for actual,name in zip(found[0][1:],('distinct','collision','sharing','used')):close(actual,values[name])
        close(summary['prefix'][level]['collision_prefix'],values['collision'])
    for name,bank in raw['strict_heldout']['banks'].items():
        z=raw['continuous_encoder']['strict_heldout']['banks'][name]
        found=[r for r in rows if r[0]==name.removeprefix('strict_struct_')];assert len(found)==1,name
        values=[bank['pairs'],bank['topology']['hop_full_visible']['count'],bank['distance']['mean'],
                z['distance']['mean'],bank['topology']['hop_full_visible']['distance_spearman'],
                z['topology']['hop_full_visible']['distance_spearman']]
        for actual,expected_value in zip(found[0][1:],values):close(actual,expected_value)
        assert bank['distance_ranking_auc']['status']=='NA' and z['distance_ranking_auc']['status']=='NA'
    archive=OUT/'entry_cf0_uncached';entry_sid=sid_headlines(load_sid(archive/'item_sids.json'))
    events=[json.loads(line) for line in (archive/'logs/training_metrics.jsonl').read_text().splitlines()]
    entry_end=next(r for r in reversed(events) if r['event']=='train_end')
    close(entry_end['final_collision'],entry_sid['L3']['collision'])
    historical=raw['historical_tiger'];assert sha256(ROOT/historical['sid_path'])==historical['sid_sha256']
    close(sid_headlines(load_sid(ROOT/historical['sid_path']))['L3']['collision'],
          historical['pure_sid_metrics']['collision_prefix3']['collision_prefix'])
    assert historical['geometry_metrics']['status']=='NA'
    result={'status':'PASS','command':sys.executable+' -m tools.issue251.recompute_report','exit_code':0,
        'curvature_from_per_item':heads,'discrete_from_actual_sid':discrete,
        'first_real_entry_collision':entry_sid['L3']['collision'],'strict_banks_checked':len(raw['strict_heldout']['banks']),
        'report_sha256':sha256(report),'raw_report_sha256':sha256(OUT/'HARD_SID_EVALUATION.json'),
        'sid_sha256':sha256(sid_path),'source_bindings_sha256':sha256(source),'recheck_source_sha256':sha256(Path(__file__)),
        'scope':'Independent aggregation/report reconciliation after full raw-asset NumPy remeasurement; not new training.'}
    (OUT/'REPORT_RECHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
