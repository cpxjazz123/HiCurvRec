import json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from metrics_core import prefix_stats
from stats import pearson,spearman

def test_prefix_known_answers():
    rows=[[1,2,3,999],[1,2,3,999],[1,2,4,999],[2,2,4,999]]
    s=prefix_stats(rows)
    assert s['items']==4
    assert s['distinct_L1']==2 and s['distinct_L2']==2 and s['distinct_L3']==3
    assert abs(s['collision']-0.25)<1e-12
    assert s['groups_max_L1']==3 and s['dead_codes_L1']==254
    assert abs(s['H_L1']-0.8112781244591328)<1e-12
    assert abs(s['H_L2_given_L1']-0.0)<1e-12
    assert abs(s['H_L3_given_L1L2']-0.6887218755408672)<1e-12

def test_constant_and_missing_are_na():
    assert pearson([1,1,1],[1,2,3]) is None
    assert spearman([1,2,3],[1,None,3]) is not None

def test_inventory_is_complete_and_no_training():
    d=json.load(open(ROOT/'INVENTORY.json'))
    assert len(d['arms'])==8
    assert d['training_count']==0
    for n,a in d['arms'].items():
        assert a['checks']['checkpoint']['status']=='PASS'
        assert a['checks']['sid']['status']=='PASS'
        assert a['checks']['recall_json']['status']=='PASS'
        assert a['checks']['stage2_training']['epochs_complete'] is True
        assert a['checks']['stage3']['rows_match'] is True
