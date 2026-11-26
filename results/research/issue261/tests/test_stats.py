import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stats import pearson, spearman, fisher_ci, p_value_pearson, leave_one_out

def test_pearson_hand():
    assert abs(pearson([1,2,3],[2,4,6])-1)<1e-12
    assert abs(pearson([1,2,3],[6,4,2])+1)<1e-12
    assert pearson([1,1],[1,2]) is None
    assert pearson([1,None,3],[1,2,3]) is not None

def test_spearman_ties_and_hand():
    assert abs(spearman([1,2,3],[10,20,30])-1)<1e-12
    assert abs(spearman([1,1,2],[1,2,3])-0.8660254037844387)<1e-12

def test_fisher_and_loo():
    lo,hi=fisher_ci(0.5,10); assert lo<0.5<hi
    loo=leave_one_out([1,2,3,4],[2,4,6,8]); assert loo==(1.0,1.0)
    assert p_value_pearson(1.0,8) is None
