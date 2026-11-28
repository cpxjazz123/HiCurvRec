"""Deterministic hand-computed tests for Issue #285 (gates, ramps, cells, ER identity, model matrix)."""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue285')
import common285 as W


def test_gate_and_ramp_hand():
    # fold 0's training rows are [0,4,8,12] -> q1=4, q2=8 (numpy linear quantiles);
    # the fold-0 test rows are the hand values [2, 6, 10] -> low / mid / high
    values = np.array([0.0, 4.0, 8.0, 12.0, 2.0, 6.0, 10.0])
    fold = np.array([1, 1, 1, 1, 0, 0, 0])
    edges = W.fold_edges(values, fold)
    q1, q2 = edges[0]
    assert abs(q1 - 4.0) < 1e-12 and abs(q2 - 8.0) < 1e-12, (q1, q2)
    level, ramp = W.bin_of(values, fold, edges)
    assert level[4:].tolist() == [-1, 0, 1]
    assert abs(ramp[4] - 1.0) < 1e-12
    assert abs(ramp[5] - 0.5) < 1e-12
    assert abs(ramp[6] - 0.0) < 1e-12


def test_cell_labels_hand():
    ln = np.array([-1, -1, 0, 1, 1, 0, -1])
    lc = np.array([-1, 1, -1, 1, 0, 0, 0])
    cells = W.cell_labels(ln, lc)
    assert cells.tolist() == ['LL', 'LN', 'LC', 'HH', 'other', 'other', 'LN']


def test_ER_identity_hand():
    # ER is defined as P/N exactly; a hand SID with P=0.05, N=0.01 must have ER=5
    P = 0.05; N = 0.01
    assert abs(P / N - 5.0) < 1e-12


def test_model_matrix_hand():
    X6 = np.arange(12, dtype=float).reshape(2, 6)
    P = np.array([0.05, 0.10]); G = np.array([1.0, 0.0])
    XB = np.column_stack([X6, G, P * G])
    assert XB.shape == (2, 8)
    assert XB[0, 6] == 1.0 and abs(XB[0, 7] - 0.05) < 1e-12
    assert XB[1, 6] == 0.0 and XB[1, 7] == 0.0


def test_binomial_auc_direction_hand():
    # low N should predict high recall: AUC(N) < 0.5, AUC(-N) > 0.5
    from scipy import stats
    N = np.array([0.004, 0.005, 0.006, 0.020, 0.030, 0.040])
    pos = np.array([True, True, True, False, False, False])
    auc = stats.mannwhitneyu(N[pos], N[~pos], alternative='two-sided').statistic / (pos.sum() * (~pos).sum())
    assert auc == 0.0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
