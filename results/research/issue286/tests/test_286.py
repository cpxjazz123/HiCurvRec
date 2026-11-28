"""Deterministic hand-computed tests for Issue #286 (constrained gate, mixture, hard gate, cross-fit)."""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue286')
import gate286 as G


def test_constrained_logit_signs():
    # increasing covariate with increasing target -> positive coefficient
    x = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([0.0, 0.0, 1.0, 1.0])
    b = G.fit_constrained_logit(np.column_stack([np.ones(4), x]), t)
    assert b[1] > 0.0
    # decreasing target with increasing covariate -> the unconstrained slope would be negative -> 0
    t2 = np.array([1.0, 1.0, 0.0, 0.0])
    b2 = G.fit_constrained_logit(np.column_stack([np.ones(4), x]), t2)
    assert abs(b2[1]) < 1e-9


def test_soft_mixture_hand():
    r6, eg, eo, g, lam = 0.05, 0.01, -0.01, 0.3, 0.5
    mix = r6 + lam * (g * eg + (1 - g) * eo)
    assert abs(mix - 0.048) < 1e-12


def test_gate_extremes():
    sig = G.sigmoid(np.array([-50.0, 0.0, 50.0]))
    assert sig[0] < 1e-9 and abs(sig[1] - 0.5) < 1e-12 and sig[2] > 1 - 1e-9
    # g=1 -> the good expert only; g=0 -> the other expert only
    r6, eg, eo, lam = 0.05, 0.02, -0.02, 1.0
    assert abs((r6 + lam * (1.0 * eg + 0.0 * eo)) - 0.07) < 1e-12
    assert abs((r6 + lam * (0.0 * eg + 1.0 * eo)) - 0.03) < 1e-12


def test_hard_gate():
    g = np.array([0.1, 0.49, 0.5, 0.9])
    hi = g >= 0.5
    assert hi.tolist() == [False, False, True, True]


def test_crossfit_excludes_the_group():
    # 8 rows, 2 folds; every cross-fitted prediction must equal the model fitted on the other fold
    G.X6 = np.arange(16, dtype=float).reshape(8, 2)
    G.y = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    fold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = G.crossfit_r6(np.arange(8), fold)
    m = G.fit_r6(np.where(fold == 0)[0], fold[fold == 0])
    assert np.allclose(pred[4:], G.pred_r6(m, np.arange(4, 8)))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
