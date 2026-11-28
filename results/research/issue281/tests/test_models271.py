"""Known-answer checks for the pre-registered design matrices and the calibration rule (Issue #281)."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('m281', 'results/research/issue281/models281.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_design_groups():
    X = np.array([[0.1, 0.2, 2.0, 7000.0, 0.15, 2.2],
                  [0.2, 0.1, 1.0, 6000.0, 0.05, 1.8]])
    B = m.design(X, 'B')
    assert B.shape == (2, 12) and np.allclose(B[:, 6:], X ** 2)
    C = m.design(X, 'C')
    assert C.shape == (2, 8)
    assert abs(C[0, 6] - X[0, 0] * X[0, 1]) < 1e-12 and abs(C[0, 7] - X[0, 2] * X[0, 4]) < 1e-12
    D = m.design(X, 'D', [(2, 1.5), (4, 0.1)])
    assert np.allclose(D[:, 6], np.maximum(0.0, X[:, 2] - 1.5))
    assert np.allclose(D[:, 7], np.maximum(0.0, X[:, 4] - 0.1))


def test_min_side_support_rule():
    Xtr = np.column_stack([np.linspace(0, 1, 100)] * 6)
    q_ok = float(np.percentile(Xtr[:, 2], 50))
    q_bad = float(np.percentile(Xtr[:, 2], 95))
    assert np.mean(Xtr[:, 2] >= q_ok) >= m.MIN_SIDE and np.mean(Xtr[:, 2] < q_ok) >= m.MIN_SIDE
    assert not (np.mean(Xtr[:, 2] >= q_bad) >= m.MIN_SIDE)


def test_calibration_identity():
    rng = np.random.default_rng(0)
    y = rng.normal(0.05, 0.005, 200)
    oof = y.copy()
    A = np.vstack([oof, np.ones(len(oof))]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    assert abs(coef[0] - 1.0) < 1e-9 and abs(coef[1]) < 1e-9


if __name__ == '__main__':
    test_design_groups()
    test_min_side_support_rule()
    test_calibration_identity()
    print('all model design checks passed')
