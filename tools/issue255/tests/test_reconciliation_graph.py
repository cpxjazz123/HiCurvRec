"""The immutable directed graph is raw CSR, not SciPy's save_npz format."""
import numpy as np

from tools.issue255 import reconcile_results


def test_raw_csr_edges_are_unweighted_and_isolates_excluded(tmp_path):
    path = tmp_path / 'visible_forward.npz'
    # 0 -> 1, 0 -> 2, 2 -> 1. Node1 is an out-isolate. Edge counts
    # must not turn the per-edge sharing rate into a frequency-weighted rate.
    np.savez(path, indptr=np.array([0, 2, 2, 3], dtype=np.int32),
             indices=np.array([1, 2, 1], dtype=np.int32),
             data=np.array([10, 1, 99]), shape=np.array([3, 3]))
    result = reconcile_results.visible_sharing(path, np.array([7, 7, 8]))
    assert result['micro'] == 1 / 3
    assert result['out_centre_macro'] == .25


def test_raw_csr_without_edges_has_no_fabricated_rate(tmp_path):
    path = tmp_path / 'visible_forward.npz'
    np.savez(path, indptr=np.zeros(4, dtype=np.int32),
             indices=np.array([], dtype=np.int32), data=np.array([], dtype=np.int64),
             shape=np.array([3, 3]))
    assert reconcile_results.visible_sharing(path, np.array([7, 7, 8])) == {
        'micro': None, 'out_centre_macro': None}
