"""Full, streamed hard-q/z evaluation: no diagnostic-centre truncation."""
from pathlib import Path
import tempfile
import unittest
import importlib.util
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'assets').is_dir())


class FullEvaluationTests(unittest.TestCase):
    def test_independent_geometry_matches_production_and_hand_values(self):
        from tools.issue251.full_evaluation import fixed_squared_numpy
        path=ROOT/'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/model/geometry.py'
        spec=importlib.util.spec_from_file_location('full_eval_geometry',path)
        geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
        x=np.array([[0.,0.],[.3,-.2],[.3,-.2]])
        y=np.array([[.5,0.],[-.1,.4],[0.,0.]])
        actual=fixed_squared_numpy(x,y,scale=1.3)
        expected=geometry.fixed_squared(torch.from_numpy(x),torch.from_numpy(y),scale=1.3).numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)
        self.assertAlmostEqual(fixed_squared_numpy(np.zeros((1,2)),np.array([[.5,0.]]))[0,0],.25)
        np.testing.assert_array_equal(actual[1],actual[2])
        np.testing.assert_allclose(fixed_squared_numpy(x,x).diagonal(),0.,atol=1e-14)

    def test_full_stream_counts_masks_macro_and_micro(self):
        from tools.issue251.full_evaluation import full_evaluate
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            root=Path(directory);(root/'shards').mkdir()
            np.savez(root/'node_moments.npz',component=np.zeros(4,dtype=int))
            np.savez(root/'index.npz',shard_names=np.array(['tiny.npz']))
            np.savez(root/'shards/tiny.npz',centre=[1,1,1,2,2],b=[2,2,2,1,1],c=[3,3,3,3,3],
                     curvature=[-.5,-.5,-.5,5.,999.],valid=np.array([1,1,1,1,0],dtype=bool),
                     bc_dist_sq=[4.,4.,4.,4.,4.])
            def euclidean(x,y,scale=1.):return ((x[:,None]-y[None,:])/scale)**2
            def squared(x,y,scale=1.):return euclidean(x,y,scale).sum(-1)
            paired=lambda x,y,scale=1.:(((x-y)/scale)**2).sum(-1)
            report=full_evaluate(np.array([[0.],[1.],[3.],[5.]]),root,metric=squared,
                                 paired_metric=paired,chunk=2)
            self.assertEqual(report['curvature']['count'],4)
            self.assertEqual(report['curvature']['excluded'],1)
            self.assertAlmostEqual(report['curvature']['micro_mae'],2.)
            self.assertAlmostEqual(report['curvature']['macro_mae'],3.)
            self.assertEqual(set(report['curvature']['per_item']),{'1','2'})
            self.assertAlmostEqual(report['curvature']['sign']['balanced_accuracy'],.5)
            self.assertAlmostEqual(report['distance_distortion']['mean_abs_error'],.5)
            self.assertEqual(report['reference_scope'],'entire frozen component')
            np.savez(root/'node_moments.npz',component=np.array([0,0,0,1]))
            with self.assertRaisesRegex(ValueError,'crosses'):
                full_evaluate(np.array([[0.],[1.],[3.],[5.]]),root,metric=squared,
                              paired_metric=paired,chunk=2)


if __name__=='__main__':unittest.main()
