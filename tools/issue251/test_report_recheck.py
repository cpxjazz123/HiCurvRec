"""Consumer-visible headline mistakes: centre weighting and fourth-token collisions."""
import unittest
from tools.issue251.recompute_report import curvature_headlines, sid_headlines


class ReportRecheckTests(unittest.TestCase):
    def test_macro_is_not_triple_weighted_micro(self):
        result=curvature_headlines({'a':{'count':1,'mean_abs_error':4.},
                                   'b':{'count':3,'mean_abs_error':2.}})
        self.assertEqual(result,{'count':4,'centres':2,'macro_mae':3.,'micro_mae':2.5})

    def test_collision_ignores_unique_export_counter(self):
        result=sid_headlines([[0,0,0,0],[0,0,0,1],[1,2,3,0]])
        self.assertEqual(result['L3']['distinct'],2)
        self.assertAlmostEqual(result['L3']['collision'],1./3)
        self.assertAlmostEqual(result['L3']['sharing'],2./3)
        self.assertEqual(result['L3']['used'],2)


if __name__=='__main__':unittest.main()
