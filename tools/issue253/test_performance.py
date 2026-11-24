"""Inclusive epoch distributions must include interepoch work, not just their mean."""
import unittest
from tools.issue253.performance import summarize_window


def records(gap=0.):
    rows=[];time=0.
    for epoch in range(1,31):
        if rows:time+=gap
        start=time;time+=.8 if epoch==30 else .4
        rows.append({'epoch':epoch,'start':start,'end':time})
    return rows


class PerformanceTests(unittest.TestCase):
    def test_nearest_rank_p95_and_exact_inclusive_mean(self):
        got=summarize_window(records())
        self.assertAlmostEqual(got['window_seconds'],12.4,places=12)
        self.assertAlmostEqual(got['mean_seconds'],12.4/30,places=12)
        self.assertAlmostEqual(got['p95_seconds'],.4,places=12)
        self.assertTrue(got['target_met']);self.assertTrue(got['passed'])
    def test_median_and_p95_cannot_hide_interepoch_work(self):
        got=summarize_window(records(.1))
        self.assertAlmostEqual(got['window_seconds'],15.3,places=12)
        self.assertAlmostEqual(got['mean_seconds'],.51,places=12)
        self.assertAlmostEqual(got['median_seconds'],.5,places=12,
                               msg='inclusive epoch median must retain interepoch work')
        self.assertAlmostEqual(got['p95_seconds'],.5,places=12,
                               msg='inclusive epoch P95 must retain interepoch work')
        self.assertFalse(got['target_met']);self.assertTrue(got['passed'])
    def test_incomplete_nonconsecutive_overlapping_or_nonfinite_windows(self):
        self.assertFalse(summarize_window(records()[:29])['passed'])
        for field,value in (('epoch',9),('start',-.1),('end',float('inf'))):
            bad=records();bad[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):summarize_window(bad)

if __name__=='__main__':unittest.main()
