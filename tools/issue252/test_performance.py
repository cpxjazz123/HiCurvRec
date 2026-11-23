"""Inclusive speed acceptance cannot hide periodic work between epoch timers."""
import unittest
from tools.issue252.performance import summarize_window


class PerformanceTests(unittest.TestCase):
    def test_interepoch_structure_gap_is_included(self):
        records=[{'epoch':i+1,'start':i*.2+(12 if i>=15 else 0),
                  'end':(i+1)*.2+(12 if i>=15 else 0)} for i in range(30)]
        result=summarize_window(records)
        self.assertAlmostEqual(result['window_seconds'],18.)
        self.assertAlmostEqual(result['mean_seconds'],.6)
        self.assertAlmostEqual(result['median_seconds'],.2)
        self.assertFalse(result['passed'])

    def test_full_consecutive_thirty_epochs_pass_and_gaps_refuse(self):
        records=[{'epoch':i+1,'start':i*.4,'end':(i+1)*.4} for i in range(30)]
        result=summarize_window(records)
        self.assertAlmostEqual(result['window_seconds'],12.)
        self.assertAlmostEqual(result['mean_seconds'],.4)
        self.assertAlmostEqual(result['p95_seconds'],.4)
        self.assertTrue(result['passed'])
        self.assertFalse(summarize_window(records[:29])['passed'])
        records[10]['epoch']=99
        with self.assertRaisesRegex(ValueError,'consecutive'):
            summarize_window(records)


if __name__=='__main__':unittest.main()
