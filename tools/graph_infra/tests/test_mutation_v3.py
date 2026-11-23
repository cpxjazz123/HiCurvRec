"""Each live mutation must fail its designated numerical/isolation assertion.

The captured assertion text is written to results/research/issue247/MUTATIONS.json so the
mutation -> failing assertion mapping is machine-readable evidence, not prose.
"""
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from tools.graph_infra import sectional, interfaces_v3
from tools.graph_infra.tests.test_sectional_v3 import graph

OUT = Path(__file__).resolve().parents[3] / 'results/research/issue247/MUTATIONS.json'


def star_value():
    return sectional.estimate(graph(4,[(0,1),(0,2),(0,3)]),np.array([[0,1,2]]),np.arange(4))['value'][0]


def path_value():
    return sectional.estimate(graph(3,[(0,1),(1,2)]),np.array([[1,0,2]]),np.arange(3))['value'][0]


def triangle_value():
    return sectional.estimate(graph(4,[(0,1),(1,2),(2,0)]),np.array([[0,1,2]]),np.arange(4))['value'][0]


class MutationTests(unittest.TestCase):
    def test_mutations(self):
        original = sectional.contributions
        records = []

        def capture(name, category, mutant, assertion, message):
            assertion()
            with patch.object(sectional,'contributions',mutant):
                try:
                    assertion()
                except AssertionError as error:
                    text = str(error)
                    self.assertIn(message, text)
                    records.append({'mutation':name,'category':category,'caught_by':message,
                                    'assertion_text':text.splitlines()[0] if text else message})
                    return
            self.fail('mutation not caught: ' + name)

        capture('sign flip of the parallelogram residual','numeric',
                lambda *a: -original(*a),
                lambda: self.assertAlmostEqual(star_value(),-.5,msg='star equals negative half'),
                'star equals negative half')
        capture('g(b,c)^2 quarter coefficient becomes three quarters','numeric',
                lambda am,ab,ac,bc: original(am,ab,ac,bc)+3*np.broadcast_to(bc,np.broadcast(am,bc).shape)**2/4,
                lambda: self.assertAlmostEqual(star_value(),-.5,msg='star equals negative half'),
                'star equals negative half')
        capture('mean over references replaced by a sum','numeric',
                lambda am,ab,ac,bc: np.broadcast_to(original(am,ab,ac,bc).sum(axis=1,keepdims=True),
                                                    original(am,ab,ac,bc).shape),
                lambda: self.assertAlmostEqual(star_value(),-.5,msg='star equals negative half'),
                'star equals negative half')
        capture('unreachable references are zero filled instead of masked','structural',
                lambda *a: np.nan_to_num(original(*a),nan=0.),
                lambda: self.assertAlmostEqual(triangle_value(),.25,msg='missing not zero filled'),
                'missing not zero filled')
        with patch.object(sectional,'endpoint_distance',lambda graph,b,c: np.ones(len(b),dtype=float)):
            try:
                self.assertAlmostEqual(path_value(),0.,msg='endpoint distance respects adjacency')
            except AssertionError as error:
                records.append({'mutation':'g(b,c) is always taken as 1 (adjacency ignored)','category':'structural',
                                'caught_by':'endpoint distance respects adjacency',
                                'assertion_text':str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: g(b,c) is always taken as 1')

        def ids_assert():
            with self.assertRaises(ValueError,msg='ID mismatch rejected'):
                interfaces_v3.validate_ids(np.array([1,0]),np.arange(2))
        ids_assert()
        with patch.object(interfaces_v3,'validate_ids',lambda *args: None):
            try:
                ids_assert()
            except AssertionError as error:
                records.append({'mutation':'item index order is not validated','category':'routing',
                                'caught_by':'ID mismatch rejected','assertion_text':str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: item index order is not validated')

        train = interfaces_v3.TrainingAssets.__new__(interfaces_v3.TrainingAssets)
        train.entries = {'strict_test':{'use':'evaluation'}}
        def route_assert():
            with self.assertRaises(ValueError,msg='hidden target rejected'):
                train.load('strict_test')
        route_assert()
        with patch.object(interfaces_v3.TrainingAssets,'load',lambda *args: {'target':1}):
            try:
                route_assert()
            except AssertionError as error:
                records.append({'mutation':'the training interface returns hidden evaluation targets','category':'routing',
                                'caught_by':'hidden target rejected','assertion_text':str(error).splitlines()[0]})
            else:
                self.fail('mutation not caught: the training interface returns hidden evaluation targets')

        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_text(json.dumps({'mutations':records,'caught':len(records)},indent=2)+'\n')


if __name__=='__main__': unittest.main()
