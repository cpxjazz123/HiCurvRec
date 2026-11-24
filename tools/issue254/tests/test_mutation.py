"""Every mutant must fail the assigned value assertion, not crash/import/timeout."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from tools.issue254 import contract, paired, mechanism
from tools.issue254.tests import test_known_answers as known
from tools.issue254.tests import test_mechanism as gate_tests


class MutationTests(unittest.TestCase):
    def caught(self, target, replacement, check, message):
        with patch(target, replacement):
            with self.assertRaisesRegex(AssertionError, message):
                check()

    def test_sign_flip(self):
        original = contract.net_change
        self.caught('tools.issue254.contract.net_change', lambda c,n:-original(c,n),
                    known.KnownAnswers().test_four_groups_signed_delta, 'improvements minus regressions')

    def test_item_denominator(self):
        original = contract.per_item_contributions
        def mutant(rows,k):
            result = original(rows,k)
            for row in result:
                row['net_contribution'] = (row['improvements']-row['regressions'])/row['users']
            return result
        self.caught('tools.issue254.contract.per_item_contributions',mutant,
                    known.KnownAnswers().test_population_denominator_and_repeated_target,'denominator is global N')

    def test_rank_off_by_one(self):
        self.caught('tools.issue254.contract.truncated_rank',lambda r,saved=10:11 if r is None else r,
                    known.KnownAnswers().test_rank_is_one_based_display_and_censored,'one-based')

    def test_fourth_suffix_misrouted_as_geometry(self):
        def mutant(a,b):
            count=0
            for x,y in zip(a,b):
                if x!=y: break
                count+=1
            return count
        self.caught('tools.issue254.contract.prefix_levels',mutant,
                    known.KnownAnswers().test_prefix_ignores_collision_suffix,'fourth token is not a geometry level')

    def test_identity_route_bypass(self):
        original=paired.build_ledger
        def mutant(left,right,samples):
            left,right=copy.deepcopy(left),copy.deepcopy(right)
            for arm in (left,right):
                for row in arm['rows']:
                    source=next(s for s in samples if s['global_row']==row['global_row'])
                    row['user']=source['user']
                    row['sample_sha256']=source['sample_sha256']
            return original(left,right,samples)
        self.caught('tools.issue254.paired.build_ledger',mutant,
                    known.KnownAnswers().test_identity_mismatch_rejected,'ValueError not raised')

    def test_missing_curvature_mask_dropped(self):
        original=mechanism.join_curvature
        def mutant(values,ids):
            mae,accuracy,count=original(values,ids)
            mae[np.isnan(mae)]=0
            return mae,accuracy,count
        self.caught('tools.issue254.tests.test_mechanism.join_curvature',mutant,
                    gate_tests.MechanismTests().test_curvature_explicit_identity_missing,
                    'missing curvature mask must remain undefined')


if __name__ == '__main__':
    unittest.main()
