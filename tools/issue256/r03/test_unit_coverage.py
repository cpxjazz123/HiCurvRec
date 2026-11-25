"""Incomplete or duplicate execution cannot count as a full unit suite."""
import pytest
from tools.issue256.r03.run_tests import aggregate_suite


def reports():
    nodes=['case_a','case_b','case_c','case_d','case_e']
    return [{'rank':rank,'exit_code':0,'all_nodeids':nodes,
        'selected_nodeids':nodes[rank::4],'passed_nodeids':nodes[rank::4],
        'failed':[],'skipped':[]} for rank in range(4)]


def test_disjoint_execution_reconciles_the_complete_collected_suite():
    result=aggregate_suite(reports())
    assert result['passed_cases']==5
    assert result['complete_nodeids']==['case_a','case_b','case_c','case_d','case_e']


@pytest.mark.parametrize('fault', ['missing','duplicate','skipped'])
def test_incomplete_execution_cannot_issue_a_full_suite_credential(fault):
    rows=reports()
    if fault=='missing':
        rows[0]['selected_nodeids']=['case_a']
        rows[0]['passed_nodeids']=['case_a']
    elif fault=='duplicate':
        rows[1]['selected_nodeids']=rows[1]['selected_nodeids']+['case_e']
        rows[1]['passed_nodeids']=rows[1]['passed_nodeids']+['case_e']
    else:
        rows[2]['skipped']=['case_c']
    with pytest.raises(ValueError,match={'missing':'complete unit collection','duplicate':'duplicate unit case',
        'skipped':'failed or skipped unit case'}[fault]):
        aggregate_suite(rows)
