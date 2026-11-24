"""Reject stale, partial or falsified calibration evidence, not source text."""
import copy
import pytest
from tools.issue256.r02.calibration_receipts import validate


def records():
    return [{'calibration_epoch':e,'calibration_step':(e-1)*6,
        'calibration_world_size':4,'calibration_batch_items':1024,'lambda':0.5,
        'module_budgets':{'module%d'%g:{'original_gradient':1.,'curvature_gradient':2.,
            'weighted_ratio':1.} for g in range(17)}} for e in (1,51,101)]


def test_complete_current_block_evidence_matches_final_weight():
    validate(records(),epochs=101,interval=50,steps_per_epoch=6,checkpoint_weight=0.5)


@pytest.mark.parametrize('fault', ['missing_anchor','wrong_step','stale_checkpoint','false_ratio','nonmaximal','nan'])
def test_invalid_block_evidence_cannot_admit_a_run(fault):
    rows=records()
    weight=0.5
    if fault=='missing_anchor':
        rows.pop(1)
    elif fault=='wrong_step':
        rows[1]['calibration_step']=0
    elif fault=='stale_checkpoint':
        weight=0.25
    elif fault=='false_ratio':
        rows[1]['module_budgets']['module3']['weighted_ratio']=0.0
    elif fault=='nonmaximal':
        rows[1]['lambda']=0.25
        for row in rows[1]['module_budgets'].values():
            row['weighted_ratio']=0.5
    else:
        rows[1]['module_budgets']['module3']['original_gradient']=float('nan')
    with pytest.raises(ValueError,match={'missing_anchor':'complete native anchors','wrong_step':'actual first step',
        'stale_checkpoint':'last anchor coefficient','false_ratio':'recorded module ratio',
        'nonmaximal':'maximal module budget','nan':'finite nonnegative'}[fault]):
        validate(rows,epochs=101,interval=50,steps_per_epoch=6,checkpoint_weight=weight)
