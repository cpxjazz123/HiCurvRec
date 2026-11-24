"""Hand answers and independent anchor enumeration for the actual refresh route."""
import importlib.util
from pathlib import Path
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools').is_dir())
SPEC = importlib.util.spec_from_file_location('periodic_budget_reference',
    ROOT/'results/research/issue256/rounds/r02/reference.py')
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def test_anchor_hand_answers():
    from model.calibration import recalibration_due
    epochs = (1, 2, 49, 50, 51, 52, 100, 101, 2950, 2951, 3000)
    actual = [recalibration_due(e, 50) for e in epochs]
    assert actual == [False, False, False, False, True, False, False, True, False, True, False], 'refresh must precede the first batch of each next native block'


def test_complete_native_schedule_matches_independent_progression():
    from model.calibration import recalibration_due
    actual = (1,) + tuple(e for e in range(2, 3001) if recalibration_due(e, 50))
    assert actual == REFERENCE.reference_epochs(3000, 50), 'all60 native block anchors must be present without duplicate initialization'
    assert actual[-1] == 2951


@pytest.mark.parametrize('epoch,interval', [(0,50),(-1,50),(1,0),(1,-2),(1.5,50),(1,2.5),(True,50)])
def test_ambiguous_schedule_fails_closed(epoch, interval):
    from model.calibration import recalibration_due
    with pytest.raises(ValueError, match='positive integers'):
        recalibration_due(epoch, interval)


def assert_anchor_receipt(records):
    assert len(records) == 2, 'actual60epoch trainer must refresh its budget at epoch51'
    assert [row['calibration_epoch'] for row in records] == [1,51], 'actual calibration epochs must be exactly1and51'
    assert [row['calibration_step'] for row in records] == [0,300], 'each anchor must use its actual first structural step'


@pytest.mark.parametrize('mutation', ['off_by_one','skip_refresh','repeat_initial'])
def test_cadence_mutations_fail_the_designated_assertion(mutation):
    records = [{'calibration_epoch':1,'calibration_step':0}, {'calibration_epoch':51,'calibration_step':300}]
    if mutation == 'off_by_one':
        records[1]['calibration_epoch'] = 50
    elif mutation == 'skip_refresh':
        records.pop()
    else:
        records[1]['calibration_step'] = 0
    message = {'off_by_one':'actual calibration epochs', 'skip_refresh':'actual60epoch trainer must refresh', 'repeat_initial':'each anchor must use its actual first structural step'}[mutation]
    with pytest.raises(AssertionError, match=message):
        assert_anchor_receipt(records)
