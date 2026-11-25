"""Hand-computed expectations: the numbers are written down, not recomputed at run time."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.iteration import analysis, registry, timeline


def test_sid_chain_known_answers():
    # Six items: L1 codes 0,0,0,0,1,1 -> p=(4/6,2/6); pairs 00,00,01,02,10,10 -> (2,1,1,2)/6;
    # triples all distinct except none -> six distinct.
    rows = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0]]
    out = analysis.sid_chain(rows)
    assert out['items'] == 6
    assert out['distinct_L1'] == 2
    assert out['distinct_L2'] == 4
    assert out['distinct_triples'] == 6
    expected_h1 = -(4 / 6 * math.log2(4 / 6) + 2 / 6 * math.log2(2 / 6))
    expected_h12 = -(2 / 6 * math.log2(2 / 6)) * 2 - (1 / 6 * math.log2(1 / 6)) * 2
    expected_h123 = math.log2(6)
    assert out['H_L1'] == pytest.approx(expected_h1, abs=1e-6)
    assert out['H_L2_given_L1'] == pytest.approx(expected_h12 - expected_h1, abs=1e-6)
    assert out['H_L3_given_L1L2'] == pytest.approx(expected_h123 - expected_h12, abs=1e-6)
    assert out['items_per_triple'] == pytest.approx(1.0, abs=1e-9)
    assert out['collision'] == pytest.approx(0.0, abs=1e-9)


def test_sid_chain_degenerate_inputs():
    single = analysis.sid_chain([[3, 4, 5, 0]])
    assert single['H_L1'] == 0.0 and single['H_triple'] == 0.0
    assert single['items_per_triple'] == 1.0
    twice = analysis.sid_chain([[1, 2, 3, 0], [1, 2, 3, 1]])
    assert twice['distinct_triples'] == 1 and twice['collision'] == pytest.approx(0.5)
    assert twice['H_L3_given_L1L2'] == 0.0


def test_role_conditioning_known_answers():
    rows = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 0]]
    table = [5, 5, 6, 7]
    out = analysis.role_conditioning(rows, table)
    assert out['available'] is True
    assert out['prefixes'] == 3
    # prefix (0,0) has roles {5} -> 1 role; items per prefix: (0,0)->2, (0,1)->1, (1,1)->1
    assert out['roles_per_prefix']['max'] == 1
    assert out['items_per_prefix']['mean'] == pytest.approx(4 / 3, abs=1e-4)
    # cells: ((0,0),5) has 2 items and 2 distinct L3; ((0,1),6) and ((1,1),7) are singletons
    assert out['prefix_role_cells'] == 3
    assert out['distinct_L3_per_cell']['mean'] == pytest.approx(4 / 3, abs=1e-4)


def test_timeline_merge_and_exclusive(tmp_path):
    line = timeline.Timeline(tmp_path / 't.jsonl')
    line.start('analysis', resource='cpu')
    line.stop('analysis')
    summary = line.summary()
    assert summary['phases']['analysis']['spans'] == 1
    assert summary['open_phases'] == []
    spans = [{'start': 0.0, 'stop': 2.0}, {'start': 1.0, 'stop': 3.0}, {'start': 5.0, 'stop': 6.0}]
    assert timeline.merged(spans) == pytest.approx(4.0)
    assert timeline.merged([]) == 0.0
    assert timeline.merged([{'start': 2.0, 'stop': 1.0}]) == 0.0


def test_registry_transitions_and_refusals(tmp_path):
    rnd = registry.Round.register(tmp_path, 'r99', arm='A', run_id='run99',
                                  contract_sha256='0' * 64, hypothesis='h')
    assert rnd.state == 'REGISTERED'
    with pytest.raises(ValueError):
        rnd.advance('ADMITTED')          # skips VALIDATED
    rnd.advance('VALIDATED', evidence={'plan': 'p.json'})
    rnd.advance('ADMITTED')
    rnd.advance('STAGE2_RUNNING')
    rnd.advance('STAGE2_RUNNING')        # a resumed running phase may re-enter itself
    assert rnd.evidence('plan') == 'p.json'
    with pytest.raises(ValueError):
        registry.Round.register(tmp_path, 'r99', arm='A', run_id='run99',
                                contract_sha256='0' * 64, hypothesis='h')
