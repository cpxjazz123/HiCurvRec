"""Changed round controllers must invalidate old test credentials."""
from pathlib import Path
import pytest
from tools.issue256.r04 import run_tests,dispatch_evidence


def test_changed_round_adapter_invalidates_tested_source(tmp_path,monkeypatch):
    repo=tmp_path/'repo'
    source=repo/'stage2_RQ-VAE/arm'
    evidence=repo/'results/research/issue256/rounds/r04'
    helper=repo/'tools/issue256/r04/controller.py'
    for path,text in [(source/'train_rqvae.py','VALUE=1\n'),
        (source/'role_table.json','{}'),
        (evidence/'CONTRACT.md','contract'),(evidence/'MUTATIONS.md','mutations'),
        (evidence/'reference.py','VALUE=1\n'),(helper,'VALUE=1\n'),
        (repo/'results/research/issue256/ACCEPTANCE.json','{}'),
        (repo/'tools/research/stage2/anticollapse/floor.py','VALUE=1\n'),
        (repo/'tools/research/stage2/anticollapse/patch_tree.py','VALUE=1\n'),
        (repo/'tools/research/stage2/tests/test_floor_history_lifetime.py','VALUE=1\n'),
        (repo/'tools/research/stage2/tests/test_floor_install_on_baseline.py','VALUE=1\n'),
        (repo/'results/research/issue256/EVALUATION.md','evaluation')]:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(run_tests,'ROOT',repo)
    monkeypatch.setattr(run_tests,'SOURCE',source)
    monkeypatch.setattr(run_tests,'EVIDENCE',evidence)
    old=run_tests.source_inventory()
    helper.write_text('VALUE=2\n')
    with pytest.raises(ValueError,match='source changed: tools/issue256/r04/controller.py'):
        dispatch_evidence.verify(repo,old)
