"""Run candidate and relevant inherited asset/framework tests, binding exact sources."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import subprocess
import time

from tools.issue256.r04.identity import ROOT, SOURCE, EVIDENCE, PYTHON
from tools.issue252.archive_audit import imported_repo_sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    files = imported_repo_sources(SOURCE, ROOT)
    for directory in (SOURCE, ROOT / 'tools/issue256/r03', ROOT / 'tools/issue255/tests'):
        for path in sorted(directory.rglob('*.py')):
            if '__pycache__' not in path.parts:
                files[str(path.relative_to(ROOT))] = digest(path)
    for path in list((SOURCE / 'config').glob('*.json')) + [
            SOURCE / 'role_table.json',
            EVIDENCE / 'CONTRACT.md', EVIDENCE / 'MUTATIONS.md', EVIDENCE / 'reference.py',
            ROOT / 'results/research/issue256/ACCEPTANCE.json',
            ROOT / 'tools/research/stage2/anticollapse/floor.py',
            ROOT / 'tools/research/stage2/anticollapse/patch_tree.py',
            ROOT / 'tools/research/stage2/tests/test_floor_history_lifetime.py',
            ROOT / 'tools/research/stage2/tests/test_floor_install_on_baseline.py',
            ROOT / 'results/research/issue256/EVALUATION.md']:
        files[str(path.relative_to(ROOT))] = digest(path)
    return files


def aggregate_suite(reports):
    """A full suite means each collected case actually passed exactly once."""
    if [r.get('rank') for r in reports] != list(range(4)):
        raise ValueError('all four sanctioned unit workers are required')
    expected=reports[0]['all_nodeids']
    passed=[]
    for row in reports:
        if row.get('exit_code') or row.get('failed') or row.get('skipped'):
            raise ValueError('failed or skipped unit case')
        if row['all_nodeids'] != expected or not row['selected_nodeids']:
            raise ValueError('inconsistent or empty complete unit collection')
        if sorted(row['selected_nodeids']) != sorted(row['passed_nodeids']):
            raise ValueError('selected unit cases did not all execute successfully')
        passed.extend(row['passed_nodeids'])
    if len(passed) != len(set(passed)):
        raise ValueError('duplicate unit case execution')
    if set(passed) != set(expected) or not expected:
        raise ValueError('executed cases differ from the complete unit collection')
    return {'passed_cases':len(passed),'complete_nodeids':expected}


def _suite(name):
    return {'candidate':((SOURCE/'tests',),SOURCE),
            'assets':((ROOT/'tools/issue255/tests',),ROOT),
            'tools':((ROOT/'tools/issue256/r03',
                      ROOT/'tools/research/stage2/tests/test_floor_history_lifetime.py',
                      ROOT/'tools/research/stage2/tests/test_floor_install_on_baseline.py'),ROOT)}[name]


def worker(name, output):
    """One isolated unit interpreter on its launcher-owned CUDA device."""
    import torch
    import pytest
    rank=int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    # Unit modules are not a DDP Trainer; preserve their original process semantics.
    # The actual device/launcher rank is recorded before isolating training env.
    device=torch.cuda.current_device()
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK'):
        os.environ.pop(key,None)
    paths,cwd=_suite(name)
    os.chdir(cwd)
    sys.path.insert(0,str(cwd))
    class Shard:
        def __init__(self):
            self.all_nodeids=[]
            self.selected_nodeids=[]
            self.passed_nodeids=[]
            self.failed=[]
            self.skipped=[]
        def pytest_collection_modifyitems(self,session,config,items):
            self.all_nodeids=[item.nodeid for item in items]
            chosen=items[rank::4]
            selected={id(item) for item in chosen}
            config.hook.pytest_deselected(items=[item for item in items if id(item) not in selected])
            items[:]=chosen
            self.selected_nodeids=[item.nodeid for item in chosen]
        def pytest_runtest_logreport(self,report):
            if report.failed:
                self.failed.append({'nodeid':report.nodeid,'phase':report.when})
            elif report.skipped:
                self.skipped.append(report.nodeid)
            elif report.when=='call' and report.passed:
                self.passed_nodeids.append(report.nodeid)
    shard=Shard()
    output=Path(output)
    args=[*(str(path) for path in paths),'-q','--basetemp='+str(EVIDENCE/('pytest_%s_rank%d'%(name,rank)))]
    code=int(pytest.main(args,plugins=[shard]))
    report={'rank':rank,'worker_pid':os.getpid(),'device_index':device,
        'exit_code':code,'pytest_args':args,'cwd':str(cwd),
        'all_nodeids':shard.all_nodeids,'selected_nodeids':shard.selected_nodeids,
        'passed_nodeids':shard.passed_nodeids,'failed':shard.failed,'skipped':shard.skipped,
        'qualification':'Independent unit cases, not a DDP training or performance measurement.'}
    output.mkdir(parents=True,exist_ok=True)
    (output/('%s_rank%d.json'%(name,rank))).write_text(json.dumps(report,indent=2)+'\n')
    return code


def child():
    """All CUDA unit probes stay beneath the sole four-card launcher."""
    output=Path.cwd()/'unit_suites'
    output.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ['RANK'])
    failed=False
    for name in ('candidate','assets','tools'):
        code=('import sys; from tools.issue256.r04.run_tests import worker; '
              'raise SystemExit(worker(sys.argv[1],sys.argv[2]))')
        command=[PYTHON,'-c',code,name,str(output)]
        done=subprocess.run(command,cwd=str(ROOT),capture_output=True,text=True)
        (output/('%s_rank%d.log'%(name,rank))).write_text(done.stdout+'\n'+done.stderr)
        failed |= done.returncode != 0
        print('unit',name,'rank',rank,'exit',done.returncode,flush=True)
    return int(failed)


def main():
    from tools.research.stage2.launcher import launch
    before=source_inventory()
    started=time.monotonic()
    result=EVIDENCE/'test_launcher'/('%d_%d'%(os.getpid(),time.time_ns()))
    code=launch(SOURCE/'unit_suite_entry.py',result,python=PYTHON,log_name='tests.log')
    results,failed={},[]
    for name in ('candidate','assets','tools'):
        paths=[result/'unit_suites'/('%s_rank%d.json'%(name,rank)) for rank in range(4)]
        logs=[result/'unit_suites'/('%s_rank%d.log'%(name,rank)) for rank in range(4)]
        combined=EVIDENCE/('TEST_'+name.upper()+'.log')
        combined.write_text('\n'.join(path.read_text() for path in logs if path.is_file()))
        try:
            reports=[json.loads(path.read_text()) for path in paths]
            coverage=aggregate_suite(reports)
            suite={'exit_code':0,'summary':['%d passed across4 launcher-owned unit workers; complete coverage, no skips'%coverage['passed_cases']],
                'coverage':coverage,'ranks':[{'path':str(path),'sha256':digest(path)} for path in paths]}
        except (ValueError,FileNotFoundError) as error:
            failed.append(name)
            suite={'exit_code':1,'summary':[str(error)]}
        suite.update(command=[PYTHON,'-m','tools.issue256.r04.run_tests'],
            log={'path':str(combined),'sha256':digest(combined)})
        results[name]=suite
        print(name,suite['exit_code'],suite['summary'][-1],flush=True)
    after=source_inventory()
    if before!=after:
        failed.append('source_changed')
    if code:
        failed.append('launcher')
    report={'schema':'genrec.issue256.test_results/2','passed':not failed,
        'failed_suites':failed,'suites':results,'sources_sha256':after,
        'launcher':{'entry':str(SOURCE/'unit_suite_entry.py'),'result':str(result),
            'exit_code':code,'world_size':4,'scope':'Unit case execution only; not a four-rank Trainer proof.'},
        'wall_seconds':time.monotonic()-started}
    (EVIDENCE/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
