"""Run all candidate suites once plus relevant framework regressions with bound receipts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from tools.issue252.training_admission import source_binding

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools/research/stage2/contract.py').is_file())
TREE = ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq'
OUTPUT = ROOT/'results/research/issue253/contract_tests'
COMMAND = [sys.executable, '-m', 'tools.issue253.run_contract_tests']

def main():
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [str(TREE/'tests'), str(TREE), str(ROOT)]
    directory = OUTPUT/str(time.time_ns()); directory.mkdir(parents=True)
    binding = source_binding(ROOT, TREE, 'validation')
    candidate = [p.stem for p in sorted((TREE/'tests').glob('test_*.py'))
                 if p.name not in ('test_reference.py', 'test_mutations.py', 'test_sync_free_batch.py')]
    framework = ['tools.issue252.test_training_admission', 'tools.issue252.test_run_workspace',
                 'tools.issue252.test_archive_binding', 'tools.issue252.test_final_input_generation']
    own = ['tools.issue253.'+p.stem for p in sorted((ROOT/'tools/issue253').glob('test_*.py'))
           if p.name != 'test_estimator_mutations.py']
    groups = {'deterministic': candidate+own+framework,
              'reference': ['test_reference'],
              'mutation': ['test_mutations', 'tools.issue253.test_estimator_mutations']}
    records = {}; passed = True
    for kind, names in groups.items():
        text = io.StringIO()
        result = unittest.TextTestRunner(stream=text, verbosity=2).run(
            unittest.TestLoader().loadTestsFromNames(names))
        output = text.getvalue(); print(output, end='')
        log = directory/(kind+'.log'); log.write_text(output)
        unchanged = binding == source_binding(ROOT, TREE, 'validation')
        okay = result.wasSuccessful() and result.testsRun > 0 and not result.skipped and unchanged
        report = {'schema': 'genrec.issue253.tests/1', 'kind': kind,
                  'status': 'pass' if okay else 'fail', 'tests_run': result.testsRun,
                  'failures': len(result.failures), 'errors': len(result.errors), 'skipped':len(result.skipped),
                  'source_hashes': binding, 'command': COMMAND, 'exit_code': 0 if okay else 1,
                  'output': {'path':str(log.relative_to(ROOT)), 'sha256':hashlib.sha256(log.read_bytes()).hexdigest()},
                  'source_unchanged':unchanged}
        if kind == 'mutation':
            report['mutations'] = []
            for name in ('MUTATIONS.json','ESTIMATOR_MUTATIONS.json'):
                for row in json.loads((ROOT/'results/research/issue253'/name).read_text()):
                    report['mutations'].append({**row,'caught':row.get('caught',row.get('caught_by_assertion',False))})
        path = directory/(kind+'.json'); path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        records[kind]=str(path.relative_to(ROOT)); passed &= okay
    print(json.dumps({'passed':passed,'test_reports':records},indent=2))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
