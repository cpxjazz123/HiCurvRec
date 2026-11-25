"""Full1,024-item real-loss regression on four launcher-owned GPUs, no training."""
from pathlib import Path
import faulthandler
import hashlib
import json
import os
import sys
import traceback

import torch

TREE = Path(__file__).resolve().parent
ROOT = next(p for p in TREE.parents if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(TREE))
sys.path.insert(1, str(TREE / 'tests'))


def main():
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ.get('WORLD_SIZE', 0)) != 4:
        raise RuntimeError('this real-loss regression requires the sanctioned four-card launcher')
    torch.cuda.set_device(rank)
    from test_module_budget import check_actual_calibration
    files = [TREE / 'model/mechanism.py', TREE / 'model/calibration.py',
             TREE / 'tests/test_module_budget.py', Path(__file__)]
    record = {'rank': rank, 'worker_pid': os.getpid(), 'batch_items': 1024,
              'formal_training': False, 'source_hashes': {
                  str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    try:
        faulthandler.dump_traceback_later(45, repeat=True)
        check_actual_calibration(torch.device('cuda', rank), 1024)
        record['status'] = 'PASS'
    except Exception:
        record['status'] = 'FAIL'
        record['error'] = traceback.format_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        output = ROOT / 'results/research/issue258/budget_regression'
        generation = hashlib.sha256(json.dumps(record['source_hashes'], sort_keys=True).encode()).hexdigest()
        output = output / generation
        output.mkdir(parents=True, exist_ok=True)
        (output / ('rank_%d.json' % rank)).write_text(json.dumps(record, indent=2) + '\n')
        print('budget regression', rank, record['status'], flush=True)


if __name__ == '__main__':
    main()
