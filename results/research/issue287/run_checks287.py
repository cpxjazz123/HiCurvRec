"""Issue #287: run the machine admission checks on the new candidate tree."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from tools.research.stage2 import checks

import sys as _s
tree = Path(_s.argv[1]) if len(_s.argv) > 1 else Path('stage2_RQ-VAE/TIGER_l1_pairrank_rq')
report = checks.run_all(tree)
extra = {}
for name in ('check_constants', 'check_names', 'check_four_rank', 'check_collectives',
             'check_geometry_contract', 'check_assignment_rule', 'check_encoder_gradient',
             'check_assignment_floor', 'check_identifier'):
    try:
        extra[name] = getattr(checks, name)(tree)
    except Exception as exc:  # a diagnostic that cannot run is recorded, never hidden
        extra[name] = {'status': 'ERROR', 'name': name, 'detail': repr(exc)}
out = {'schema': 'genrec.issue287.checks/1', 'tree': str(tree), 'run_all': report, 'diagnostics': extra}
(Path('results/research/issue287') / (tree.name + '_CHECKS.json')).write_text(json.dumps(out, indent=1, default=str) + '\n')
print(json.dumps(out, indent=1, default=str)[:2500])
