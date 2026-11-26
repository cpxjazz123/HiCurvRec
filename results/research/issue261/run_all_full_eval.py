"""Sequential persisted full-evaluation driver for issue #261."""
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(REPO))
from tools.iteration.execution import run
PYBIN='/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
ARMS=[(100,'TIGER_gnn_attention_sectional_rq'),(101,'TIGER_three_expert_attention_rq'),(102,'TIGER_module_budget_prefix_rq'),(104,'TIGER_role_conditioned_l1_rq'),(105,'TIGER_role_conditioned_l3_rq'),(106,'TIGER_role_conditioned_l2_rq'),(107,'TIGER_topology_residual_l2_rq')]
for n,arm in ARMS:
 argv=[PYBIN,'results/research/issue261/run_full_eval.py',arm]
 if n<=102: argv.append('--allow-infra-drift')
 r=run(f'issue261-full-eval-{n}',argv,out_dir='results/research/issue261',timeout=1800,note=f'full 3-expert curvature/reconstruction heldout arm {n}')
 print(n,arm,r.status,r.exit_code,r.duration_s,r.log,flush=True)
