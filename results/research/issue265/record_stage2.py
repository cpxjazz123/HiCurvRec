"""Export gate + §63 layer record for the issue265 Stage2 SID."""
import importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
SID=(ROOT/'results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/'
     'TIGER_euclid_local_curve_supervision_rq/result/item_sids.json')
shared=ROOT/'tools/genrec-mcp/server.py'
spec=importlib.util.spec_from_file_location('genrec_mcp_server',shared)
m=importlib.util.module_from_spec(spec); sys.modules['genrec_mcp_server']=m; spec.loader.exec_module(m)
gate=m.stage3_export_gate(SID); print('gate',json.dumps(gate,ensure_ascii=False))
if not gate.get('allow'): raise SystemExit('export gate refused')
from tools.issue252.layer_metrics import record_layers
r=record_layers(SID,repo=ROOT,arm='TIGER_euclid_local_curve_supervision_rq')
print('layer',json.dumps({k:r[k] for k in ('path','sha256')},ensure_ascii=False))
