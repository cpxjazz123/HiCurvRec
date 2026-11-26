"""Use the existing export/archive audit interfaces for the new Stage2 result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools.issue252.archive_audit import create_binding, sha256
from tools.research.stage2 import export_check
ARM = "TIGER_relative_curvature_ranking_rq"
RUNS = sorted((ROOT / "results/stage2_RQ-VAE/_runs").glob(f"*/{ARM}/result/rqvae_best.pth"),
              key=lambda path: path.stat().st_mtime)
if not RUNS:
    raise RuntimeError("no Stage2 checkpoint")
result = RUNS[-1].parent
tree = result.parent / "src"
sid = result / "item_sids.json"
checkpoint = result / "rqvae_best.pth"
manifest = export_check.check(tree, checkpoint, sid, rewrite=True)
export_path = result / "export_manifest.json"
export_check.write_manifest(manifest, export_path)
binding_path = result / "archive_binding.json"
receipt = create_binding(tree, checkpoint, sid, export_path, output=binding_path)
report = {"schema": "genrec.issue267.export_finalize/1",
          "run": result.parent.name, "tree": str(tree), "result": str(result),
          "manifest": manifest, "manifest_sha256": sha256(export_path),
          "binding_sha256": sha256(binding_path),
          "binding_source": receipt.get("source_tree"), "checkpoint_sha256": sha256(checkpoint),
          "sid_sha256": sha256(sid)}
(ROOT / "results/research/issue267/analysis/EXPORT_FINALIZE.json").write_text(
    json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("run", "manifest_sha256", "binding_sha256",
                                         "checkpoint_sha256", "sid_sha256")}, indent=2))
