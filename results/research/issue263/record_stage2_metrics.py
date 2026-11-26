"""Create the mandatory §63 layer record for the immutable Stage2 SID.

This script only reads the exported SID, its checkpoint and the frozen validation
parquet. It never reads a Stage3 result. It first runs the framework export gate
so the SID is the checkpoint's own table and the archive binding exists.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SID = (ROOT / "results/stage2_RQ-VAE/_runs/"
       "stage2_ae668bc04dc44d52a8bed5f9408e8e74/"
       "TIGER_frozen_topology_gnn_rq/result/item_sids.json")


def main():
    # The MCP server is a standalone module, not a package.
    import importlib.util, sys
    shared = ROOT / "tools/genrec-mcp/server.py"
    spec = importlib.util.spec_from_file_location("genrec_mcp_server", shared)
    module = importlib.util.module_from_spec(spec)
    sys.modules["genrec_mcp_server"] = module
    spec.loader.exec_module(module)
    gate = module.stage3_export_gate(SID)
    print("export_gate", json.dumps(gate, ensure_ascii=False))
    if not gate.get("allow"):
        raise SystemExit("export gate refused")
    from tools.issue252.layer_metrics import record_layers
    record = record_layers(SID, repo=ROOT)
    print("layer_record", json.dumps({k: record[k] for k in ("path", "sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
