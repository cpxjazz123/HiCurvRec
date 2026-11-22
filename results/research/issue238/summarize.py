#!/usr/bin/env python3
"""Print the one table the report ends with: Stage2 diagnostics + Stage3 recall per arm."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STAGE2_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
ARMS = [
    ("TIGER_RQ-VAE", "baseline (Euclidean, T5)"),
    ("TIGER_g237_struct_rq", "structure replaces the input"),
    ("TIGER_g237_sem_rq", "structure as residual"),
    ("TIGER_kappa_cost_rq", "curved cost, all three levels"),
    ("TIGER_kappa_sk_only_rq", "curved cost, Sinkhorn level only"),
    ("TIGER_af3_graphgeo_rq", "per-item geometry from AF3 curvature (19h coupling)"),
]


def sid_quality(arm: str) -> dict:
    """The stage-3 gate's own measurement: collision, quantisation error, effective rank."""
    import subprocess
    sid = REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"
    if not sid.is_file():
        return {}
    done = subprocess.run(
        [STAGE2_PYTHON, str(REPO / "tools" / "research" / "stage2" / "sid_quality.py"),
         str(sid), "--waive-collision"],
        capture_output=True, text=True, timeout=1800)
    if done.returncode != 0 or not done.stdout.strip():
        return {}
    return json.loads(done.stdout.strip().splitlines()[-1])


def stage2(arm: str) -> dict:
    manifest = REPO / "results" / "stage2_RQ-VAE" / arm / "export_manifest.json"
    out: dict = {}
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        out["differing"] = data.get("original_differing_items")
    metrics = sorted((REPO / "results" / "stage2_RQ-VAE" / arm / "stage2_metrics").glob("*.json"))
    if metrics:
        data = json.loads(metrics[-1].read_text(encoding="utf-8"))
        diag = data.get("diagnostics") or {}
        if "l3_collision" in diag:
            out["recorded_collision"] = diag["l3_collision"]
    out.update({k: v for k, v in sid_quality(arm).items()
                if k in ("collision", "quant_error", "effective_rank")})
    return out


def stage3(arm: str):
    hits = sorted((REPO / "results" / "stage3_T5Train" / arm).glob("**/test_final.json"))
    if not hits:
        return None
    return json.loads(hits[-1].read_text(encoding="utf-8")).get("test_recall@10")


def main() -> int:
    print(f"{'arm':28s} {'collision':>10s} {'quant':>8s} {'rank':>6s} {'diff':>6s} {'R@10':>8s}  note")
    for arm, note in ARMS:
        s2, r10 = stage2(arm), stage3(arm)
        cols = [
            f"{arm:28s}",
            f"{s2.get('collision', float('nan')):10.4f}",
            f"{s2.get('quant_error', float('nan')):8.3f}",
            f"{s2.get('effective_rank', float('nan')):6.2f}",
            f"{s2.get('differing', -1):6d}",
            f"{r10:8.5f}" if r10 is not None else f"{'pending':>8s}",
        ]
        print(" ".join(cols) + f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
