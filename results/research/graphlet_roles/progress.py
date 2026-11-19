"""Compact progress line for one arm of issue #217."""
import json
import sys
from pathlib import Path

arm = sys.argv[1] if len(sys.argv) > 1 else "Q"
path = Path(f"results/stage2_RQ-VAE/TIGER_graphlet_role_{arm}/logs/training_metrics.jsonl")
if not path.is_file():
    print(f"{arm}: no metrics yet")
    raise SystemExit
rows = [json.loads(line) for line in path.open()]
train = [r for r in rows if r.get("event") == "train"]
if not train:
    print(f"{arm}: {len(rows)} metric rows, no train rows yet")
    raise SystemExit
last = train[-1]
end = [r for r in rows if r.get("event") == "train_end"]
eta = (3000 - last["epoch"]) * last["epoch_time_s"] / 60
print(f"{arm} epoch {last['epoch']}/3000 aux {last.get('aux') or 0:.3f} "
      f"w {last.get('aux_weight') or 0:.2e} eta {eta:.1f} min ended={bool(end)}"
      + (f" collision={end[-1]['final_collision']:.4f}" if end else ""))
