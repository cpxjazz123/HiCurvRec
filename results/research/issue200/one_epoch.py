"""Run the F0B trainer for one epoch locally, with every output redirected into this folder, to see what the smoke driver could not.

No hyperparameter is changed except the epoch count (the driver does the same), and the redirect is verified before running: the first version of this pattern in
#194 wrote into a frozen arm's result directory by accident, and that is not repeating.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_f0b_rq"


def main() -> int:
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location("f0b_trainer", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["f0b_trainer"] = module
    spec.loader.exec_module(module)

    out = HERE / "one_epoch_run"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    redirects = {"RESULT_DIR": out, "LOG_DIR": out / "logs", "OUTPUT_SID": out / "sids_for_hgrec.npy",
                 "OUTPUT_JSON": out / "item_sids.json", "CHECKPOINT": out / "rqvae_best.pth",
                 "CHECKPOINT_PATH": out / "rqvae_best.pth", "METRICS_PATH": out / "logs" / "training_metrics.jsonl"}
    module.EPOCHS = 1
    for attr, value in redirects.items():
        if hasattr(module, attr):
            setattr(module, attr, value)
    for attr in redirects:
        resolved = getattr(module, attr, None)
        if resolved is not None:
            resolved = Path(resolved).resolve()
            if HERE.resolve() not in resolved.parents:
                raise SystemExit(f"refusing to run: {attr} points at {resolved}")

    module.main()

    rows = [json.loads(line) for line in (out / "logs" / "training_metrics.jsonl").read_text().splitlines() if line.strip()]
    print("events:", [row.get("event") for row in rows])
    print("train rows:", sum(1 for row in rows if row.get("event") == "train"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
