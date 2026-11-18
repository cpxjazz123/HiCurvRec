"""Did the monitor change training? Compare the monitored run with two unmonitored ones.

Cross-process bit-identity is not attainable on this host: `torch.use_deterministic_algorithms`
warns that cuBLAS is not deterministic, and the export audit already showed two
processes rotating 0.4% of the codeword assignments. The control is therefore a
noise floor: two unmonitored runs with the same seed measure how far apart two
identical recipes land, and the monitored run must not sit further away than that.

    python3 results/research/qda/control_compare.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ARMS = ("QDA_F0_seed42", "QDA_F0_seed42_nomonitor", "QDA_F0_seed42_nomonitor2")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_rows(arm: str) -> dict:
    metrics = RUNS / arm / "logs" / "training_metrics.jsonl"
    last = {}
    for line in metrics.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == "train":
            last = record
    return last


def state(arm: str) -> dict:
    payload = torch.load(RUNS / arm / "rqvae_best.pth", map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    return {key: value for key, value in state.items() if torch.is_tensor(value)}


def compare(a: dict, b: dict) -> dict:
    shared = [key for key in a if key in b and a[key].shape == b[key].shape]
    delta = 0.0
    weight = 0.0
    worst = 0.0
    for key in shared:
        x = a[key].double()
        y = b[key].double()
        delta += float((x - y).pow(2).sum())
        weight += float(x.pow(2).sum())
        worst = max(worst, float((x - y).abs().max()))
    return {"tensors": len(shared), "relative_l2": (delta ** 0.5) / (weight ** 0.5) if weight else None,
            "max_abs": worst}


def main() -> None:
    rows = {arm: last_rows(arm) for arm in ARMS}
    print(f"{'arm':32s} {'epoch':>6s} {'loss':>12s} {'sid_sha':>14s}")
    for arm in ARMS:
        sid = RUNS / arm / "item_sids.json"
        print(f"{arm:32s} {rows[arm].get('epoch', 0):6d} {rows[arm].get('loss', 0):12.8f} "
              f"{sha256_of(sid)[:12] if sid.is_file() else 'missing':>14s}")

    states = {arm: state(arm) for arm in ARMS if (RUNS / arm / "rqvae_best.pth").is_file()}
    pairs = [("monitor vs off-1", "QDA_F0_seed42", "QDA_F0_seed42_nomonitor"),
             ("off-1 vs off-2", "QDA_F0_seed42_nomonitor", "QDA_F0_seed42_nomonitor2"),
             ("monitor vs off-2", "QDA_F0_seed42", "QDA_F0_seed42_nomonitor2")]
    out = {"last_rows": rows, "pairs": {}}
    print()
    for label, left, right in pairs:
        if left not in states or right not in states:
            continue
        result = compare(states[left], states[right])
        out["pairs"][label] = result
        print(f"{label:20s} relative_l2={result['relative_l2']:.6e} max_abs={result['max_abs']:.3e}")

    # Per-epoch loss trajectories, if every run recorded the same epochs.
    episodes = {arm: {} for arm in ARMS}
    for arm in ARMS:
        metrics = RUNS / arm / "logs" / "training_metrics.jsonl"
        for line in metrics.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") == "train":
                episodes[arm][record["epoch"]] = record["loss"]
    common = sorted(set.intersection(*(set(episodes[arm]) for arm in ARMS)))
    if common:
        on_off = np.mean([abs(episodes["QDA_F0_seed42"][e] - episodes["QDA_F0_seed42_nomonitor"][e])
                          for e in common])
        off_off = np.mean([abs(episodes["QDA_F0_seed42_nomonitor"][e]
                               - episodes["QDA_F0_seed42_nomonitor2"][e]) for e in common])
        out["loss_trajectory"] = {"common_epochs": len(common),
                                  "mean_abs_on_vs_off": float(on_off),
                                  "mean_abs_off_vs_off": float(off_off)}
        print(f"\nloss trajectory over {len(common)} epochs: "
              f"|on-off|={on_off:.3e}  |off-off|={off_off:.3e}")

    (HERE / "control_compare.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote control_compare.json")


if __name__ == "__main__":
    main()
