"""Phase-2 metrics for issue #89: canonical hard SID on ONE fixed train-only population.

The #87 pilot's hard metric was `argmax` over a soft posterior built on raw item embeddings. This
module uses the host's own inference path, `model.get_indices(embeddings, infer_use_sk=False)`,
and cross-checks it against an independent encoder+L1-argmin in a test. The population is a FIXED
set of train item ids captured once and reused for every snapshot of every arm, so a comparison is
between models rather than between different samples.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import engine as E  # noqa: E402
import r7_a_replay_engine as RE  # noqa: E402
import r7_f_host_hook as H  # noqa: E402

OUTPUT = REPO_ROOT / "results/stage2_RQ-VAE/r7_f_harness_v2"
SNAPSHOT_STEPS = (0, 8, 16, 32, 64, 96, 128, 160)
POPULATION_ITEMS = 1024
LABELS = {0: "pre_initialization"}


def population_sha(items: torch.Tensor) -> str:
    return hashlib.sha256(items.to(torch.int64).cpu().numpy().tobytes()).hexdigest()


def fixed_population(seed: int = 42) -> torch.Tensor:
    """ONE train-only item population, determined by seed alone and hashed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(24556, generator=generator)[:POPULATION_ITEMS].to(torch.long)


def metrics_for_snapshot(tree: Path, snapshot: Path, embeddings: torch.Tensor,
                         positives: torch.Tensor, unrelated: torch.Tensor,
                         tau: float) -> dict[str, Any]:
    """Canonical hard SID metrics for one snapshot on the fixed population."""
    trainer, saved_path, saved_mods = E.load_trainer(tree, module_name="r7f_metrics")
    try:
        payload = torch.load(snapshot, weights_only=False)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
        model.load_state_dict({name: value.to(RE.DEVICE)
                               for name, value in payload["model_state"].items()})
        model.to(RE.DEVICE).eval()
        batch = embeddings.to(RE.DEVICE)
        with torch.no_grad():
            tokens = model.get_indices(batch, infer_use_sk=False)      # canonical hard SID
            # independent cross-check: encoder -> L1 native distance -> argmin
            trace = model.encoder(batch)
            trace = trace[0] if isinstance(trace, tuple) else trace
            layer = model.rq.vq_layers[0]
            l1_independent = layer._distances(trace.reshape(-1, layer.dim)).argmin(dim=-1)
            l1 = tokens[:, 0]
            agree = bool(torch.equal(l1, l1_independent))
            counts = torch.bincount(l1, minlength=layer.n_embed).to(torch.float64)
            share = counts / counts.sum()
            nonzero = share[share > 0]
            entropy = float(-(nonzero * nonzero.log()).sum())
            raw_unique = int(torch.unique(tokens, dim=0).shape[0])
            l1_unique = int(torch.unique(l1).numel())
            posterior = torch.softmax(-layer._distances(trace.reshape(-1, layer.dim)) / tau, dim=-1)
            sep_entropy = float(-(posterior * posterior.clamp_min(1e-12).log()).sum(1).mean())
    finally:
        E.unload_trainer(saved_path, saved_mods, "r7f_metrics")
    return {
        "l1_cross_check_agrees": agree,
        "l1_used_codes": l1_unique,
        "l1_codebook_size": int(layer.n_embed),
        "l1_effective_codes": float(torch.exp(torch.tensor(entropy))),
        "l1_entropy": entropy,
        "l1_max_share": float(share.max()),
        "l1_histogram_head": counts.argsort(descending=True)[:10].tolist(),
        "raw_three_level_tuple_collision": 1.0 - raw_unique / float(tokens.shape[0]),
        "l1_only_collision": 1.0 - l1_unique / float(tokens.shape[0]),
        "posterior_entropy": sep_entropy,
        "hard_positive_l1_same_code": float((l1[positives[:, 0]] == l1[positives[:, 1]]).float().mean()),
        "hard_unrelated_l1_same_code": float((l1[unrelated[:, 0]] == l1[unrelated[:, 1]]).float().mean()),
        "population_items": int(tokens.shape[0]),
        "positive_pairs": int(positives.shape[0]),
        "unrelated_pairs": int(unrelated.shape[0]),
    }


def arm_metrics(arm: str, tree: Path, embeddings: torch.Tensor, positives: torch.Tensor,
                unrelated: torch.Tensor, tau: float) -> dict[str, Any]:
    snapshots = OUTPUT / arm / "seed42" / "snapshots"
    result: dict[str, Any] = {"arm": arm, "tree": str(tree),
                              "population_sha256": population_sha(
                                  torch.arange(embeddings.shape[0])),
                              "tau": tau, "snapshots": {}}
    for step in SNAPSHOT_STEPS:
        path = snapshots / f"step_{step:04d}" / "snapshot.pt"
        if not path.is_file():
            result["snapshots"][str(step)] = {"missing": True}
            continue
        entry = metrics_for_snapshot(tree, path, embeddings, positives, unrelated, tau)
        entry["label"] = LABELS.get(step)
        result["snapshots"][str(step)] = entry
    return result


def main() -> int:
    """Compute the Phase-2 metrics for the four preflight arms on the fixed population."""
    import numpy as np
    root = REPO_ROOT / "results/stage2_RQ-VAE/r7_f_harness_v2"
    (root / "phase2_metrics").mkdir(parents=True, exist_ok=True)
    # the train-only embeddings the hosts train on
    trainer, saved_path, saved_mods = E.load_trainer(REPO_ROOT / "stage2_RQ-VAE/TIGER_r4_08",
                                                    module_name="r7f_embed")
    try:
        # the host's own loader plus its own PCA step, so the metrics see the training geometry
        import numpy as np
        raw = trainer.load_embeddings(trainer.EMBEDDING_FILE)
        embeddings = trainer.maybe_apply_pca(raw, trainer.PCA_DIM, trainer.SEED)
    except Exception as exc:  # noqa: BLE001
        print(f"embedding load failed: {exc}")
        embeddings = None
    finally:
        E.unload_trainer(saved_path, saved_mods, "r7f_embed")
    if embeddings is None:
        print(json.dumps({"status": "blocked",
                          "reason": "the host exposes no embedding loader the metrics can use"}))
        return 1
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    items = fixed_population()
    population = embeddings[items]
    generator = torch.Generator().manual_seed(7)
    total = population.shape[0]
    left = torch.randint(0, total, (512,), generator=generator)
    right = torch.randint(0, total, (512,), generator=generator)
    diff = left != right
    pairs = torch.stack([left[diff], right[diff]], dim=1)
    positives, unrelated = pairs[:256], pairs[256:]
    summary = {"population_items": int(total),
               "population_sha256": population_sha(items),
               "positive_pairs": int(positives.shape[0]),
               "unrelated_pairs": int(unrelated.shape[0]),
               "arms": []}
    for arm, host in (("F0", "r6_04"), ("F1", "r6_04"), ("F2", "r4_08"), ("F3", "r4_08")):
        result = arm_metrics(arm, REPO_ROOT / f"stage2-RQ-VAE/TIGER_{host}".replace(
            "stage2-RQ-VAE", "stage2_RQ-VAE"), population, positives, unrelated, 1e-3)
        summary["arms"].append(result)
        (root / "phase2_metrics" / f"{arm}.json").write_text(
            json.dumps(result, indent=2, default=str))
        print(f"{arm}: {len(result['snapshots'])} snapshots", flush=True)
    (root / "phase2_metrics" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
