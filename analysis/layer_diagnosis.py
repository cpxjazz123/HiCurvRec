"""Layer-wise residual and codebook diagnosis for one Stage 2 arm.

The question this answers: when a run's identifiers collapse, is the damage in
the LAYER that was changed, or in the layer that follows it?  The quantities
are the ones a diagnosis needs, and they are all computed from ONE forward pass
over the items plus the trained codebooks:

  * the INPUT of every layer (the encoder's latent for L1, the residual for L2
    and L3) and the residual the layer leaves behind;
  * the layer's codebook (the trained codewords);
  * the assignments (the SID's tokens).

Nothing here re-implements a quantizer: the per-layer residuals are DERIVED
from the encoder's latent, the trained codewords and the assignment the model
itself made, so every arm is measured on its own arithmetic.  The Euclidean
statistics are comparable across the arms because all three encode into the
same Euclidean latent space - the hyperbolic arm changed the COST of the level-1
assignment, not the space the vectors live in.

Usage:
    python analysis/layer_diagnosis.py --arm TIGER_curvature-stratum-sinkhorn-v1
    python analysis/layer_diagnosis.py --arm TIGER_RQ-VAE --checkpoint <path>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
STAGE2_ROOT = REPO / "results" / "stage2_RQ-VAE"
TREE_ROOT = REPO / "stage2_RQ-VAE"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
#: The pairs used for the mean distance (the full matrix would be 24587^2).
DISTANCE_SAMPLE = 2048
SEED = 42


def load_tree_model(arm: str):
    """The arm's own model package and its trained weights."""
    tree = TREE_ROOT / arm
    if not tree.is_dir():
        raise SystemExit("no such arm tree: " + str(tree))
    for name in ("model",):
        if str(tree) not in sys.path:
            sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "arm_model_%s" % arm.replace("-", "_"), tree / "model" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, tree


def build_config(arm: str):
    """The arm's own configuration, read from its trainer's constants."""
    trainer_path = TREE_ROOT / arm / "train_rqvae.py"
    source = trainer_path.read_text(encoding="utf-8")
    values = {}
    for name in ("HIDDEN_SIZES", "CODEBOOK_NUM", "CODEBOOK_SIZE", "CODEBOOK_DIM",
                 "BETA", "VQ_TYPE", "EMA_DECAY", "SK_EPSILON", "SK_ITERS", "PCA_DIM",
                 "XAVIER_INIT", "NUM_WORKERS", "STRAT_ENABLED"):
        for line in source.splitlines():
            if line.startswith(name + " "):
                try:
                    values[name] = ast_literal(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
                break
    sizes = values.get("CODEBOOK_SIZE", (256, 256, 256))
    if isinstance(sizes, int):
        sizes = (sizes,) * int(values.get("CODEBOOK_NUM", 3))
    from types import SimpleNamespace

    return SimpleNamespace(
        hidden_sizes=tuple(values.get("HIDDEN_SIZES", (512, 256, 128))),
        codebook_num=int(values.get("CODEBOOK_NUM", 3)),
        codebook_size=tuple(sizes),
        codebook_dim=int(values.get("CODEBOOK_DIM", 32)),
        dropout=0.0, beta=float(values.get("BETA", 0.25)), vq_type=str(values.get("VQ_TYPE", "vq")),
        ema_decay=float(values.get("EMA_DECAY", 0.99)), fix_code_embs=False,
        sk_epsilon=float(values.get("SK_EPSILON", 0.003)),
        sk_iters=int(values.get("SK_ITERS", 50)),
        stratified_enabled=bool(values.get("STRAT_ENABLED", False)),
        drift_enabled=bool(values.get("STRAT_ENABLED", False)),
    )


def ast_literal(text: str):
    import ast

    try:
        return ast.literal_eval(text.split("#")[0].strip())
    except (ValueError, SyntaxError):
        return None


def effective_rank(matrix: torch.Tensor) -> float:
    """The participation ratio of the covariance's eigenvalues.

    A set of vectors that really uses d directions has an effective rank of d;
    a set that has collapsed onto one direction has an effective rank near 1.
    """
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        return 0.0
    covariance = (centered.t() @ centered) / float(centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float(total * total / float((eigenvalues * eigenvalues).sum()))


def mean_pairwise_distance(matrix: torch.Tensor, sample: int = DISTANCE_SAMPLE,
                            seed: int = SEED) -> float:
    """The mean distance between the vectors, on a random subsample."""
    count = matrix.shape[0]
    if count < 2:
        return 0.0
    generator = torch.Generator().manual_seed(seed)
    if count > sample:
        index = torch.randperm(count, generator=generator)[:sample]
        matrix = matrix.index_select(0, index)
    distance = torch.cdist(matrix, matrix)
    n = matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return float(distance[mask].mean())


def entropy(counts) -> float:
    counts = [float(c) for c in counts if c > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total, 2) for c in counts)


def _install_mechanism_state(module, model, device) -> None:
    """Give a mechanism its own frozen state, computed by its own code."""
    import pandas as pd

    stratum = getattr(module, "stratum", None)
    if stratum is None:
        return
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    item_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    sequences = stratum.recover_train_sequences(frame)
    ranks = stratum.cohesion_ranks(sequences, item_ids)
    model.set_stratum_table(torch.from_numpy(np.asarray(ranks, dtype=np.float32)).to(device),
                            torch.from_numpy(np.asarray(item_ids, dtype=np.int64)).to(device))


def diagnose(arm: str, checkpoint: Path | None = None, out: Path | None = None) -> dict:
    module, tree = load_tree_model(arm)
    config = build_config(arm)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = np.load(EMBEDDINGS)
    model = module.RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    weights = checkpoint or (STAGE2_ROOT / arm / "rqvae_best.pth")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    payload = state.get("state_dict", state)
    missing = model.load_state_dict(payload, strict=False)
    if missing.unexpected_keys:
        raise SystemExit("the checkpoint does not fit this tree: %r" % missing.unexpected_keys[:4])
    model.eval()
    # An arm whose mechanism needs its own frozen state gets it from the arm's own
    # code, so the measurement is of the arm as it was trained.
    if hasattr(model.rq, "set_stratum_table"):
        _install_mechanism_state(module, model, device)

    items = torch.arange(embeddings.shape[0], dtype=torch.long)
    latents = []
    tokens = []
    with torch.no_grad():
        for start in range(0, items.numel(), 1024):
            batch = torch.from_numpy(embeddings[start:start + 1024]).to(device)
            ids = items[start:start + 1024].to(device)
            try:
                encoded = model.encoder(batch)
                _, _, _, row = model.rq(encoded, item_ids=ids)
            except TypeError:
                encoded = model.encoder(batch)
                _, _, _, row = model.rq(encoded)
            latents.append(encoded.cpu())
            tokens.append(row.cpu())
    z = torch.cat(latents).double()
    assignments = torch.cat(tokens)

    codebooks = []
    for layer in model.rq.vq_layers:
        codebooks.append(layer.get_code_embs().detach().cpu().double())
    # a layer whose codebook is a projection (SimVQ) is projected here too
    codebooks = [torch.as_tensor(c, dtype=torch.float64) for c in codebooks]

    report = {"arm": arm, "items": int(z.shape[0]), "levels": {}}
    residual = z
    for level in range(len(codebooks)):
        index = assignments[:, level]
        codewords = codebooks[level]
        chosen = codewords.index_select(0, index)
        # the quantisation error the model actually made
        chosen_error = (residual - chosen).pow(2).sum(-1).sqrt().mean()
        # the error the BEST code of the layer would have made
        best_error = torch.cdist(residual, codewords).min(dim=1).values.mean()
        used = torch.unique(index)
        per_group = {}
        if level + 1 < len(codebooks):
            following = assignments[:, level + 1]
            groups = {}
            for key, value in zip(index.tolist(), following.tolist()):
                groups.setdefault(key, []).append(value)
            effective = []
            conditional = 0.0
            for key, values in groups.items():
                counts = {}
                for value in values:
                    counts[value] = counts.get(value, 0) + 1
                effective.append(2.0 ** entropy(counts.values()))
                conditional += (len(values) / index.numel()) * entropy(counts.values())
            per_group = {"groups": len(groups),
                         "effective_next_codes_median": float(np.median(effective)),
                         "effective_next_codes_min": float(min(effective)),
                         "effective_next_codes_max": float(max(effective)),
                         "conditional_entropy_next_given_this_bits": float(conditional),
                         "conditional_effective_next_codes": float(2.0 ** conditional)}
        report["levels"]["L%d" % (level + 1)] = {
            "input_mean_pairwise_distance": mean_pairwise_distance(residual),
            "input_effective_rank": effective_rank(residual),
            "input_mean_norm": float(residual.norm(dim=1).mean()),
            "codebook_mean_pairwise_distance": mean_pairwise_distance(codewords, sample=256),
            "codebook_effective_rank": effective_rank(codewords),
            "codebook_used_codes": int(used.numel()),
            "codebook_max_share": float(torch.bincount(index, minlength=codewords.shape[0]).max()
                                        / index.numel()),
            "codebook_entropy_bits": entropy(torch.bincount(index, minlength=codewords.shape[0]).tolist()),
            "chosen_quantisation_error": float(chosen_error),
            "nearest_code_error": float(best_error),
            **per_group,
        }
        residual = residual - chosen

    path = out or (STAGE2_ROOT / arm / "layer_diagnosis.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args(argv)
    report = diagnose(arguments.arm,
                      Path(arguments.checkpoint) if arguments.checkpoint else None,
                      Path(arguments.out) if arguments.out else None)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
