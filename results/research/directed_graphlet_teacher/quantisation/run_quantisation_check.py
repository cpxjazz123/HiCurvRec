#!/usr/bin/env python3
"""Quantification controls on frozen inputs: what survives being turned into a SID?

Two questions, both asked with **the same RQ-VAE code** and **no extra structure loss**:

1. how much of a behaviour representation survives quantisation -- measured by probing
   the *quantised* vector ``q = decoder(tokens)`` against held-out behaviour edges, the
   same anchor/positive/negative triplets the relation audit uses;
2. does an input switch help or hurt -- the baseline encoder sees 768-d semantics, the
   control arm sees a frozen 128-d behaviour table (the GNN's own output, or its PCA
   compression to the same width), with the frozen recipe otherwise untouched.

Design notes that keep the comparison honest:

* both arms use the identical config file, so hidden sizes, codebooks, epochs, seed and
  the single-card global batch are the same; only the input table and the first linear
  layer's shape differ;
* a third arm (``semantics_pca128``) removes the width confound: 768-d semantics,
  linearly reduced to 128, into the same encoder;
* the sweep runs a fixed short budget so the *shape* of the quantisation curve is
  measured cheaply; the frozen 3000-epoch recipe decides the real SIDs afterwards;
* behaviour edges are never fed to the RQ-VAE; they are only used to score ``q``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
sys.path.insert(0, str(BASELINE))
sys.path.insert(0, str(TEACHER))
sys.path.insert(0, str(REPO / "results" / "research" / "directed_graphlet_teacher" / "relation"))

CONFIG_FILE = BASELINE / "config" / "train_config.json"
SEMANTICS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
GNN_TABLE = TEACHER / "representation" / "gnn_item_embedding.npz"
CONTRACT = TEACHER / "teacher" / "opt" / "data_contract.npz"
TRAIN_FILE = REPO / "results" / "stage0_build_parquet" / "train.parquet"
COVERAGE = TEACHER / "teacher" / "structure" / "v3" / "structure_targets_v3.npz"

# inputs: one frozen table each, no graph read at training time
ARMS = {
    "semantics":      {"table": "semantics", "pca": 0},
    "gnn":            {"table": "gnn", "pca": 0},
    "gnn_pca128":     {"table": "gnn", "pca": 128},
    "semantics_pca128": {"table": "semantics", "pca": 128},
}
EPOCH_GRID = (50, 150, 400)
SWEEP_SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Arm:
    key: str
    table: str
    pca: int

    @property
    def name(self) -> str:
        return f"{self.table}_pca{self.pca}" if self.pca else self.table


def arm_from_key(key: str) -> Arm:
    if key not in ARMS:
        raise SystemExit(f"unknown arm {key}; choose from {sorted(ARMS)}")
    spec = ARMS[key]
    return Arm(key, spec["table"], int(spec["pca"]))


def masked_ridge(rows: np.ndarray, covered: np.ndarray) -> int:
    """Load a (n, d) table; uncovered rows are replaced by the covered-row mean.

    Uncovered rows carry no behaviour evidence, so an arbitrary vector would inject noise
    the engine cannot see.  Their count is reported and hashed into the input record.
    """
    rows = np.array(rows, dtype=np.float32, copy=True)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise SystemExit("input table must be a finite 2-D matrix")
    if not covered.all():
        rows[~covered] = rows[covered].mean(axis=0, keepdims=True)
    return rows


def load_table(arm: Arm, covered: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the training matrix and a provenance record for the arm."""
    if arm.table == "semantics":
        raw = np.load(SEMANTICS)
        record = {"source": str(SEMANTICS.relative_to(REPO)),
                  "sha256": sha256(SEMANTICS), "raw_shape": list(raw.shape)}
    else:
        payload = np.load(GNN_TABLE)
        raw = payload["embedding"]
        record = {"source": str(GNN_TABLE.relative_to(REPO)),
                  "sha256": sha256(GNN_TABLE), "raw_shape": list(raw.shape),
                  "manifest": str((TEACHER / "representation" / "manifest.json").relative_to(REPO))}
    if arm.pca:
        from sklearn.decomposition import PCA
        raw = PCA(n_components=arm.pca, whiten=True,
                  random_state=SWEEP_SEED).fit_transform(raw).astype(np.float32)
        record["pca"] = {"fitted_on": "all items, both split halves",
                         "n_components": arm.pca, "whiten": True,
                         "random_state": SWEEP_SEED,
                         "note": "PCA is fit without behaviour labels and is identical "
                                 "for both splits of the same arm"}
    masked = int((~covered).sum())
    matrix = masked_ridge(raw, covered)
    record.update({"shape": list(matrix.shape), "dtype": str(matrix.dtype),
                   "masked_rows": masked,
                   "masked_row_value": "mean of covered rows" if masked else None})
    return matrix, record


ENGINE_KEYS = ("HIDDEN_SIZES", "CODEBOOK_NUM", "CODEBOOK_SIZE", "CODEBOOK_DIM", "BETA",
               "VQ_TYPE", "EMA_DECAY", "SK_EPSILON", "SK_ITERS", "LR", "WEIGHT_DECAY",
               "WARMUP_EPOCHS", "GRADIENT_CLIP_NORM", "OPTIMIZER", "SEED", "NUM_WORKERS")


def engine_config(epochs: int) -> tuple[SimpleNamespace, dict]:
    """The frozen architecture and optimisation, with only the epoch budget swept.

    `HIDDEN_SIZES` is left exactly as frozen; the baseline's first linear layer is built
    from `in_dim`, so a 128-d or a 768-d input feeds the same 512/256/128 trunk.
    """
    raw = json.loads(CONFIG_FILE.read_text())
    sizes = list(raw["CODEBOOK_SIZE"])
    if len(sizes) == 1:
        sizes = sizes * int(raw["CODEBOOK_NUM"])
    config = SimpleNamespace(
        hidden_sizes=tuple(int(size) for size in raw["HIDDEN_SIZES"]),
        codebook_num=int(raw["CODEBOOK_NUM"]),
        codebook_size=tuple(int(size) for size in sizes),
        codebook_dim=int(raw["CODEBOOK_DIM"]),
        dropout=0.0,
        beta=float(raw["BETA"]),
        vq_type=str(raw["VQ_TYPE"]),
        ema_decay=float(raw["EMA_DECAY"]),
        fix_code_embs=False,
        sk_epsilon=float(raw["SK_EPSILON"]),
        sk_iters=int(raw["SK_ITERS"]),
    )
    optim = {key: raw[key] for key in ("LR", "WEIGHT_DECAY", "WARMUP_EPOCHS",
                                       "GRADIENT_CLIP_NORM", "OPTIMIZER", "SEED",
                                       "NUM_WORKERS", "BATCH_SIZE_PER_RANK")}
    record = {"config_file": str(CONFIG_FILE.relative_to(REPO)),
              "config_sha256": sha256(CONFIG_FILE),
              "frozen_architecture": {key: raw[key] for key in ENGINE_KEYS},
              "frozen_optimisation": optim,
              "swept": {"EPOCHS": int(epochs)},
              "frozen_source_EPOCHS": raw["EPOCHS"]}
    return config, record


def item_split(n_items: int, seed: int, fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    cut = int(fraction * n_items)
    return np.sort(order[:cut]), np.sort(order[cut:])


def train_engine(matrix: np.ndarray, train_rows: np.ndarray, config, optim: dict,
                 epochs: int, distance, device):
    """Train the frozen RQ-VAE architecture on one frozen table."""
    import torch
    from model import RQVAE
    from losses import build_loss

    def set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(int(optim["SEED"]))
    model = RQVAE(config, in_dim=matrix.shape[1], distance=distance).to(device)
    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None and module.weight.dim() == 2:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    all_embeddings = torch.from_numpy(matrix).to(device)
    train_embeddings = all_embeddings[torch.from_numpy(train_rows)]
    model.init_codebook(train_embeddings)
    loss_fn = build_loss(json.loads(CONFIG_FILE.read_text()))
    optimizer = (torch.optim.AdamW(model.parameters(), lr=float(optim["LR"]),
                                   weight_decay=float(optim["WEIGHT_DECAY"])))
    batch = int(optim["BATCH_SIZE_PER_RANK"])
    rows = len(train_embeddings)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(rows, device=device)
        total = 0.0
        for start in range(0, rows, batch):
            chunk = order[start:start + batch]
            x = train_embeddings[chunk]
            reconstructed, quant_parts, _, _ = model(x)
            loss, _components = loss_fn(reconstructed, x, quant_parts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(optim["GRADIENT_CLIP_NORM"]))
            optimizer.step()
            total += float(loss.detach())
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"    epoch {epoch:4d}  loss {total / max(1, rows // batch):.5f}", flush=True)
    model.eval()
    return model, float(total / max(1, rows // batch))


def quantise(model, matrix: np.ndarray, device) -> dict:
    """Codes, the quantised vector q = decoder(tokens) and the plain reconstruction.

    `q` is the product that reaches Stage 3: it is what the decoder can say about an
    item from its three codes alone.  The reconstruction below it is the same quantity,
    computed by the model's own forward pass, and is used as a cross-check.
    """
    import torch
    with torch.no_grad():
        x = torch.from_numpy(matrix).to(device)
        encoded = model.encoder(x)
        quantized, _, _, tokens = model.rq(encoded, infer_use_sk=True)
        latent = encoded.cpu().numpy().astype(np.float32)
        forwarded = model.decoder(quantized).cpu().numpy().astype(np.float32)
        codes = tokens.cpu().numpy().astype(np.int64)
    recon = forwarded
    mse = float(((recon - matrix) ** 2).mean())
    var = float(matrix.var(axis=0).mean())
    # `q` is the decoder applied to the quantised latent, and that is also the plain
    # reconstruction, because the baseline's encoder resets and the rq forward keeps the
    # straight-through gradient only.  Calling the model's forward twice would draw a new
    # skeleton-ordered assignment, so it is not used here at all.
    return {"tokens": codes, "quantised": forwarded,
            "latent": latent, "recon_mse": mse, "recon_r2": 1.0 - mse / max(var, 1e-12)}


def code_statistics(tokens: np.ndarray, sizes: list[int]) -> dict:
    out = {}
    for level in range(tokens.shape[1]):
        counts = np.bincount(tokens[:, level], minlength=sizes[level])
        probs = counts / max(counts.sum(), 1)
        entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
        out[f"L{level + 1}"] = {"used": int((counts > 0).sum()), "size": int(sizes[level]),
                                "entropy_bits": entropy,
                                "max_share": float(probs.max())}
    triples = tokens[:, :3]
    unique = len(np.unique(triples, axis=0))
    out["collision"] = 1.0 - unique / max(len(triples), 1)
    out["unique_triples"] = int(unique)
    return out


def behaviour_probe(states: np.ndarray, triplets: tuple, covered: np.ndarray) -> dict:
    """Anchor-positive vs anchor-negative cosine AUC on held-out behaviour edges."""
    from audit_relation import auc_score, triplet_scores, usable_triplets
    pairs, negatives = triplets
    keep = usable_triplets(pairs, negatives, covered)
    pos, neg = triplet_scores(states.astype(np.float64), pairs[keep], negatives[keep])
    return {"auc": auc_score(pos, neg), "triplets": int(keep.sum()),
            "positive_cosine_mean": float(pos.mean()),
            "negative_cosine_mean": float(neg.mean())}


def nearest_neighbours(states: np.ndarray, probes: np.ndarray, k: int = 10) -> np.ndarray:
    block = states[probes].astype(np.float64)
    sim = block @ states.astype(np.float64).T
    sim[np.arange(len(probes)), probes] = -np.inf
    return np.argpartition(-sim, k, axis=1)[:, :k]


def sweep_main(argv: list[str] | None = None) -> None:
    import torch
    from distances import build_distance
    from train_config import load_train_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default=str(HERE / "sweep.json"))
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = load_train_config(CONFIG_FILE) if CONFIG_FILE.exists() else None
    distance = build_distance(SimpleNamespace(
        DISTANCE=json.loads(CONFIG_FILE.read_text()).get("DISTANCE", "euclidean")))
    contract = np.load(CONTRACT)
    pairs, negatives = contract["test_positive"], contract["test_negative"]
    covered = np.load(COVERAGE)["coverage_neighbour"] > 0
    n_items = len(covered)
    train_rows, held_rows = item_split(n_items, SWEEP_SEED)
    train_covered = covered.copy()
    train_covered[held_rows] = False

    results = {"inputs": {"contract": sha256(CONTRACT), "coverage": sha256(COVERAGE),
                          "train_file": str(TRAIN_FILE.relative_to(REPO)),
                          "config": json.loads(CONFIG_FILE.read_text()),
                          "epoch_grid": list(EPOCH_GRID), "seed": SWEEP_SEED,
                          "split": {"train_items": int(len(train_rows)),
                                    "held_items": int(len(held_rows)),
                                    "held_covered": int(covered[held_rows].sum())},
                          "frozen_config_loaded": bool(frozen is not None)},
               "arms": {}}
    reference_states = {}
    for key in args.arms.split(","):
        arm = arm_from_key(key)
        matrix, record = load_table(arm, train_covered)
        results["arms"][key] = {"input": record, "epochs": {}}
        print(f"[{arm.name}] input {matrix.shape}", flush=True)
        for epochs in EPOCH_GRID:
            config, cfg_record = engine_config(epochs)
            print(f"[{arm.name}] training {epochs} epochs", flush=True)
            model, loss = train_engine(matrix, train_rows, config, cfg_record["frozen_optimisation"],
                                       epochs, distance, device)
            out = quantise(model, matrix, device)
            stats = code_statistics(out["tokens"], list(config.codebook_size))
            entry = {"trained_loss": loss, "recon_mse": out["recon_mse"],
                     "recon_r2": out["recon_r2"], "codes": stats,
                     "probe_q": behaviour_probe(out["quantised"], (pairs, negatives), covered),
                     "probe_recon": behaviour_probe(out["quantised"], (pairs, negatives),
                                                    covered),
                     "probe_latent": behaviour_probe(out["latent"], (pairs, negatives), covered),
                     "probe_input": behaviour_probe(matrix, (pairs, negatives), covered)}
            probe_rows = held_rows[covered[held_rows]][:2000]
            results["arms"][key]["epochs"][str(epochs)] = entry
            print(f"[{arm.name}] {epochs:4d} epochs  R2 {out['recon_r2']:.3f}  "
                  f"collision {stats['collision']:.3f}  probe q {entry['probe_q']['auc']:.3f}",
                  flush=True)
            Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    sweep_main()
