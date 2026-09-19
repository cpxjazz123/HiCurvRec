"""Train RecBole3.0's TIGER RQ-VAE and export HG-Rec-compatible SIDs.

The sequence recommender is intentionally outside this file.  This stage only
replaces the old hyperbolic tokenizer.  Integer SIDs follow RecBole's common
namespace (raw code + 1 in the HG loader); collisions are resolved exactly as
RecBole's ``sid_collision_handling=extend`` path by appending one fixed-width
disambiguation level.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import RQVAE


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset/Amazon_2023_Instruments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_file", type=Path, default=DATA_DIR / "item_emb.parquet")
    parser.add_argument("--train_file", type=Path, default=DATA_DIR / "train_recbole.parquet")
    parser.add_argument("--output_sid", type=Path, default=pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/sids_for_hgrec_recbole.npy"))
    parser.add_argument("--output_json", type=Path, default=pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids_recbole.json"))
    parser.add_argument("--checkpoint", type=Path, default=pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_recbole_best.pth"))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--codebook_num", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--codebook_dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--vq_type", choices=["vq", "ema", "simvq"], default="vq")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--sk_epsilon", type=float, default=0.003)
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=0,
        help="Optional PCA dimension before RQ-VAE; 0 keeps the original 768-d embeddings.",
    )
    parser.add_argument(
        "--xavier_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the Xavier initialization from the upstream TIGER implementation.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["AdamW", "Adagrad"], default="AdamW")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")


def load_embeddings(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    if "embedding" not in frame:
        raise ValueError(f"Embedding file must contain an embedding column: {path}")
    embeddings = np.stack(frame["embedding"].to_numpy()).astype("float32", copy=False)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embedding matrix: shape={embeddings.shape}")
    return embeddings


def maybe_apply_pca(embeddings: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    if pca_dim <= 0:
        return embeddings
    if pca_dim >= embeddings.shape[1]:
        raise ValueError(f"--pca_dim must be smaller than input dimension {embeddings.shape[1]}")
    from sklearn.decomposition import PCA

    reduced = PCA(n_components=pca_dim, whiten=True, random_state=seed).fit_transform(embeddings)
    return reduced.astype("float32", copy=False)


def initialize_tiger_weights(model: RQVAE) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _extend_collisions(tokens: np.ndarray, codebook_sizes: list[int]) -> np.ndarray:
    """Match RecBole3.0's extend path with one fixed extra SID level."""
    if tokens.ndim != 2 or tokens.shape[1] != len(codebook_sizes):
        raise ValueError(f"Token shape {tokens.shape} disagrees with codebook sizes {codebook_sizes}")
    groups: dict[tuple[int, ...], list[int]] = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    offsets = np.cumsum([0, *codebook_sizes], dtype=np.int64)
    result = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        for occurrence, item_id in enumerate(item_ids):
            result[item_id, :-1] = np.asarray(row, dtype=np.int64)
            result[item_id, -1] = int(offsets[-1] + occurrence)
    return result


def _tokenizer_config(args: argparse.Namespace) -> SimpleNamespace:
    sizes = list(args.codebook_size)
    if len(sizes) == 1:
        sizes *= int(args.codebook_num)
    if len(sizes) != int(args.codebook_num):
        raise ValueError("--codebook_size must contain one size or one size per codebook level")
    return SimpleNamespace(
        hidden_sizes=tuple(args.hidden_sizes),
        codebook_num=int(args.codebook_num),
        codebook_size=tuple(sizes),
        codebook_dim=int(args.codebook_dim),
        dropout=0.0,
        beta=float(args.beta),
        vq_type=args.vq_type,
        ema_decay=float(args.ema_decay),
        fix_code_embs=False,
        sk_epsilon=float(args.sk_epsilon),
        sk_iters=int(args.sk_iters),
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_interval <= 0:
        raise ValueError("epochs, batch_size, and eval_interval must be positive")
    set_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    embeddings = maybe_apply_pca(load_embeddings(args.embedding_file), args.pca_dim, args.seed)
    train_frame = pd.read_parquet(args.train_file)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    if train_ids.size == 0 or train_ids.min() < 0 or train_ids.max() >= len(embeddings):
        raise ValueError("Training target item ids do not match the embedding rows")

    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    config = _tokenizer_config(args)
    model = RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if args.xavier_init:
        initialize_tiger_weights(model)
    loader = DataLoader(
        TensorDataset(train_embeddings),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = max(0, args.warmup_epochs * len(loader))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        return max(0.0, float(total_steps - step) / float(total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"[RecBole RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)}")
    print(f"[RecBole RQ-VAE] config={vars(args)}")
    print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    with torch.no_grad():
        model.init_codebook(train_embeddings.to(device))

    best_collision = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    codebook_sizes = list(config.codebook_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        for (batch,) in tqdm(loader, desc=f"RQ-VAE {epoch}", leave=False, ncols=100):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))

        if epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue
        model.eval()
        with torch.no_grad():
            tokens = model.get_indices(all_embeddings.to(device), infer_use_sk=False).cpu().numpy().astype(np.int64)
        unique = int(len(np.unique(tokens, axis=0)))
        collision = 1.0 - unique / len(tokens)
        print(
            f"[RQ-VAE] epoch={epoch} loss={np.mean(losses):.8f} recon={np.mean(recons):.8f} "
            f"raw_unique={unique}/{len(tokens)} collision={collision:.6f}",
            flush=True,
        )
        if collision < best_collision:
            best_collision = collision
            stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "config": vars(args), "epoch": epoch, "collision": collision}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[RQ-VAE] early stop at epoch={epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("RQ-VAE did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw_tokens = model.get_indices(all_embeddings.to(device), infer_use_sk=False).cpu().numpy().astype(np.int64)
    sid = _extend_collisions(raw_tokens, codebook_sizes)
    if len(np.unique(sid, axis=0)) != len(sid):
        raise RuntimeError("RecBole SID extension failed to make item codes unique")

    args.output_sid.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_sid, sid)
    payload = {str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)}
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[RecBole RQ-VAE] exported raw_shape={raw_tokens.shape} sid_shape={sid.shape} "
        f"unique={len(np.unique(sid, axis=0))} -> {args.output_sid}",
        flush=True,
    )


if __name__ == "__main__":
    main()
