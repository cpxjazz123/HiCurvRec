from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import curvature_config as experiment
from model import RQVAE
from model.layers import CURVATURE_MAX, CURVATURE_MIN


SEED = 42
MAX_GLOBAL_STEPS = int(experiment.MAX_GLOBAL_STEPS)
EVAL_INTERVAL_STEPS = int(experiment.EVAL_INTERVAL_STEPS)
BATCH_SIZE_PER_RANK = 1024
LOADER_BATCH_SIZE = BATCH_SIZE_PER_RANK // 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZES = (512, 256, 128)
CODEBOOK_SIZE = (256, 256, 256)
CODEBOOK_DIM = 32
BETA = 0.25
SK_EPSILON = 0.003
SK_ITERS = 50
BEHAVIOR_WEIGHT = 0.20
BEHAVIOR_RAMP_START = 20_000
BEHAVIOR_RAMP_END = 40_000
BEHAVIOR_TEMPERATURE = 0.07
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 0
FEATURE_NAMES = (
    "log1p_in_degree",
    "log1p_out_degree",
    "log1p_unique_out_neighbors",
    "log1p_two_hop_growth_ratio",
    "log1p_transition_repeat_density",
)

REPO_ROOT = Path(experiment.REPO_ROOT)
EMBEDDING_FILE = Path(experiment.EMBEDDING_FILE)
TRAIN_FILE = Path(experiment.TRAIN_FILE)
STAGE2_RESULT_DIR = Path(experiment.STAGE2_RESULT_DIR)
OUT_DIR = Path(experiment.RQVAE_OUT_DIR)
LOG_DIR = Path(experiment.STAGE2_LOG_DIR)
FREE_CURVATURE_FILE = REPO_ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter49/item_curvatures.npy"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
LAUNCHER_SCRIPT = Path(__file__).with_name("curvature_RQ-VAE.py")
TORCHRUN = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun"
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"
MASTER_PORT = 50202


class TransitionDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, frame: pd.DataFrame):
        pairs = []
        for history, target in frame[["history", "target"]].itertuples(index=False, name=None):
            if history is None or not isinstance(history, (list, tuple, np.ndarray)) or len(history) == 0:
                continue
            source_id, target_id = int(history[-1]), int(target)
            if min(source_id, target_id) < 0 or max(source_id, target_id) >= len(embeddings):
                raise ValueError("Train transition item IDs do not match embedding rows")
            pairs.append((source_id, target_id))
        if not pairs:
            raise ValueError("Training data contains no history-target pairs")
        self.embeddings = embeddings
        self.pairs = torch.tensor(pairs, dtype=torch.long)

    def __len__(self):
        return int(self.pairs.shape[0])

    def __getitem__(self, index):
        source_id, target_id = self.pairs[index]
        return source_id, target_id, self.embeddings[source_id], self.embeddings[target_id]


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
    matrix = np.asarray(np.load(path), dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix must be finite and two-dimensional")
    return matrix


def build_transition_features(frame: pd.DataFrame, item_count: int):
    in_degree = np.zeros(item_count, dtype=np.int64)
    out_degree = np.zeros(item_count, dtype=np.int64)
    outgoing = [set() for _ in range(item_count)]
    for history, target in frame[["history", "target"]].itertuples(index=False, name=None):
        if history is None or not isinstance(history, (list, tuple, np.ndarray)) or len(history) == 0:
            continue
        source_id, target_id = int(history[-1]), int(target)
        if min(source_id, target_id) < 0 or max(source_id, target_id) >= item_count:
            raise ValueError("Train transition item IDs do not match embedding rows")
        out_degree[source_id] += 1
        in_degree[target_id] += 1
        outgoing[source_id].add(target_id)
    unique_out = np.fromiter((len(neighbors) for neighbors in outgoing), dtype=np.float32, count=item_count)
    growth = np.zeros(item_count, dtype=np.float32)
    for item_id, neighbors in enumerate(outgoing):
        if not neighbors:
            continue
        two_hop = set()
        for neighbor in neighbors:
            two_hop.update(outgoing[neighbor])
        two_hop.discard(item_id)
        growth[item_id] = len(two_hop) / float(len(neighbors))
    repeat_density = out_degree / np.maximum(unique_out, 1.0)
    raw = np.stack(
        (
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(unique_out),
            np.log1p(growth),
            np.log1p(repeat_density),
        ),
        axis=1,
    ).astype(np.float32)
    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = ((raw - mean) / scale).astype(np.float32)
    return raw, normalized, mean, scale


def build_positive_curvatures(features: np.ndarray, reference_curvatures: np.ndarray):
    features = np.asarray(features, dtype=np.float32)
    reference_curvatures = np.asarray(reference_curvatures, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 4:
        raise ValueError("Expected normalized train-only transition features with branching columns")
    if reference_curvatures.shape != (features.shape[0],) or not np.isfinite(reference_curvatures).all():
        raise ValueError("Free-arm curvature reference must contain one finite value per item")
    branching_score = features[:, 1:4].astype(np.float64).sum(axis=1)
    item_ids = np.arange(features.shape[0], dtype=np.int64)
    rank_order = np.lexsort((item_ids, branching_score))
    curvatures = np.empty_like(reference_curvatures)
    curvatures[rank_order] = np.sort(reference_curvatures, kind="stable")
    return curvatures, branching_score


def tokenizer_config():
    return SimpleNamespace(
        hidden_sizes=HIDDEN_SIZES,
        codebook_size=CODEBOOK_SIZE,
        codebook_num=len(CODEBOOK_SIZE),
        codebook_dim=CODEBOOK_DIM,
        dropout=0.0,
        beta=BETA,
        sk_epsilon=SK_EPSILON,
        sk_iters=SK_ITERS,
    )


def _record(rank, **fields):
    if rank != 0:
        return
    row = {"timestamp": time.time(), **fields}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _extend_collisions(tokens: np.ndarray) -> np.ndarray:
    groups = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    offset = int(sum(CODEBOOK_SIZE))
    extended = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        for occurrence, item_id in enumerate(item_ids):
            extended[item_id, :-1] = row
            extended[item_id, -1] = offset + occurrence
    return extended


def _spearman(left, right):
    return float(pd.Series(left).corr(pd.Series(right), method="spearman"))


def _save_snapshot(raw_model, tokens, curvatures, feature_raw, feature_mean, feature_scale, step):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE2_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = STAGE2_RESULT_DIR / "dataset/Instruments"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT_DIR / "rqvae_best.pth"
    torch.save(
        {
            "state_dict": {name: value.detach().cpu() for name, value in raw_model.state_dict().items()},
            "global_step": int(step),
            "run_mode": "POSITIVE_STRUCTURE_RANK_MATCHED_CURVATURE_BEHAVIOR",
            "seed": SEED,
            "curvature_range": [CURVATURE_MIN, CURVATURE_MAX],
            "structural_feature_names": list(FEATURE_NAMES),
            "structural_feature_mean": feature_mean,
            "structural_feature_scale": feature_scale,
        },
        checkpoint,
    )
    raw_path = OUT_DIR / "sids_raw.npy"
    np.save(raw_path, tokens)
    sid = _extend_collisions(tokens)
    sid_path = dataset_dir / "sids_for_hgrec.npy"
    np.save(sid_path, sid)
    json_path = STAGE2_RESULT_DIR / "item_sids.json"
    json_path.write_text(
        json.dumps({str(i): [int(value) for value in row] for i, row in enumerate(sid)}, indent=2),
        encoding="utf-8",
    )
    curvature_path = STAGE2_RESULT_DIR / "item_curvatures.npy"
    np.save(curvature_path, curvatures)
    correlations = {
        name: _spearman(curvatures, feature_raw[:, column])
        for column, name in enumerate(FEATURE_NAMES)
    }
    _record(
        0,
        event="snapshot_saved",
        global_step=int(step),
        checkpoint=str(checkpoint),
        raw_sids=str(raw_path),
        sids_for_hgrec=str(sid_path),
        item_sids=str(json_path),
        item_curvatures=str(curvature_path),
        raw_unique=int(len(np.unique(tokens, axis=0))),
        sid_unique=int(len(np.unique(sid, axis=0))),
        curvature_min=float(curvatures.min()),
        curvature_mean=float(curvatures.mean()),
        curvature_std=float(curvatures.std()),
        curvature_max=float(curvatures.max()),
        curvature_structure_spearman=correlations,
    )


def _launch_via_torchrun():
    if not Path(TORCHRUN).is_file() or not Path(PYTHON).is_file():
        raise FileNotFoundError("Required genrec_env_v2 Python/torchrun executable is missing")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    command = [TORCHRUN, "--standalone", "--nnodes=1", "--nproc_per_node=4", f"--master_port={MASTER_PORT}", str(LAUNCHER_SCRIPT)]
    log_path = LOG_DIR / "train_migrated.log"
    print(f"[Iter50] launching four-card DDP: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)


def main():
    if "RANK" not in os.environ:
        _launch_via_torchrun()
        return
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(SEED + rank)

    embeddings_np = load_embeddings(EMBEDDING_FILE)
    frame = pd.read_parquet(TRAIN_FILE)
    feature_raw, feature_np, feature_mean, feature_scale = build_transition_features(frame, len(embeddings_np))
    reference_curvatures = np.load(FREE_CURVATURE_FILE).astype(np.float32, copy=False)
    curvature_np, branching_score = build_positive_curvatures(feature_np, reference_curvatures)
    all_embeddings = torch.from_numpy(embeddings_np)
    all_item_ids = torch.arange(len(embeddings_np), dtype=torch.long, device=device)
    curvature_table = torch.from_numpy(curvature_np).to(device)
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    if train_ids.size == 0 or train_ids.min() < 0 or train_ids.max() >= len(embeddings_np):
        raise ValueError("Training target item IDs do not match embedding rows")
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)].to(device)
    train_item_ids = torch.as_tensor(train_ids, device=device)

    model = RQVAE(tokenizer_config(), in_dim=embeddings_np.shape[1], item_curvatures=curvature_table).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    model.rq.behavior_loss_weight = BEHAVIOR_WEIGHT
    model.rq.behavior_temperature = BEHAVIOR_TEMPERATURE
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module

    train_dataset = TransitionDataset(all_embeddings, frame)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=False)
    loader = DataLoader(train_dataset, batch_size=LOADER_BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
    raw_model.set_curriculum_step(0)

    if rank == 0:
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text("", encoding="utf-8")
        _record(
            0,
            event="train_start",
            run_mode="POSITIVE_STRUCTURE_RANK_MATCHED_CURVATURE_BEHAVIOR",
            cold_start=True,
            checkpoint_loaded=False,
            initialization="xavier_encoder_decoder_then_kmeans_codebooks; fixed curvature rank-matched to Iter49 Free distribution",
            seed=SEED,
            max_global_steps=MAX_GLOBAL_STEPS,
            eval_interval_steps=EVAL_INTERVAL_STEPS,
            loader_batch_size=LOADER_BATCH_SIZE,
            effective_item_batch_size=BATCH_SIZE_PER_RANK,
            total_effective_item_batch_size=BATCH_SIZE_PER_RANK * world_size,
            dataset_kind="train_history_target_pairs",
            dataset_size=len(train_dataset),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            hidden_sizes=list(HIDDEN_SIZES),
            codebook_size=list(CODEBOOK_SIZE),
            codebook_dim=CODEBOOK_DIM,
            beta=BETA,
            sk_epsilon=SK_EPSILON,
            sk_iters=SK_ITERS,
            behavior_weight=BEHAVIOR_WEIGHT,
            behavior_ramp=[BEHAVIOR_RAMP_START, BEHAVIOR_RAMP_END],
            behavior_temperature=BEHAVIOR_TEMPERATURE,
            curvature_range=[float(curvature_np.min()), float(curvature_np.max())],
            curvature_mean=float(curvature_np.mean()),
            curvature_std=float(curvature_np.std()),
            curvature_mode="fixed_nontrainable_positive_structure_rank_assignment",
            curvature_reference_file=str(FREE_CURVATURE_FILE),
            curvature_score="sum(z(log1p(out_degree)), z(log1p(unique_out_neighbors)), z(log1p(two_hop_growth_ratio)))",
            curvature_score_spearman={
                name: _spearman(branching_score, feature_raw[:, column])
                for column, name in enumerate(FEATURE_NAMES)
            },
            structural_feature_names=list(FEATURE_NAMES),
            world_size=world_size,
            all_items=len(all_embeddings),
            train_target_items=len(train_embeddings),
            embedding_shape=list(embeddings_np.shape),
        )
    dist.barrier()

    with torch.no_grad():
        raw_model.init_codebook(train_embeddings, train_item_ids)
    dist.barrier()

    global_step = 0
    local_steps = 0
    epoch = 0
    interval_loss = interval_recon = interval_behavior = 0.0
    interval_updates = 0
    last_progress = time.time()
    while global_step < MAX_GLOBAL_STEPS:
        sampler.set_epoch(epoch)
        model.train()
        for source_ids, target_ids, source_embeddings, target_embeddings in loader:
            source_ids = source_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            batch_ids = torch.cat((source_ids, target_ids), dim=0)
            batch = torch.cat((source_embeddings, target_embeddings), dim=0).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _, behavior_loss, _ = model(
                batch, batch_ids, behavior_ids=(source_ids, target_ids)
            )
            loss, recon_loss = raw_model.compute_loss(batch, reconstructed, quant_loss, behavior_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at synchronized step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            local_steps += 1
            step_tensor = torch.tensor(local_steps, dtype=torch.int64, device=device)
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            global_step = int(step_tensor.item())
            raw_model.set_curriculum_step(global_step)
            interval_loss += float(loss.detach())
            interval_recon += float(recon_loss.detach())
            interval_behavior += float(behavior_loss.detach())
            interval_updates += 1
            if rank == 0 and time.time() - last_progress >= 5:
                print(
                    f"[Iter50] step={global_step}/{MAX_GLOBAL_STEPS} loss={float(loss.detach()):.6f} "
                    f"recon={float(recon_loss.detach()):.6f} behavior={float(behavior_loss.detach()):.6f} "
                    f"behavior_weight={raw_model.rq.get_behavior_weight():.4f}",
                    flush=True,
                )
                last_progress = time.time()
            evaluate = global_step % EVAL_INTERVAL_STEPS == 0 or global_step >= MAX_GLOBAL_STEPS
            if not evaluate:
                continue
            sums = torch.tensor([interval_loss, interval_recon, interval_behavior, interval_updates], dtype=torch.float64, device=device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            denom = max(float(sums[3].item()), 1.0)
            averages = (sums[:3] / denom).cpu().tolist()
            interval_loss = interval_recon = interval_behavior = 0.0
            interval_updates = 0
            model.eval()
            for layer in raw_model.rq.vq_layers:
                layer._skip_ddp_reduce = True
            try:
                if rank == 0:
                    with torch.no_grad():
                        tokens_t, layer_stats, curvature_t = raw_model.get_indices_with_stats(
                            all_embeddings.to(device), all_item_ids
                        )
                    tokens = tokens_t.cpu().numpy().astype(np.int64)
                    curvature = curvature_t.cpu().numpy().astype(np.float32)
                    usage = [stats["usage_counts"].cpu().tolist() for stats in layer_stats]
                    entropy = [float(stats["assignment_entropy_nats"].item()) for stats in layer_stats]
                    raw_unique = int(len(np.unique(tokens, axis=0)))
                    _record(
                        0,
                        event="train",
                        global_step=global_step,
                        loss=float(averages[0]),
                        recon=float(averages[1]),
                        behavior_loss=float(averages[2]),
                        behavior_weight=raw_model.rq.get_behavior_weight(),
                        raw_unique=raw_unique,
                        raw_total=len(tokens),
                        collision=1.0 - raw_unique / len(tokens),
                        codebook_usage_counts=usage,
                        codebook_used=[sum(count > 0 for count in row) for row in usage],
                        assignment_entropy_nats=entropy,
                        curvature_min=float(curvature.min()),
                        curvature_mean=float(curvature.mean()),
                        curvature_std=float(curvature.std()),
                        curvature_max=float(curvature.max()),
                    )
                    print(
                        f"[Iter50] evaluation step={global_step} raw_unique={raw_unique}/{len(tokens)} "
                        f"c_mean={curvature.mean():.6f} c_std={curvature.std():.6f}",
                        flush=True,
                    )
                    if global_step >= MAX_GLOBAL_STEPS:
                        _save_snapshot(
                            raw_model, tokens, curvature, feature_raw, feature_mean, feature_scale, global_step
                        )
                dist.barrier()
            finally:
                for layer in raw_model.rq.vq_layers:
                    layer._skip_ddp_reduce = False
            if global_step >= MAX_GLOBAL_STEPS:
                break
        epoch += 1
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
