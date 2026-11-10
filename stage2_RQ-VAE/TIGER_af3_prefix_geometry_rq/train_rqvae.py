"""Train RecBole3.0's TIGER RQ-VAE (4 卡 DDP) and export HG-Rec-compatible SIDs.

2026-09-22 改造 (CLAUDE.md §1 + §3):
- 参数全部硬编码为模块常量 (禁 argparse/CLI flag)
- 加 _launch_via_torchrun 自动 fork 4 卡 DDP (nproc=4, master_port=50202, 避免与 stage3 的 50201 冲突)
- 加全程 metrics JSONL 收集器 (每个 train/eval/ckpt_saved/train_end 一行)
- 输出路径硬编码到 results/stage2_RQ-VAE/TIGER_RQ-VAE/

2026-10-01 改造 (CLAUDE.md §2):
- 删除 PATIENCE / stale 计数与 early-stop 分支, 固定跑满 EPOCHS
- checkpoint 只在训练结束时保存一次 (final_epoch); 不再按 collision 最优
  覆盖, 中途评估仅记录, Stage3 只接受训练终点权重
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm

from model import RQVAE
from model.model import MixedRQVAE
from model.af3_runtime import (input_identity, resolve_shared_readout,
                               snapshot_sha256, snapshot_state)


# === 路径常量（无绝对路径；由文件自身向上找仓库标记推导）===
# Stage0 emits user-event parquet directly into results/stage0_build_parquet/.
def _repository_root() -> Path:
    """The repository root, found by walking up to the stage2_RQ-VAE directory.

    No absolute path and no parent counting: this file is built in the round
    workspace and copied into stage2_RQ-VAE/ before it trains, so the number of
    parents differs between the two locations while the marker does not.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
# The arm name is this tree's promoted name; a candidate copy changes only this line.
ARM_NAME = "TIGER_af3_prefix_geometry_rq"
# === Issue #139: the arm of this tree ===
# F0: mixed space, weight fixed at 1/2. G: one global weight per level, chosen
# by the outer loop. L: one weight per prefix. E: the same local loop with the
# curvature at zero, so both halves are Euclidean. A: the mixed space at 1/2
# plus the existing AF3 auxiliary supervision.
ARM_VARIANT = "AF3PREFGEO"
CURVATURE = 0.0 if ARM_VARIANT == "E" else 1.0
OUTER_LOOP = ARM_VARIANT in ("G", "L", "E")
OUTER_LOOP_EPOCHS = (1500, 2000, 2500, 3000)
# No wall-clock prefix truncation: Issue #139 requires a complete deterministic
# sweep. A failed full-sweep cost precheck is RESOURCE_BLOCKED, not permission
# to silently change which coordinates the experiment evaluates.
#: The frozen readout, shared by the outer-loop arms. Written once by F0 at its
#: 1500th epoch from the complete hard-SID readout of the training items.
READOUT_FILE = REPO_ROOT / "results" / "research" / "issue139" / "readout.json"
#: The frozen AF3 table of the training behaviour graph (issue #136's export).
AF3_CACHE = (REPO_ROOT / "results" / "research" / "issue121" / "community_curvature"
             / "train_edge_af3.npz")
AUX_AF3 = ARM_VARIANT == "A"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME

EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE     = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
OUTPUT_SID     = RESULT_DIR / "sids_for_hgrec.npy"
OUTPUT_JSON    = RESULT_DIR / "item_sids.json"
CHECKPOINT     = RESULT_DIR / "rqvae_best.pth"
LOG_DIR        = REPO_ROOT / "stage2_RQ-VAE" / ARM_NAME / "logs"
METRICS_PATH   = LOG_DIR / "training_metrics.jsonl"


# === 硬编码超参 (原 argparse default 值, 保留原 TIGER 默认) ===
EPOCHS              = 3000
BATCH_SIZE_PER_RANK = 1024       # 每卡 batch, 总 batch = 1024 * 4 = 4096
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
HIDDEN_SIZES        = (512, 256, 128)
CODEBOOK_NUM        = 3
CODEBOOK_SIZE       = (256, 256, 256)
CODEBOOK_DIM        = 32
BETA                = 0.25
VQ_TYPE             = "vq"
EMA_DECAY           = 0.99
SK_EPSILON          = 0.003
SK_ITERS            = 50
PCA_DIM             = 0
XAVIER_INIT         = True
WARMUP_EPOCHS       = 50
GRADIENT_CLIP_NORM  = 1.0
OPTIMIZER           = "AdamW"
SEED                = 42
NUM_WORKERS         = 0  # 2026-09-24: 0 防 DDP fork storm (4 rank × 4 worker = 16 子进程在 barrier 后挂死)
EVAL_INTERVAL       = 50
# CLAUDE.md §2: Stage2 不设任何 early-stop — 固定跑满 EPOCHS,
# 候选是否采用完全交由下游 stage3 test_R@10 裁决.
# 保留 best-collision checkpoint 仅作为导出用的最后一次快照.


# === DDP launcher 配置 (与 stage3 不同 master_port 避免冲突) ===
_LAUNCHER = {
    # torchrun lives next to the interpreter that runs this file: no absolute
    # path, and the same interpreter the project pins in CLAUDE.md §2.
    "torchrun":    str(Path(sys.executable).with_name("torchrun")),
    "script":      os.path.abspath(__file__),
    "nproc":       4,
    "master_port": 50202,
    "log":         str(LOG_DIR / "train_migrated.log"),
    "visible_dev": "0,1,2,3",
}


# === 2026-09-22: 全程训练 metrics JSONL 收集器 (rank 0 only) ===
_metrics_lock = threading.Lock()
_T0 = time.time()
# Truncate-on-train_start: every fresh run starts with an empty metrics file
# so a single ``training_metrics.jsonl`` always represents exactly one run
# (no leftover rows from a prior failed attempt).
_metrics_truncated = False
def _record(rank, event, **fields):
    if rank != 0:
        return
    rec = {"event": event, "timestamp": time.time(), "wall_time_s": round(time.time() - _T0, 3)}
    rec.update(fields)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    global _metrics_truncated
    with _metrics_lock:
        # Truncate only the very first time we append — that keeps the
        # file scoped to this run while still letting multiple ``_record``
        # calls in the same epoch run in append mode (cheap).
        if not _metrics_truncated:
            open(METRICS_PATH, "w", encoding="utf-8").close()
            _metrics_truncated = True
        with open(METRICS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


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
    """Load an (N, D) item-embedding matrix.

    Accepts either:
    - a Parquet frame with an ``embedding`` column of per-row arrays
      (the historical RecBole3.0 ``item_emb.parquet`` convention), or
    - an ``*.npy`` file containing an ``(N, D)`` ``float32`` array
      (stage1's ``sentence_t5.npy`` output).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        embeddings = np.asarray(np.load(path), dtype="float32")
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        if "embedding" not in frame:
            raise ValueError(f"Embedding file must contain an embedding column: {path}")
        embeddings = np.stack(frame["embedding"].to_numpy()).astype("float32", copy=False)
    else:
        raise ValueError(f"Unsupported embedding file suffix: {path}")
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



# === Issue #139: the AF3 outer loop =====================================
def _frozen_structure():
    """The graph, its target, the fixed support and per-pair targets."""
    from model import af3_structure as structure
    # The existing cache stores graph-position endpoints, not embedding IDs.
    # Reconstruct its exact TRAIN-only convention to recover and verify the
    # mapping, rather than assuming the graph rows equal sorted target IDs.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.research.science.community_curvature import data as graph_data
    graph = structure.load_graph(AF3_CACHE)
    canonical = graph_data.build_graph("train")
    if (graph["n_nodes"] != canonical.n_nodes
            or not np.array_equal(graph["rows"], canonical.edges[0])
            or not np.array_equal(graph["cols"], canonical.edges[1])):
        raise ValueError("AF3 cache does not match the canonical TRAIN graph")
    graph["item_ids"] = np.asarray(canonical.nodes, dtype=np.int64)
    kappa = structure.target_kappa(graph)
    if not np.allclose(kappa["kappa"], graph["weights"], rtol=0, atol=1e-6):
        raise ValueError("cached AF3 does not match the canonical TRAIN target")
    graph["degree"] = kappa["degree"]
    embeddings = np.load(EMBEDDING_FILE)
    if (len(np.unique(graph["item_ids"])) != graph["n_nodes"]
            or graph["item_ids"].min() < 0 or graph["item_ids"].max() >= len(embeddings)):
        raise ValueError("canonical TRAIN item mapping does not match embedding rows")
    support = structure.build_support(graph, kappa, embeddings[graph["item_ids"]], k=16, seed=42)
    lookup = {(int(i), int(j)): float(v)
              for i, j, v in zip(kappa["rows"], kappa["cols"], kappa["kappa"])}
    targets = np.array([lookup.get((int(i), int(j)), 0.0) for i, j in support["pairs"]])
    return graph, support, targets


def _readout_parameters(raw_module, train_embeddings, support, device):
    """``tau`` and the intercept of the shared readout, from one snapshot."""
    from model import af3_structure as structure
    from model.mixed_rq import encode_tokens
    from model.af3_selector import _readout_of
    with torch.no_grad():
        z = raw_module.encoder(train_embeddings.to(device))
        tokens = encode_tokens(z, raw_module.rq)
        phi = _readout_of(z, tokens, raw_module.rq)
    parameters = structure.calibrate(phi, support, device=phi.device)
    parameters["snapshot_sha256"] = snapshot_sha256(raw_module)
    return parameters


def _prepare_readout_inputs(raw_module):
    if not hasattr(raw_module, "_structure_bundle"):
        raw_module._structure_bundle = _frozen_structure()
    graph, support, _ = raw_module._structure_bundle
    common_code = [Path(__file__).parent / "model" / name for name in
                   ("af3_structure.py", "af3_engine.py", "af3_runtime.py", "mixed_rq.py",
                    "poincare_contract.py", "model.py", "layers.py")]
    identity = input_identity(graph, support, AF3_CACHE, EMBEDDING_FILE, TRAIN_FILE, common_code)
    return graph, support, identity


def _on_rank_zero(operation, rank, world_size):
    """Broadcast the result OR failure of a rank-zero-only mechanism operation.

    A missing shared calibration or a failed outer loop must not leave the
    other ranks waiting forever at a collective that rank zero never enters.
    This is not a launcher and does not grant training admission.
    """
    message = [None]
    if rank == 0:
        try:
            message[0] = {"ok": True, "value": operation()}
        except Exception as exc:
            message[0] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
    if world_size > 1:
        dist.broadcast_object_list(message, src=0)
    if not message[0]["ok"]:
        raise RuntimeError("rank-zero mechanism operation failed: " + message[0]["error"])
    return message[0]["value"]


@torch.no_grad()
def run_outer_loop(epoch, raw_module, train_embeddings, all_embeddings, device,
                   state_bundle, readout, record_fn):
    """One scheduled round: choose the geometry weights, never train."""
    import time
    from model.af3_engine import StructuralState
    from model.af3_selector import (choose_weight, prefix_groups, prefix_probe_counts,
                                    resolution_report, within_feasible_region,
                                    structural_guard_metrics, prefix_eligibility_error,
                                    resolve_local_table)
    from model.mixed_rq import encode_tokens
    from model.af3_selector import _readout_of
    import torch.nn.functional as F

    graph, support, targets = state_bundle
    # TRAIN graph vertices include history items and need not equal the
    # tokenizer's training target rows. Use the verified position -> ID map.
    item_ids = torch.as_tensor(graph["item_ids"], dtype=torch.long,
                                device=all_embeddings.device)
    graph_embeddings = all_embeddings[item_ids].to(device)
    started = time.time()
    if not hasattr(raw_module, "_structure_state"):
        raw_module._structure_state = StructuralState(
            support["pairs"], support["is_edge"], targets, graph["n_nodes"],
            tau=readout["tau"], intercept=readout["intercept"],
            scale=max(1.0, float(np.subtract(*np.quantile(
                targets[support["is_edge"]], [0.75, 0.25])))))
    state = raw_module._structure_state
    with torch.no_grad():
        z = raw_module.encoder(graph_embeddings)
        tokens = encode_tokens(z, raw_module.rq)
        phi = _readout_of(z, tokens, raw_module.rq)
    state.full(phi)
    transition = {"status": "not_applicable", "passes": 0, "deleted": []}
    if ARM_VARIANT in ("L", "E"):
        # The reviewed transition: a local entry whose prefix fell below the
        # activation rule is deleted, the codes are re-encoded under the
        # ancestor's weight, and the procedure repeats until nothing is stale.
        transition = resolve_local_table(
            raw_module.rq.alpha_table, z, raw_module.rq, state, tokens,
            encode=lambda: encode_tokens(z, raw_module.rq),
            readout=lambda current: _readout_of(z, current, raw_module.rq))
        if transition["status"] != "converged":
            raise RuntimeError("the local table did not reach a consistent state: "
                               + json.dumps(transition)[:300])
        if transition["deleted"]:
            tokens = encode_tokens(z, raw_module.rq)
            phi = _readout_of(z, tokens, raw_module.rq)
            state.full(phi)
            print("[outer loop] local table transition: %d stale entries deleted"
                  % len(transition["deleted"]), flush=True)
    start_distortion = state.summary()["distortion"]
    start_resolution = resolution_report(tokens, max(raw_module.rq.levels[0].n_embed, 2))
    with torch.no_grad():
        recon_start = float(F.mse_loss(raw_module.decoder(phi), graph_embeddings))
    degree = np.zeros(graph["n_nodes"])
    np.add.at(degree, graph["rows"], 1.0)
    np.add.at(degree, graph["cols"], 1.0)
    start = {"semantic": recon_start, "resolution": start_resolution}
    is_edge = support["is_edge"]
    start.update(structural_guard_metrics(state.probability, support["pairs"], is_edge, degree))

    def evaluate(candidate_tokens, outcome):
        if ARM_VARIANT in ("L", "E"):
            eligibility_error = prefix_eligibility_error(
                raw_module.rq.alpha_table, candidate_tokens, state)
            if eligibility_error:
                return False, eligibility_error
        resolution = resolution_report(candidate_tokens, max(raw_module.rq.levels[0].n_embed, 2))
        with torch.no_grad():
            candidate_phi = _readout_of(z, candidate_tokens, raw_module.rq)
            recon = float(F.mse_loss(raw_module.decoder(candidate_phi), graph_embeddings))
        guards = structural_guard_metrics(state.probability, support["pairs"], is_edge, degree)
        outcome["guard_metrics"] = dict(guards, semantic=recon, resolution=resolution)
        return within_feasible_region(start, resolution, recon,
                                      guards["adjacency_brier"], guards["degree_error"])

    record = {"round": int(getattr(raw_module, "_round_index", 0)), "epoch": int(epoch),
              "distortion_start": start_distortion, "accepted": [], "skipped": [],
              "rejected": [], "guard_start": start}
    raw_module._round_index = record["round"] + 1
    levels = len(raw_module.rq.levels)
    for level in range(1, levels + 1):
        if ARM_VARIANT == "G":
            groups = {(): np.arange(tokens.shape[0])}
        elif level == 1:
            # The first level has only the root: its table entry is the one
            # weight every item shares.
            groups = {(): np.arange(tokens.shape[0])}
        else:
            # A level's table is keyed by the *earlier* levels, so a third-level
            # entry is addressed by the first two tokens.
            groups = prefix_groups(tokens, level - 1)
        for prefix, items in groups.items():
            if level > 1 and len(items) < 32:
                record["skipped"].append({"level": level, "prefix": list(prefix),
                                          "items": int(len(items)),
                                          "reason": "fewer than 32 items, so it inherits its ancestor"})
                continue
            if level > 1:
                edges, unobserved = prefix_probe_counts(items, state)
                if edges < 64 or unobserved < 64:
                    record["skipped"].append({"level": level, "prefix": list(prefix),
                                              "items": int(len(items)),
                                              "probes": [edges, unobserved],
                                              "reason": "insufficient probes, so it inherits its ancestor"})
                    continue
            outcome = choose_weight(z, raw_module.rq, raw_module.rq.alpha_table, state,
                                    level=level, prefix=prefix, items=items,
                                    current_tokens=tokens, start=start,
                                    evaluate_feasibility=evaluate)
            entry = {"level": level, "prefix": list(prefix), "items": int(len(items)),
                     "attempts": outcome["attempts"], "accepted": outcome["accepted"]}
            if outcome["accepted"]:
                record["accepted"].append(entry)
                tokens = encode_tokens(z, raw_module.rq)
            else:
                record["rejected"].append(entry)
    record["distortion_end"] = state.summary()["distortion"]
    record["seconds"] = round(time.time() - started, 3)
    record["alpha_table"] = raw_module.rq.alpha_table.to_dict()
    record_fn(0, "outer_loop", **record)
    print(f"[outer loop] epoch={epoch} round={record['round']} "
          f"D {record['distortion_start']:.5f} -> {record['distortion_end']:.5f} "
          f"accepted={len(record['accepted'])} rejected={len(record['rejected'])} "
          f"skipped={len(record['skipped'])} "
          f"in {record['seconds']}s", flush=True)
    return record

def _tokenizer_config() -> SimpleNamespace:
    sizes = list(CODEBOOK_SIZE)
    if len(sizes) == 1:
        sizes *= CODEBOOK_NUM
    if len(sizes) != CODEBOOK_NUM:
        raise ValueError("CODEBOOK_SIZE must contain one size or one size per codebook level")
    return SimpleNamespace(
        hidden_sizes=HIDDEN_SIZES,
        codebook_num=CODEBOOK_NUM,
        codebook_size=tuple(sizes),
        codebook_dim=CODEBOOK_DIM,
        dropout=0.0,
        beta=BETA,
        vq_type=VQ_TYPE,
        ema_decay=EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=SK_EPSILON,
        sk_iters=SK_ITERS,
    )


# The collective timeout of the process group. The default of ten minutes is an
# infrastructure limit, not a scientific one: the outer loop of the local arms is
# a legitimate long computation on rank 0 while the other three wait at the
# weight-table broadcast, and an E-arm round was measured to exceed ten minutes
# and abort the run. Two hours is long enough for any legitimate round and short
# enough that a genuinely stuck job is still reported as a failure.
PROCESS_GROUP_TIMEOUT_S = 7200


# The single-card contract of the frozen recipe: when the framework asks for it,
# the trainer runs in this process as the only rank instead of forking four. The
# host that runs this arm has one card, so the four-rank path is not available.
SINGLE_GPU_ENV = "GENREC_SINGLE_GPU"
#: The per-rank batch in single-card mode. The four-rank recipe is four ranks of
#: 1024; on one card the same per-rank batch gives a quarter of the global batch,
#: which is a different optimisation regime and is recorded as such.
SINGLE_CARD_BATCH = 4096


def _single_card_mode() -> bool:
    return os.environ.get(SINGLE_GPU_ENV) == "1"


def main() -> None:
    # === DDP 初始化 ===
    if _single_card_mode():
        # One process, one rank, no process group: every collective below is
        # guarded by `world_size > 1` and therefore skipped.
        rank, world_size, local_rank = 0, 1, 0
        global BATCH_SIZE_PER_RANK
        BATCH_SIZE_PER_RANK = SINGLE_CARD_BATCH
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        if rank == 0:
            print("[single card] one process, one rank, per-process batch %d"
                  % BATCH_SIZE_PER_RANK, flush=True)
    elif "RANK" in os.environ and int(os.environ.get("RANK", -1)) >= 0:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(seconds=PROCESS_GROUP_TIMEOUT_S))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size == 1:
            # A one-rank group is the single-card case reached through torchrun:
            # keep the four-rank global batch so the optimisation does not move.
            BATCH_SIZE_PER_RANK = SINGLE_CARD_BATCH
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(SEED + rank)
    embeddings = maybe_apply_pca(load_embeddings(EMBEDDING_FILE), PCA_DIM, SEED)
    train_frame = pd.read_parquet(TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    if train_ids.size == 0 or train_ids.min() < 0 or train_ids.max() >= len(embeddings):
        raise ValueError("Training target item ids do not match the embedding rows")

    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]
    config = _tokenizer_config()
    model = MixedRQVAE(config, in_dim=embeddings.shape[1], curvature=CURVATURE).to(device)
    if ARM_VARIANT == "G":
        # G chooses independent per-layer globals. Otherwise the local-table
        # ancestor fallback would also change later layers when L1 changes.
        for level in range(1, CODEBOOK_NUM + 1):
            model.rq.alpha_table.set(level, (), 0.5)
    if XAVIER_INIT:
        initialize_tiger_weights(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_module = model.module if isinstance(model, DDP) else model
    readout_parameters, af3_state = None, None
    sampling = np.random.default_rng(SEED)
    if OUTER_LOOP or AUX_AF3:
        def prepare_consumer():
            if not READOUT_FILE.is_file():
                raise FileNotFoundError("missing F0 epoch-1500 shared readout: " + str(READOUT_FILE))
            graph, _, identity = _prepare_readout_inputs(raw_module)
            parameters = resolve_shared_readout(ARM_VARIANT, 0, READOUT_FILE, None, identity)
            return {"readout": parameters, "graph": graph if AUX_AF3 else None}

        prepared = _on_rank_zero(prepare_consumer, rank, world_size)
        readout_parameters = prepared["readout"]
        if AUX_AF3:
            from model.af3_auxiliary import prepare_state, augment_loss
            af3_state = prepare_state(prepared["graph"], all_embeddings, device)

    train_dataset = TensorDataset(train_embeddings)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=False) if world_size > 1 else None
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_RANK,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=False,  # 2026-09-24: 关闭避免每个 epoch 都 fork 一次
    )

    if OPTIMIZER.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, EPOCHS * len(loader))
    warmup_steps = max(0, WARMUP_EPOCHS * len(loader))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        return max(0.0, float(total_steps - step) / float(total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        print(f"[RecBole RQ-VAE] device={device} all_items={len(all_embeddings)} train_items={len(train_embeddings)} world_size={world_size}", flush=True)
        print(f"[RecBole RQ-VAE] config: epochs={EPOCHS} batch_size_per_rank={BATCH_SIZE_PER_RANK} total_batch={BATCH_SIZE_PER_RANK*world_size} lr={LR}", flush=True)
        _record(rank, "train_start", epochs=EPOCHS, batch_size_per_rank=BATCH_SIZE_PER_RANK, total_batch_size=BATCH_SIZE_PER_RANK*world_size, lr=LR, weight_decay=WEIGHT_DECAY, hidden_sizes=list(HIDDEN_SIZES), codebook_num=CODEBOOK_NUM, codebook_size=list(CODEBOOK_SIZE), codebook_dim=CODEBOOK_DIM, beta=BETA, vq_type=VQ_TYPE, ema_decay=EMA_DECAY, sk_epsilon=SK_EPSILON, sk_iters=SK_ITERS, pca_dim=PCA_DIM, xavier_init=XAVIER_INIT, warmup_epochs=WARMUP_EPOCHS, gradient_clip_norm=GRADIENT_CLIP_NORM, optimizer=OPTIMIZER, seed=SEED, num_workers=NUM_WORKERS, eval_interval=EVAL_INTERVAL, early_stop="disabled", all_items=len(all_embeddings), train_items=len(train_embeddings), embedding_shape=list(embeddings.shape), world_size=world_size)

    # === codebook init: only rank 0 (codebook 不是 DDP 参数, 全 rank 共享) ===
    if rank == 0:
        print("[RecBole RQ-VAE] initializing codebooks with KMeans", flush=True)
    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    if world_size > 1:
        dist.barrier()

    # Stage3 only consumes the final-epoch weights, so tracking the best
    # collision across epochs would be misleading; keep the last evaluation.
    final_collision = float("inf")
    final_raw_unique = 0
    codebook_sizes = list(config.codebook_size)

    for epoch in range(1, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        model.train()
        losses: list[float] = []
        recons: list[float] = []
        auxiliary_logs = []
        _epoch_t0 = time.time()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            # === DDP 包装下, 通过 .module 调 compute_loss (避免 DDP all_reduce 干扰) ===
            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"RQ-VAE loss became non-finite at epoch {epoch}")
            if AUX_AF3:
                loss, auxiliary_log = augment_loss(loss, raw_module, af3_state, epoch, sampling)
                auxiliary_logs.append(auxiliary_log)
            loss.backward()
            if GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            recons.append(float(recon_loss.detach()))
        epoch_time_s = round(time.time() - _epoch_t0, 3)

        # === DDP all_reduce: 跨 rank 求平均 loss / recon ===
        local_loss = torch.tensor(float(np.mean(losses)) if losses else 0.0, device=device)
        local_recon = torch.tensor(float(np.mean(recons)) if recons else 0.0, device=device)
        if world_size > 1:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_recon, op=dist.ReduceOp.SUM)
        avg_loss = float(local_loss.item()) / max(world_size, 1)
        avg_recon = float(local_recon.item()) / max(world_size, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        if AUX_AF3 and auxiliary_logs:
            active = [entry for entry in auxiliary_logs if entry["active"]]
            _record(rank, "auxiliary", epoch=epoch, active_steps=len(active),
                    ramp=auxiliary_logs[-1]["ramp"],
                    mean_scale=float(np.mean([entry["scale"] for entry in active])) if active else 0.0,
                    last_step=auxiliary_logs[-1])

        # === Issue #139: the scheduled outer-loop round ===
        if ((ARM_VARIANT == "F0" and epoch == 1500)
                or (OUTER_LOOP and epoch in OUTER_LOOP_EPOCHS)):
            model.eval()
            def scheduled_operation():
                graph, support, identity = _prepare_readout_inputs(raw_module)
                graph_embeddings = all_embeddings[torch.as_tensor(graph["item_ids"], dtype=torch.long)]
                parameters = resolve_shared_readout(
                    ARM_VARIANT, epoch, READOUT_FILE,
                    lambda: _readout_parameters(raw_module, graph_embeddings, support, device), identity)
                if ARM_VARIANT == "F0":
                    if parameters.get("snapshot_sha256") != snapshot_sha256(raw_module):
                        raise ValueError("existing F0 readout is bound to a different model snapshot")
                    _record(rank, "shared_readout_frozen", epoch=epoch, readout=parameters)
                else:
                    run_outer_loop(epoch, raw_module, train_embeddings, all_embeddings, device,
                                   raw_module._structure_bundle, parameters, _record)
                return {"alpha_table": raw_module.rq.alpha_table.to_dict(), "readout": parameters}

            payload = _on_rank_zero(scheduled_operation, rank, world_size)
            from model.mixed_rq import AlphaTable as _AlphaTable
            raw_module.rq.alpha_table = _AlphaTable.from_dict(payload["alpha_table"])
            readout_parameters = payload["readout"]
            model.train()

        # === 非 eval 间隔: 只 log train 行 ===
        if epoch % EVAL_INTERVAL != 0 and epoch != EPOCHS:
            if rank == 0:
                print(f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} lr={cur_lr:.3e} time={epoch_time_s}s", flush=True)
                _record(rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr, epoch_time_s=epoch_time_s, cumulative_time_s=round(time.time() - _T0, 3))
            continue

        # === eval 间隔: only rank 0 跑 (避免 4 卡 N× 全集评估) ===
        model.eval()
        # Skip DDP all_reduce inside VQLayer.forward when rank 0 runs eval alone:
        # rank 1-3 are stuck in barrier and won't join, would deadlock.
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = True
        raw_unique = None
        collision_v = None
        try:
            if rank == 0:
                with torch.no_grad():
                    raw_tokens = raw_module.get_indices(
                        all_embeddings.to(device), infer_use_sk=False
                    ).cpu().numpy().astype(np.int64)
                raw_unique = int(len(np.unique(raw_tokens, axis=0)))
                collision_v = 1.0 - raw_unique / len(raw_tokens)
                final_collision = collision_v
                final_raw_unique = raw_unique
                print(
                    f"[RQ-VAE] epoch={epoch} loss={avg_loss:.8f} recon={avg_recon:.8f} "
                    f"lr={cur_lr:.3e} raw_unique={raw_unique}/{len(raw_tokens)} "
                    f"collision={collision_v:.6f} time={epoch_time_s}s",
                    flush=True,
                )
                _record(
                    rank, "train", epoch=epoch, loss=avg_loss, recon=avg_recon, lr=cur_lr,
                    epoch_time_s=epoch_time_s,
                    cumulative_time_s=round(time.time() - _T0, 3),
                    raw_unique=raw_unique, raw_total=len(raw_tokens), collision=collision_v,
                )
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
        if world_size > 1:
            dist.barrier()

    # === 训练结束: only rank 0 保存终点 checkpoint 并导出 SID ===
    # Stage3 只接受训练终点权重；不保留任何中间或"最优 collision"快照。
    if rank == 0:
        final_state = snapshot_state(raw_module)
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": final_state,
                "epoch": EPOCHS,
                "collision": final_collision,
                "selection": "final_epoch",
                "arm_variant": ARM_VARIANT,
                "readout": readout_parameters,
            },
            CHECKPOINT,
        )
        _record(
            rank, "ckpt_saved", epoch=EPOCHS, path=str(CHECKPOINT),
            reason="final_epoch", collision=final_collision,
            raw_unique=final_raw_unique,
        )
        raw_module.eval()
        # Skip DDP all_reduce inside VQLayer.forward during the export
        # inference pass — rank 1-3 are stuck in barrier, would deadlock.
        for layer in raw_module.rq.vq_layers:
            layer._skip_ddp_reduce = True
        try:
            with torch.no_grad():
                raw_tokens = raw_module.get_indices(
                    all_embeddings.to(device), infer_use_sk=False
                ).cpu().numpy().astype(np.int64)
        finally:
            for layer in raw_module.rq.vq_layers:
                layer._skip_ddp_reduce = False
        sid = _extend_collisions(raw_tokens, codebook_sizes)
        if len(np.unique(sid, axis=0)) != len(sid):
            raise RuntimeError("RecBole SID extension failed to make item codes unique")
        OUTPUT_SID.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_SID, sid)
        payload = {str(item_id): [int(value) for value in row] for item_id, row in enumerate(sid)}
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[RecBole RQ-VAE] exported raw_shape={raw_tokens.shape} sid_shape={sid.shape} unique={len(np.unique(sid, axis=0))} -> {OUTPUT_SID}", flush=True)
        _record(rank, "train_end", total_wall_time_s=round(time.time() - _T0, 3), final_collision=final_collision, checkpoint_selection="final_epoch", sid_shape=list(sid.shape), output_sid=str(OUTPUT_SID), output_json=str(OUTPUT_JSON), unique_after_extend=len(np.unique(sid, axis=0)))

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def _visible_card_count() -> int:
    """How many cards this process can actually see."""
    try:
        import torch
        return max(1, int(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 - a host without torch still has to run CPU
        return 0


def _launch_via_torchrun() -> None:
    """Four ranks on a four-card host; this process alone when there is one card.

    The frozen recipe starts four ranks through the framework's launcher. On a
    host with a single visible card that launch cannot work: ranks 1 to 3 ask for
    ``cuda:1`` to ``cuda:3`` and torch aborts with 'invalid device ordinal', which
    is what this host was doing. A one-card host therefore runs the trainer in
    this process, with the per-process batch that keeps the global batch at the
    four-rank value, and no process group at all.
    """
    if int(os.environ.get("RANK", -1)) >= 0:
        return  # 已经在 torchrun 启动的子进程里
    if _visible_card_count() <= 1:
        os.environ[SINGLE_GPU_ENV] = "1"
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = _LAUNCHER["visible_dev"]
    env["NCCL_IB_DISABLE"]     = "1"
    env["NCCL_P2P_DISABLE"]    = "1"
    env["NCCL_SHM_DISABLE"]    = "1"
    env["NCCL_TIMEOUT"]        = "3600"
    env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    # The four ranks are started by the framework's launcher, not by this file:
    # the port, the world size, the working directory and the log belong to the
    # frozen contract, so that no candidate has to re-implement them.
    sys.path.insert(0, str(REPO_ROOT))
    from tools.research.stage2 import contract as stage2_contract
    from tools.research.stage2 import launcher as stage2_launcher
    rc = stage2_launcher.launch(
        entry=_LAUNCHER["script"], result_dir=RESULT_DIR,
        port=stage2_contract.STAGE2_PORT, world_size=int(_LAUNCHER["nproc"]),
        python=_LAUNCHER["torchrun"].rsplit("/", 1)[0] + "/python3.9",
        env_extra={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "NCCL_IB_DISABLE",
                                       "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
                                       "NCCL_TIMEOUT", "TORCH_NCCL_BLOCKING_WAIT")})
    sys.exit(rc)


if __name__ == "__main__":
    _launch_via_torchrun()
    main()

