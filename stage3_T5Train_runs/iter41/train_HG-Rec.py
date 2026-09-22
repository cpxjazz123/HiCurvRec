"""Train HG-Rec with the vanilla TIGER protocol used by RecBole3.0.

Hardware: tuned for 4x L40S (46GB each).
Validation runs as DDP across all 4 ranks (4x speedup over rank-0 only).
torch.compile is OFF by default; the model has + multi-GPU DDP has functorch
compat issues on this stack. Enable manually only after verifying.

=== 2026-09-17 DDP launcher enabled ===
`python3 train_HG-Rec.py` 自动 fork torchrun --nproc_per_node=4 (硬编码 4 卡).
如果在 torchrun 下调用 (RANK 已 set), 直接执行 main, 不再二次 fork.



The hyperbolic component only supplies the SID file.  The sequence model,
token namespace, prefix expansion, generation and full-evaluation behavior in
this entry point are deliberately the same as RecBole's ``model=tiger``.
"""

from contextlib import nullcontext
import json
import logging
import math
import os
import random
import time
import re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.dataloader import GenRecDataLoader
from data.dataset import GenRecDataset
from model.hg_rec import HG_Rec
from model.utils import ensure_dir, get_local_time, set_color


# Keep the existing deterministic setting used by the HG-Rec experiments.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


# === 模块级常量 (CLAUDE.md §4 + §1 R30/R43: 全部硬编码, 禁止 argparse / CLI flag) ===
# 训练 / 吞吐
BATCH_SIZE       = 4096   # 4 卡 DDP 每卡 1024 (8192 OOM @ 46GB L40S, 4096 安全)
INFER_SIZE       = 1024   # 4 卡 DDP 每卡 256, 等效 4x throughput (vs 单卡 512)
NUM_EPOCHS       = 400      # iter11: 2x 长训练, 让 Gumbel-Softmax 充分收敛
EVAL_INTERVAL    = 5      # validate every N epochs (NO_EVAL=False 时生效)
EVAL_START_EPOCH = 200    # iter11: 200 epoch 后才评估, 让 stage2 编码先充分学, 避免早期 valid 噪声
EARLY_EVAL_EPOCHS = (50, 100, 150)  # iter17: 早期诊断点, 监控 valid 走势; 仅日志, 不影响 best_monitor / early_stop
EARLY_STOP       = 10     # iter11: patience=10, 避免 valid 仍在涨时被截断
FAST             = True   # A100 吞吐向: 关 deterministic, 开 cuDNN benchmark
BF16             = True   # BF16 autocast (require CUDA+bf16-supported)
COMPILE          = False  # torch.compile 总开关 (与 DDP functorch 旧栈有兼容问题)
COMPILE_MODE     = "default"  # torch.compile 模式: default / reduce-overhead / max-autotune

# 模型 / Transformer
NUM_LAYERS          = 4
NUM_DECODER_LAYERS  = 4
D_MODEL             = 128
D_FF                = 1024
NUM_HEADS           = 6
D_KV                = 64
DROPOUT_RATE        = 0.1
ACTIVATION_FUNCTION = "relu"
FEED_FORWARD_PROJ   = "relu"
DEVICE              = "cuda"  # DDP 路径下由 local_rank 覆盖, 非 DDP 时用此

# Optimizer
LR            = 0.003
WEIGHT_DECAY  = 0.05
# === iter12: cosine LR schedule + 5% warmup (Loshchilov & Hutter 2017 SGDR) ===
# 让 stage3 训练在 epoch 200 后 LR 缓降, 帮助 valid/test 突破 0.06 ceiling.
USE_LR_SCHEDULER = True   # iter12: 启用 cosine annealing
LR_WARMUP_PCT   = 0.05    # 5% warmup (前 5% epoch 从 LR/10 升到 LR)
LR_MIN_FACTOR   = 0.1     # cosine 末端 LR = LR * LR_MIN_FACTOR (即 3e-4)

# Token / 数据集命名空间 (0 = 运行时由 train_dataset 覆盖)
VOCAB_SIZE    = 0
EOS_TOKEN_ID  = 0
PAD_TOKEN_ID  = 0
N_USER_TOKENS = 1
MAX_LEN       = 20

# 数据集 / 路径
DATASET_NAME  = "Amazon_2023_Instruments"
DATASET_PATH  = "./dataset/"
CODEBOOK_SIZE = [256, 256, 256, 1]
CODE_PATH     = "item_sids_iter41.json"  # iter41 L1-only sk_eps=0.05 (L0=0.50 fixed, =iter11 baseline), 4-token collision extension
TRAIN_FILE    = "train_recbole.parquet"
VALID_FILE    = "valid_recbole.parquet"
TEST_FILE     = "test_recbole.parquet"
# === 按 RQ-VAE 变体分类输出目录 ===
# 从 CODE_PATH 文件名 (item_sids_<variant>.json) 自动派生 variant 名,
# 写入 LOG_PATH / SAVE_PATH 子目录, 让 test_final.json / ckpt 按 RQ-VAE 分类.
_RQVAE_VARIANT_MAP = {
    "item_sids_recbole.json":         "tiger_baseline",
    "item_sids_iter2.json":           "iter2",
    "item_sids_iter3.json":           "iter3",
    "item_sids_iter4.json":           "iter4",
    "item_sids_iter5.json":           "iter5_mixed_curvature_hhs",
    "item_sids_iter6.json":           "iter6",
    "item_sids_iter8.json":           "iter8_mgc_fixed_c",
    "item_sids_iter9.json":           "iter9",
    "item_sids_iter10.json":          "iter10_gumbel_softmax_anneal",
    "item_sids_iter11.json":          "iter11_L0_collapse_sk05",
    "item_sids_iter28.json":          "iter28_density_radial_alpha025",
    "item_sids_iter32.json":          "iter32_per_layer_hetero_c_sk_eps",
    "item_sids_iter33.json":          "iter33_per_item_curvature",
    "item_sids_iter12.json":          "iter12_L0_sk01",
    "item_sids_iter13.json":          "iter13_L0_sk005",
    "item_sids_iter14.json":          "iter14_L0_sk001",
    "item_sids_iter20.json":          "iter20_L0L1_05_sk05",
    "item_sids_iter23.json":          "iter23_dup2unique_iter20",
    "item_sids_iter24.json":          "iter24_collision_ext_iter20",
    "item_sids_iter23.json":          "iter25_dup2unique_L2_iter20",
    "item_sids_iter11.json":          "iter11_L0_collapse_sk05",
}
RQVAE_VARIANT  = _RQVAE_VARIANT_MAP.get(CODE_PATH, "unknown_variant")
LOG_PATH      = "./logs/" + RQVAE_VARIANT + "/"
SAVE_PATH     = "./ckpt/" + RQVAE_VARIANT + "/"
SEED          = 42

# 评估 / Screen
TOPK_LIST                = [5, 10]   # 必须包含 10 (RecBole 监控 ndcg@10)
BEAM_SIZE                = 20        # >= max(TOPK_LIST)
NO_EVAL                  = False     # True 时仅按 train_loss 选/早停
SKIP_TEST                = False     # True 时训练后跳过 final test
SCREEN_BASELINE_LOG      = ""        # 早期对照旧 run Recall/NDCG 曲线; 空串 = 不启用
SCREEN_START_EPOCH       = 40
SCREEN_WINDOW            = 5
SCREEN_MIN_NDCG_GAIN      = 0.001
SCREEN_MIN_RECALL_GAIN   = 0.001
SCREEN_SKIP_TEST_ON_FAIL = True

# DataLoader
NUM_WORKERS          = 16
PREFETCH_FACTOR      = 8
EXCLUDE_HISTORY      = True   # 评估时排除 seen items
NO_EXCLUDE_HISTORY   = False  # 拍平, 与 EXCLUDE_HISTORY 互斥, runtime 派生
PIN_MEMORY           = True   # 拍平后最终值, runtime 仍按 device 派生
NO_PIN_MEMORY        = False  # 拍平, 与 PIN_MEMORY 互斥, runtime 派生


def cleanup_distributed():
    # === R7-fix v5: dist.destroy_process_group 在多 NUMA 拓扑上挂住,
    # 用后台线程 + 短超时跑 destroy, 超时直接 os._exit 跳过 NCCL 退出. ===
    if dist.is_available() and dist.is_initialized():
        import threading
        _done = threading.Event()
        def _destroy():
            try:
                dist.destroy_process_group()
            finally:
                _done.set()
        _t = threading.Thread(target=_destroy, daemon=True)
        _t.start()
        if not _done.wait(timeout=10):
            # 10s 还没 destroy 完, 直接强制退出 (跳过 NCCL 收尾)
            os._exit(0)
        # 正常路径再走 return


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_file(dataset_dir: Path, filename: str) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else dataset_dir / path


def _resolve_code_file(dataset_dir: Path, requested: str) -> Path:
    """Resolve both a direct RecBole-style SID path and old HG prefixes."""
    requested_path = Path(requested)
    candidates = []
    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.extend(
            [
                dataset_dir / requested_path,
                dataset_dir / (dataset_dir.name + requested_path.name),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"SID file does not exist; checked: {joined}")


def make_sample_collator(dataset: GenRecDataset, *, include_eval_metadata=False):
    """Create the [user, history SID..., EOS] TIGER input and target labels."""

    def sample_collator(item):
        history_tokens = [token for sid in item["history"] for token in sid]
        input_ids = [
            dataset.user_token_for(item["user"]),
            *history_tokens,
            dataset.eos_token,
        ]
        result = {
            "input_ids": input_ids,
            "labels": [*item["target"], dataset.eos_token],
        }
        if include_eval_metadata:
            result["target_item"] = int(item["target_item"])
            result["seen_item_ids"] = list(item["seen_item_ids"])
        return result

    return sample_collator


def _autocast_context(device, use_bf16):
    """Use A100-friendly BF16 autocast without changing the default path."""
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _loader_options(config):
    """Return loader options that overlap CPU collation with GPU work."""
    options = {
        "num_workers": int(config["num_workers"]),
        "pin_memory": bool(config["pin_memory"]),
    }
    if options["num_workers"] > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = int(config["prefetch_factor"])
    return options


def train(model, train_loader, optimizer, device, epoch, *, use_bf16=False):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in tqdm(
        train_loader,
        ncols=100,
        desc=set_color(f"Training Epoch {epoch}", "pink"),
    ):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_bf16):
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def evaluate(
    model,
    eval_loader,
    eval_dataset: GenRecDataset,
    topk_list,
    beam_size,
    device,
    *,
    exclude_history=True,
    use_bf16=False,
):
    """Evaluate with RecBole's full item-level TIGER semantics.

    Generation stays on the GPU, while the small set of beam tuples is mapped
    back to item ids before metrics are computed. Invalid tuples, duplicate
    items and items in the complete seen history are skipped, then the
    deterministic item order fills a short list just like RecBole.
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    topk_list = [int(k) for k in topk_list]
    if not topk_list:
        raise ValueError("topk_list must not be empty")
    max_k = max(topk_list)
    if beam_size < max_k:
        raise ValueError(f"beam_size ({beam_size}) must be >= max top-k ({max_k})")

    recall_sums = {k: 0.0 for k in topk_list}
    ndcg_sums = {k: 0.0 for k in topk_list}
    n_eval = 0
    with torch.inference_mode():
        for batch in tqdm(
            eval_loader,
            ncols=100,
            desc=set_color("Evaluating", "pink"),
        ):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            batch_size = int(input_ids.shape[0])

            with _autocast_context(device, use_bf16):
                # === R7-fix v7: 显式 max_new_tokens = n_digit+1 (含 SOS),
                # 避免模型 generation_config.max_length 用训练期 n_digit=3 太短
                # 导致生成只有 4 token, slice 1:5 只能取 3 个 SID token ===
                generated = raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_new_tokens=eval_dataset.n_digit + 1,
                )
            if generated.ndim != 2 or generated.shape[0] != batch_size * beam_size:
                raise ValueError(
                    "Generation must return [batch * beam, sequence] when using "
                    f"num_return_sequences=beam; got {tuple(generated.shape)}"
                )
            generated = generated.reshape(batch_size, beam_size, -1)
            sid_tokens = generated[:, :, 1 : 1 + eval_dataset.n_digit]
            if sid_tokens.shape[-1] != eval_dataset.n_digit:
                raise ValueError(
                    f"Generation returned SID width {sid_tokens.shape[-1]}, "
                    f"expected {eval_dataset.n_digit}"
                )

            # RecBole maps complete generated SID tuples to items, then
            # removes invalid/duplicate/seen items before taking top-k.
            # Only beam_size (normally 20) tuples per example cross to CPU.
            sid_tokens_cpu = sid_tokens.detach().cpu()
            target_items = batch.get("target_item")
            if target_items is None:
                target_sid_tokens = batch["labels"][:, : eval_dataset.n_digit]
                target_items = torch.as_tensor(
                    [
                        eval_dataset.code_to_item.get(
                            tuple(int(v) for v in row.tolist()), -1
                        )
                        for row in target_sid_tokens
                    ],
                    dtype=torch.long,
                )
            seen_items = batch.get("seen_item_ids")
            seen_mask = batch.get("seen_mask")
            selected_rows = []
            for row_index in range(batch_size):
                excluded = set()
                if exclude_history and seen_items is not None and seen_mask is not None:
                    excluded = {
                        int(item_id)
                        for item_id, keep in zip(
                            seen_items[row_index].tolist(),
                            seen_mask[row_index].tolist(),
                        )
                        if keep
                    }
                selected = []
                selected_set = set()
                for beam_index in range(beam_size):
                    token_tuple = tuple(
                        int(v) for v in sid_tokens_cpu[row_index, beam_index].tolist()
                    )
                    item_id = eval_dataset.code_to_item.get(token_tuple)
                    if item_id is None or item_id in selected_set or item_id in excluded:
                        continue
                    selected.append(item_id)
                    selected_set.add(item_id)
                    if len(selected) == max_k:
                        break
                if len(selected) < max_k:
                    for item_id in eval_dataset.fallback_item_ids:
                        if item_id in selected_set or item_id in excluded:
                            continue
                        selected.append(item_id)
                        selected_set.add(item_id)
                        if len(selected) == max_k:
                            break
                if len(selected) < max_k:
                    raise RuntimeError(
                        f"Could not produce {max_k} item predictions for eval row {row_index}"
                    )
                selected_rows.append(selected)

            target_items_cpu = target_items.detach().cpu().tolist()
            for row_index, selected in enumerate(selected_rows):
                target_item = int(target_items_cpu[row_index])
                try:
                    rank_value = selected.index(target_item)
                except ValueError:
                    rank_value = None
                for k in topk_list:
                    if rank_value is not None and rank_value < k:
                        recall_sums[k] += 1.0
                        ndcg_sums[k] += 1.0 / math.log2(rank_value + 2.0)
            n_eval += batch_size

    if n_eval == 0:
        raise ValueError("Cannot evaluate an empty split")
    # RecBole's metric registry uses lowercase result keys.
    recalls = {f"recall@{k}": recall_sums[k] / n_eval for k in topk_list}
    ndcgs = {f"ndcg@{k}": ndcg_sums[k] / n_eval for k in topk_list}
    return recalls, ndcgs, n_eval


def _build_dataset(config, filename, mode, code_file):
    dataset_dir = Path(config["dataset_path"]) / config["dataset_name"]
    return GenRecDataset(
        dataset_path=_dataset_file(dataset_dir, filename),
        code_path=code_file,
        mode=mode,
        codebook_size=config["codebook_size"],
        max_len=config["max_len"],
        PAD_TOKEN=config["pad_token_id"] if "pad_token_id" in config else 0,
        n_user_tokens=config["n_user_tokens"],
    )


def _load_validation_curve(path):
    """Read validation Recall/NDCG pairs from a previous run's log.

    The checkpoint is never loaded and no weights are reused.  The curve is
    only a cheap decision boundary for rejecting an unpromising SID candidate
    before spending the full T5 schedule on it.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"screen baseline log does not exist: {path}")
    curve = {}
    current_epoch = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = re.search(r"Epoch\s+(\d+)/", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        if current_epoch is None:
            continue
        recall_match = re.search(r"recall@10['\"]?:\s*([0-9.eE+-]+)", line)
        if recall_match and "Validation recall" in line:
            curve.setdefault(current_epoch, {})["recall@10"] = float(recall_match.group(1))
        ndcg_match = re.search(r"ndcg@10['\"]?:\s*([0-9.eE+-]+)", line)
        if ndcg_match and "Validation ndcg" in line:
            curve.setdefault(current_epoch, {})["ndcg@10"] = float(ndcg_match.group(1))
    return {
        epoch: metrics
        for epoch, metrics in curve.items()
        if "recall@10" in metrics and "ndcg@10" in metrics
    }


def _baseline_metrics_at(curve, epoch):
    """Use the same epoch when available, otherwise the latest earlier one."""
    if epoch in curve:
        return curve[epoch]
    earlier = [candidate for candidate in curve if candidate <= epoch]
    return curve[max(earlier)] if earlier else None


def main():
    # === 训练/模型配置: 0 argparse, 全部硬编码模块常量 (CLAUDE.md §4) ===
    config = {
        # 训练/吞吐
        "batch_size":       BATCH_SIZE,
        "infer_size":       INFER_SIZE,
        "num_epochs":       NUM_EPOCHS,
        "eval_interval":    EVAL_INTERVAL,
        "eval_start_epoch": EVAL_START_EPOCH,
        "early_eval_epochs": EARLY_EVAL_EPOCHS,
        "early_stop":       EARLY_STOP,
        "fast":             FAST,
        "bf16":             BF16,
        "use_lr_scheduler": USE_LR_SCHEDULER,
        "lr_warmup_pct":    LR_WARMUP_PCT,
        "lr_min_factor":    LR_MIN_FACTOR,
        "compile":          COMPILE,
        "compile_mode":     COMPILE_MODE,
        # 模型
        "num_layers":           NUM_LAYERS,
        "num_decoder_layers":   NUM_DECODER_LAYERS,
        "d_model":              D_MODEL,
        "d_ff":                 D_FF,
        "num_heads":            NUM_HEADS,
        "d_kv":                 D_KV,
        "dropout_rate":         DROPOUT_RATE,
        "activation_function":  ACTIVATION_FUNCTION,
        "feed_forward_proj":    FEED_FORWARD_PROJ,
        "device":               DEVICE,
        # Optimizer
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
        # Token
        "vocab_size":    VOCAB_SIZE,
        "eos_token_id":  EOS_TOKEN_ID,
        "pad_token_id":  PAD_TOKEN_ID,
        "n_user_tokens": N_USER_TOKENS,
        "max_len":       MAX_LEN,
        # 数据/路径
        "dataset_name":  DATASET_NAME,
        "dataset_path":  DATASET_PATH,
        "codebook_size": CODEBOOK_SIZE,
        "code_path":     CODE_PATH,
        "train_file":    TRAIN_FILE,
        "valid_file":    VALID_FILE,
        "test_file":     TEST_FILE,
        "log_path":      LOG_PATH,
        "save_path":     SAVE_PATH,
        "seed":          SEED,
        # 评估/Screen
        "topk_list":                TOPK_LIST,
        "beam_size":                BEAM_SIZE,
        "no_eval":                  NO_EVAL,
        "skip_test":                SKIP_TEST,
        "screen_baseline_log":      SCREEN_BASELINE_LOG,
        "screen_start_epoch":       SCREEN_START_EPOCH,
        "screen_window":            SCREEN_WINDOW,
        "screen_min_ndcg_gain":     SCREEN_MIN_NDCG_GAIN,
        "screen_min_recall_gain":   SCREEN_MIN_RECALL_GAIN,
        "screen_skip_test_on_fail": SCREEN_SKIP_TEST_ON_FAIL,
        # DataLoader (含拍平别名, 供 line 576-582 runtime 派生)
        "num_workers":         NUM_WORKERS,
        "prefetch_factor":     PREFETCH_FACTOR,
        "exclude_history":     EXCLUDE_HISTORY,
        "no_exclude_history":  NO_EXCLUDE_HISTORY,
        "pin_memory":          PIN_MEMORY,
        "no_pin_memory":       NO_PIN_MEMORY,
    }
    if config["n_user_tokens"] <= 0:
        raise ValueError("RecBole TIGER requires --n_user_tokens >= 1")
    if 10 not in config["topk_list"]:
        raise ValueError("topk_list must contain 10 because RecBole monitors ndcg@10")
    if config["beam_size"] < max(config["topk_list"]):
        raise ValueError("beam_size must be at least the largest requested top-k")
    if config["num_workers"] < 0 or config["prefetch_factor"] <= 0:
        raise ValueError("num_workers must be >= 0 and prefetch_factor must be > 0")
    if config["eval_interval"] <= 0 and not config["no_eval"]:
        raise ValueError("eval_interval must be a positive integer unless --no_eval is set")
    if config["eval_start_epoch"] <= 0 and not config["no_eval"]:
        raise ValueError("eval_start_epoch must be positive unless --no_eval is set")
    if config["screen_start_epoch"] <= 0 or config["screen_window"] <= 0:
        raise ValueError("screen_start_epoch and screen_window must be positive")
    if config["screen_min_ndcg_gain"] < 0 or config["screen_min_recall_gain"] < 0:
        raise ValueError("screen gains must be non-negative")

    if config["fast"]:
        # The default remains deterministic for reproducibility.  This opt-in
        # path is useful on A100 when throughput is the priority.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if config["device"].startswith("cuda") and torch.cuda.is_available():
            device = torch.device(config["device"])
        else:
            device = torch.device("cpu")
    if config["bf16"]:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("--bf16 requires a CUDA GPU with BF16 support")
    set_seed(config["seed"])

    screen_curve = _load_validation_curve(config["screen_baseline_log"])
    if config["screen_baseline_log"] and not screen_curve:
        raise ValueError(
            f"No complete Validation recall/ndcg pairs found in {config['screen_baseline_log']}"
        )

    dataset_dir = Path(config["dataset_path"]) / config["dataset_name"]
    code_file = _resolve_code_file(dataset_dir, config["code_path"])
    train_dataset = _build_dataset(config, config["train_file"], "train", code_file)
    valid_dataset = (
        _build_dataset(config, config["valid_file"], "evaluation", code_file)
        if not config["no_eval"]
        else None
    )

    # Runtime-derived config (就地赋值, 模块常量不动; line 584 之后无代码读 no_exclude_history/no_pin_memory, 故保留可见性)
    config["vocab_size"] = train_dataset.vocab_size if config["vocab_size"] == 0 else config["vocab_size"]
    config["eos_token_id"] = train_dataset.eos_token if config["eos_token_id"] == 0 else config["eos_token_id"]
    config["sid_length"] = train_dataset.n_digit
    config["max_token_seq_len"] = train_dataset.max_token_seq_len
    config["decoder_start_token_id"] = config["pad_token_id"]
    config["exclude_history"] = not config["no_exclude_history"]
    config["pin_memory"] = device.type == "cuda" and not config["no_pin_memory"]

    if rank == 0:
        cur_time = get_local_time()
        log_path = os.path.join(config["log_path"], config["dataset_name"], cur_time)
        ckpt_path = os.path.join(config["save_path"], config["dataset_name"], cur_time)
        ensure_dir(log_path)
        ensure_dir(ckpt_path)
        logging.basicConfig(
            filename=os.path.join(log_path, "HG_Rec.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info("Configuration: %s", config)
        print(
            f"[RecBole-aligned] train={len(train_dataset)} "
            f"valid={len(valid_dataset) if valid_dataset is not None else 'skipped'} "
            f"num_items={train_dataset.num_items} vocab={config['vocab_size']} "
            f"eos={config['eos_token_id']} n_digit={train_dataset.n_digit} "
            f"beam={config['beam_size']} topk={config['topk_list']} exclude_history={config['exclude_history']}",
            flush=True,
        )
        if config["bf16"]:
            print("[AMP] BF16 autocast enabled for train/eval", flush=True)
    else:
        log_path = ckpt_path = None

    train_collator = make_sample_collator(train_dataset)
    eval_collator = (
        make_sample_collator(valid_dataset, include_eval_metadata=True)
        if valid_dataset is not None
        else None
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config["seed"],
            drop_last=False,
        )
        if is_ddp
        else None
    )
    loader_options = _loader_options(config)
    train_loader = GenRecDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        sample_collator=train_collator,
        max_token_seq_len=train_dataset.max_token_seq_len,
        **loader_options,
    )
    valid_sampler = (
        DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=config["seed"],
            drop_last=False,
        )
        if is_ddp and valid_dataset is not None
        else None
    )
    valid_loader = (
        GenRecDataLoader(
            valid_dataset,
            batch_size=config["infer_size"],
            shuffle=False,
            sampler=valid_sampler,
            sample_collator=eval_collator,
            max_token_seq_len=valid_dataset.max_token_seq_len,
            include_seen=True,
            **loader_options,
        )
        if valid_dataset is not None
        else None
    )
    # === R7-fix: rank-0-only validation 用的 full (non-sharded) loader ===
    # 4 卡 DDP 时 valid_sampler 是 DistributedSampler, 分片给 4 个 rank;
    # 这里给 rank 0 一个非分片的全集 loader, 单卡 eval 全部 valid 用户,
    # 彻底绕开 NCCL all_reduce 在多 NUMA 拓扑上的死锁.
    valid_loader_full = (
        GenRecDataLoader(
            valid_dataset,
            batch_size=config["infer_size"],
            shuffle=False,
            sampler=None,
            sample_collator=eval_collator,
            max_token_seq_len=valid_dataset.max_token_seq_len,
            include_seen=True,
            **loader_options,
        )
        if valid_dataset is not None and rank == 0
        else None
    )

    model = HG_Rec(config).to(device)
    train_model = model
    if config["compile"]:
        # DDP + torch.compile works since PyTorch 2.0; only the legacy
        # `--fast` benchmark path was single-GPU only. We removed the guard
        # so users can opt into compile on multi-GPU DDP runs.
        # genrec_env currently contains a newer torch package together with an
        # older top-level functorch compatibility package.  torch.compile
        # imports functorch.compile, whose eager import expects this debug-only
        # symbol even though current inductor paths do not use it.  Bridge the
        # renamed symbol locally instead of changing the shared environment.
        import torch._functorch.partitioners as _partitioners
        if (
            not hasattr(_partitioners, "draw_joint_graph")
            and hasattr(_partitioners, "draw_graph")
        ):
            _partitioners.draw_joint_graph = _partitioners.draw_graph
            if rank == 0:
                print(
                    "[CompileCompat] aliased draw_joint_graph -> draw_graph",
                    flush=True,
                )
        train_model = torch.compile(model, mode=config["compile_mode"], dynamic=False)
        if rank == 0:
            logging.info("torch.compile enabled for training with mode=%s", config["compile_mode"])
            print(
                f"[Compile] torch.compile enabled for training ({config['compile_mode']}); evaluation remains uncompiled",
                flush=True,
            )
    if rank == 0:
        print(model.n_parameters, flush=True)
        logging.info(model.n_parameters)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    # === iter12: cosine LR schedule + 5% warmup ===
    scheduler = None
    if config.get("use_lr_scheduler", False):
        from torch.optim.lr_scheduler import LambdaLR
        import math
        total_epochs = int(config["num_epochs"])
        warmup_epochs = max(1, int(total_epochs * config["lr_warmup_pct"]))
        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # linear warmup: lr/10 → lr
                return (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(config["lr_min_factor"] + (1.0 - config["lr_min_factor"]) * cosine)
        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

    best_monitor = -1.0
    best_train_loss = float("inf")
    early_stop_counter = 0
    best_checkpoint = None
    screen_best_ndcg_gain = float("-inf")
    screen_best_recall_gain = float("-inf")
    screen_decided = not bool(screen_curve)
    screen_passed = None
    screen_failed = False
    screen_history = []
    # === R7-fix v6: 共享信号目录 (所有 rank 都知道, 与 ckpt_path 无关) ===
    _ddp_sync_dir = os.path.join(config["log_path"], "_ddp_sync")
    if rank == 0:
        os.makedirs(_ddp_sync_dir, exist_ok=True)
    _early_stop_file = os.path.join(_ddp_sync_dir, "early_stop_signal.txt")
    _shutdown_file = os.path.join(_ddp_sync_dir, "shutdown_signal.txt")
    for epoch in range(1, config["num_epochs"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        train_loss = train(
            train_model,
            train_loader,
            optimizer,
            device,
            epoch,
            use_bf16=config["bf16"],
        )
        if scheduler is not None:
            scheduler.step()
        if rank == 0:
            logging.info("Epoch %d/%d train_loss=%s", epoch, config["num_epochs"], train_loss)
            cur_lr = optimizer.param_groups[0]["lr"]
            n_ep = config["num_epochs"]
            es = config["early_stop"]
            print(
                f"[train] epoch={epoch}/{n_ep} done loss={train_loss:.6f} lr={cur_lr:.3e} "
                f"patience={early_stop_counter}/{es} best_recall@10={best_monitor:.6f}",
                flush=True,
            )

        if is_ddp:
            dist.barrier()
        early_stop_signal = False
        should_validate = False
        if config["no_eval"] and rank == 0:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                early_stop_counter = 0
                best_checkpoint = os.path.join(ckpt_path, "HG_Rec_best.pth")
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(raw_model.state_dict(), best_checkpoint)
                logging.info("Best train loss=%s; saved %s", best_train_loss, best_checkpoint)
            else:
                early_stop_counter += 1
                logging.info(
                    "No train loss improvement; patience=%d",
                    early_stop_counter,
                )
                early_stop_signal = early_stop_counter >= config["early_stop"]
        elif not config["no_eval"] and rank == 0:
            should_validate = (
                epoch >= config["eval_start_epoch"]
                and (epoch - config["eval_start_epoch"]) % config["eval_interval"] == 0
            ) or (epoch == config["num_epochs"]) or (epoch in config["early_eval_epochs"])
        if not config["no_eval"] and should_validate:
            # === R7-fix v2: 完全消除 validation 阶段的所有 NCCL collective ===
            # 上一版还保留了 dist.barrier() + dist.broadcast(monitor_tensor, src=0),
            # 在多 NUMA 拓扑上仍会死锁 (cuStreamSynchronize 60+ min).
            # 现在: rank 0 单卡 eval, 写结果到共享文件, 其他 rank 完全跳过 collective,
            # 通过轮询文件获取 early_stop signal.
            # 信号文件目录: 所有 rank 都能计算 (LOG_PATH 硬编码, 与 ckpt_path 无关)
            if rank == 0:
                _t_valid_start = time.time()
                n_ep = config["num_epochs"]
                print(
                    f"[valid] epoch={epoch}/{n_ep} START (single_rank beam={config['beam_size']} topk={config['topk_list']})",
                    flush=True,
                )
                recalls, ndcgs, n_eval_local = evaluate(
                    model,
                    valid_loader_full,
                    valid_dataset,
                    config["topk_list"],
                    config["beam_size"],
                    device,
                    exclude_history=config["exclude_history"],
                    use_bf16=config["bf16"],
                )
                _t_valid_elapsed = time.time() - _t_valid_start
                n_eval_total = n_eval_local
                print(
                    f"[valid] epoch={epoch} rank=0/0 DONE "
                    f"n_eval={n_eval_total} elapsed={_t_valid_elapsed:.1f}s",
                    flush=True,
                )
                logging.info("Validation recall=%s", recalls)
                logging.info("Validation ndcg=%s", ndcgs)
                print(
                    f"[valid] epoch={epoch} n_eval={n_eval_total} recall={recalls} ndcg={ndcgs}",
                    flush=True,
                )
                monitor_value = recalls["recall@10"]
                if epoch in config["early_eval_epochs"]:
                    # iter17: 早期诊断 eval, 仅记录, 不影响 best_monitor / early_stop / ckpt
                    logging.info(
                        "[early-val diagnostic] epoch=%d n_eval=%d recall=%s ndcg=%s",
                        epoch, n_eval_total, recalls, ndcgs,
                    )
                    print(
                        f"[early-val] epoch={epoch} n_eval={n_eval_total} "
                        f"recall={recalls} ndcg={ndcgs} (diagnostic only, no best_monitor / early_stop impact)",
                        flush=True,
                    )
                    # 写 "0" 信号文件以防其他 rank 进入 polling (向后兼容)
                    with open(_early_stop_file, "w", encoding="utf-8") as _fh:
                        _fh.write(f"{epoch}\t0\n")
                elif monitor_value > best_monitor:
                    best_monitor = monitor_value
                    early_stop_counter = 0
                    best_checkpoint = os.path.join(ckpt_path, "HG_Rec_best.pth")
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), best_checkpoint)
                    logging.info("Best Recall@10=%s; saved %s", best_monitor, best_checkpoint)
                    print(
                        f"[valid] epoch={epoch} new best Recall@10={best_monitor:.6f} -> {best_checkpoint}",
                        flush=True,
                    )
                    # === 写文件给其他 rank 看 (替代 dist.broadcast) ===
                    with open(_early_stop_file, "w", encoding="utf-8") as _fh:
                        _fh.write(f"{epoch}\t0\n")  # 0 = 继续训练
                else:
                    early_stop_counter += 1
                    logging.info("No Recall@10 improvement; patience=%d", early_stop_counter)
                    print(
                        f"[valid] epoch={epoch} no Recall@10 improvement (current={monitor_value:.6f}, best={best_monitor:.6f}, patience={early_stop_counter}/{config['early_stop']})",
                        flush=True,
                    )
                    _should_stop = early_stop_counter >= config["early_stop"]
                    with open(_early_stop_file, "w", encoding="utf-8") as _fh:
                        _fh.write(f"{epoch}\t{1 if _should_stop else 0}\n")
                    if _should_stop:
                        early_stop_signal = True
            else:
                # === 非 rank 0: 完全不参与 validation, 不调任何 NCCL collective ===
                # 通过轮询 _early_stop_file 文件获取 rank 0 的决策
                # 等文件出现 (rank 0 eval 大约 25s)
                _wait_start = time.time()
                while not os.path.exists(_early_stop_file):
                    if time.time() - _wait_start > 120:  # 最长等 2 min
                        raise RuntimeError(
                            f"rank {rank} waited 120s for _early_stop_file but it never appeared"
                        )
                    time.sleep(1.0)
                # 读取 rank 0 的信号
                with open(_early_stop_file, "r", encoding="utf-8") as _fh:
                    _sig_line = _fh.readline().strip()
                _sig_epoch, _sig_val = _sig_line.split("\t")
                # 只采纳最新信号 (rank 0 当前 epoch 的)
                if int(_sig_epoch) == epoch:
                    if int(_sig_val) == 1:
                        early_stop_signal = True

            if rank == 0 and screen_curve and epoch >= config["screen_start_epoch"] and not screen_decided:
                baseline = _baseline_metrics_at(screen_curve, epoch)
                if baseline is not None:
                    ndcg_gain = ndcgs["ndcg@10"] - baseline["ndcg@10"]
                    recall_gain = recalls["recall@10"] - baseline["recall@10"]
                    screen_best_ndcg_gain = max(screen_best_ndcg_gain, ndcg_gain)
                    screen_best_recall_gain = max(screen_best_recall_gain, recall_gain)
                    screen_history.append(
                        {
                            "epoch": epoch,
                            "baseline_ndcg@10": baseline["ndcg@10"],
                            "baseline_recall@10": baseline["recall@10"],
                            "candidate_ndcg@10": ndcgs["ndcg@10"],
                            "candidate_recall@10": recalls["recall@10"],
                            "ndcg_gain": ndcg_gain,
                            "recall_gain": recall_gain,
                        }
                    )
                    logging.info(
                        "Screen epoch=%d baseline_ndcg@10=%.6f candidate_ndcg@10=%.6f "
                        "gain=%.6f baseline_recall@10=%.6f candidate_recall@10=%.6f "
                        "gain=%.6f best_gains=(%.6f, %.6f)",
                        epoch,
                        baseline["ndcg@10"],
                        ndcgs["ndcg@10"],
                        ndcg_gain,
                        baseline["recall@10"],
                        recalls["recall@10"],
                        recall_gain,
                        screen_best_ndcg_gain,
                        screen_best_recall_gain,
                    )
                    screen_epoch_count = len(screen_history)
                    if screen_epoch_count >= config["screen_window"]:
                        screen_passed = (
                            screen_best_ndcg_gain >= config["screen_min_ndcg_gain"]
                            and screen_best_recall_gain >= config["screen_min_recall_gain"]
                        )
                        screen_decided = True
                        screen_failed = not screen_passed
                        if screen_passed:
                            logging.info(
                                "SCREEN PASS: best gains NDCG@10=%.6f Recall@10=%.6f",
                                screen_best_ndcg_gain,
                                screen_best_recall_gain,
                            )
                        else:
                            logging.warning(
                                "SCREEN FAIL: best gains NDCG@10=%.6f Recall@10=%.6f; "
                                "required=(%.6f, %.6f)",
                                screen_best_ndcg_gain,
                                screen_best_recall_gain,
                                config["screen_min_ndcg_gain"],
                                config["screen_min_recall_gain"],
                            )
                            early_stop_signal = True
        # === R7-fix v3: 同步 early_stop_signal (rank 0 已写文件, 其他 rank 已轮询),
        # 不再调 dist.broadcast (多 NUMA 拓扑上会死锁). ===
        if early_stop_signal:
            break

    if rank == 0 and screen_curve:
        screen_summary = {
            "passed": screen_passed,
            "failed": screen_failed,
            "screen_start_epoch": config["screen_start_epoch"],
            "screen_window": config["screen_window"],
            "min_ndcg_gain": config["screen_min_ndcg_gain"],
            "min_recall_gain": config["screen_min_recall_gain"],
            "best_ndcg_gain": screen_best_ndcg_gain,
            "best_recall_gain": screen_best_recall_gain,
            "history": screen_history,
        }
        with open(os.path.join(log_path, "screen_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(screen_summary, handle, indent=2)
        if screen_failed and config["screen_skip_test_on_fail"]:
            logging.info("Final test skipped because the early baseline screen failed")
            print("[RecBole-aligned] screen failed; final test skipped", flush=True)
            # === R7-fix v4: rank 0 写 shutdown 文件, 其他 rank 轮询退出 (避开 NCCL barrier 死锁) ===
            if rank == 0:
                with open(_shutdown_file, "w", encoding="utf-8") as _fh:
                    _fh.write("screen_failed_skip_test\n")
            if rank != 0:
                _w = time.time()
                while not os.path.exists(_shutdown_file):
                    if time.time() - _w > 1800:
                        break  # 30 min 安全上限
                    time.sleep(2.0)
            return

    if config["skip_test"]:
        if rank == 0:
            logging.info("Final test skipped by --skip_test")
            print("[RecBole-aligned] final test skipped (--skip_test)", flush=True)
            with open(_shutdown_file, "w", encoding="utf-8") as _fh:
                _fh.write("skip_test\n")
        if rank != 0:
            _w = time.time()
            while not os.path.exists(_shutdown_file):
                if time.time() - _w > 1800:
                    break
                time.sleep(2.0)
        return

    # RecBole reloads the best checkpoint and evaluates test automatically.
    # === R7-fix v4: rank 0 单独做 test eval, 其他 rank 等 shutdown 文件 ===
    if rank == 0:
        test_dataset = _build_dataset(config, config["test_file"], "evaluation", code_file)
        test_loader = GenRecDataLoader(
            test_dataset,
            batch_size=config["infer_size"],
            shuffle=False,
            sample_collator=make_sample_collator(
                test_dataset, include_eval_metadata=True
            ),
            max_token_seq_len=test_dataset.max_token_seq_len,
            include_seen=True,
            **loader_options,
        )
        raw_model = model.module if hasattr(model, "module") else model
        if best_checkpoint is not None and os.path.exists(best_checkpoint):
            raw_model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        test_recalls, test_ndcgs, _n_test = evaluate(
            raw_model,
            test_loader,
            test_dataset,
            config["topk_list"],
            config["beam_size"],
            device,
            exclude_history=config["exclude_history"],
            use_bf16=config["bf16"],
        )
        test_result = {
            "best_checkpoint": best_checkpoint,
            "n_eval": len(test_dataset),
            **{f"test_{key}": value for key, value in test_recalls.items()},
            **{f"test_{key}": value for key, value in test_ndcgs.items()},
        }
        logging.info("Test=%s", test_result)
        with open(os.path.join(log_path, "test_final.json"), "w", encoding="utf-8") as handle:
            json.dump(test_result, handle, indent=2)
        print(f"[RecBole-aligned] test={test_result}", flush=True)
        # === rank 0 完成 test eval, 通知其他 rank 退出 ===
        with open(_shutdown_file, "w", encoding="utf-8") as _fh:
            _fh.write("test_done\n")
    else:
        _w = time.time()
        while not os.path.exists(_shutdown_file):
            if time.time() - _w > 1800:
                break  # 30 min 安全上限
            time.sleep(2.0)
    # === 全部 rank 直接 return, 不调 dist.destroy_process_group (NCCL 退出也可能死锁) ===
    return


# === 4 卡 DDP launcher: 自动 fork torchrun (硬编码, 无 CLI flags) ===
import subprocess as _subprocess
import sys as _sys

_LAUNCHER = {
    "torchrun":   "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "script":     os.path.abspath(__file__),
    "nproc":      4,                      # 4 卡 DDP (R36n 历史: 单卡 16s/epoch, DDP 应等效)
    "master_port": 50201,
    "log":        "logs/_stage3_launcher.log",
    "visible_dev": "0,1,2,3",            # 4 卡
}


def _launch_via_torchrun():
    """Fork torchrun subprocess with hard-coded flags."""
    log_dir = os.path.dirname(_LAUNCHER["log"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    cmd = [
        _LAUNCHER["torchrun"],
        "--standalone",
        f"--nproc_per_node={_LAUNCHER['nproc']}",
        f"--master_port={_LAUNCHER['master_port']}",
        _LAUNCHER["script"],
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = _LAUNCHER["visible_dev"]
    # === NCCL 环境 (多 NUMA 4 卡, 2 个 mlx5 NIC, 无 NVLink) ===
    # 之前 (2026-09-17 之前) 把 IB/P2P/SHM 三个全关, 等于封死所有 NCCL transport,
    # 在跨 NUMA 节点上会导致 all_reduce 死锁 (rank 0 卡 cuMemcpyDtoHAsync_v2).
    # 修复: 开启 IB 让 NCCL 走 mlx5 NIC, 保留 P2P 关闭 (无 NVLink), SHM 默认开.
    env.setdefault("NCCL_IB_DISABLE", "0")  # 走 IB/RoCE via mlx5_0/mlx5_1
    env.setdefault("NCCL_P2P_DISABLE", "1")  # 无 NVLink, P2P 强制 PCIe 反而慢且不稳
    env.setdefault("NCCL_SHM_DISABLE", "0")  # 节点内最快通道, 默认开
    env.setdefault("NCCL_NET_GDR_LEVEL", "0")  # 关闭 GPU Direct RDMA, 多 NUMA 兼容性更好
    env.setdefault("NCCL_SOCKET_IFNAME", "lo")  # 进程间协调走 lo, 数据走 IB
    env.setdefault("NCCL_TIMEOUT", "3600")  # 1h (集体大 tensor 慢, 不要被 timeout kill)
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")  # 阻塞 wait, 出问题时有 stack 而不是 timeout
    env.setdefault("NCCL_DEBUG", "WARN")  # INFO 太噪; WARN 足够暴露卡死的 transport
    with open(_LAUNCHER["log"], "wb") as fout:
        rc = _subprocess.call(cmd, stdout=fout, stderr=_subprocess.STDOUT, env=env)
    _sys.exit(rc)


if __name__ == "__main__":
    # Already a torchrun worker (RANK set) -> main directly.
    # Otherwise, fork torchrun.
    if "RANK" in os.environ:
        try:
            main()
        finally:
            cleanup_distributed()
    else:
        _launch_via_torchrun()
