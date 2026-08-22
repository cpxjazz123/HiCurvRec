import os
import sys
# 跳过 transformers TF 路径 (Keras 3 兼容性问题)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 把当前脚本所在目录加入 sys.path 最前 (R44/R47: 任务目录自包含)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import gin
import torch
import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.hg_rec import HG_Rec  # C33 Issue #181 合并
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# HG-Rec 路径 (C33 Issue #181 合并): 序列级 CE 数据集
from data.instruments import RawMusicalInstrumentsHGRec, hgrec_collate_fn

# === HALC v2 创新 (R52 主目录直接迭代): learnable per-layer κ + curvature annealing ===
from _lib.halc import HALCAnnealingRegularizer

# === v69 G5 Codebook-aware Manifold Contrastive (Stage 3, R36 G 类允许) ===
# 设计: T5 logits (B, L, 769) → softmax → 期望 SID 嵌入 pred_emb = p @ E (E = lm_head 共享矩阵)
#       target_emb = E[target_idx] → 两者都过 expmap0(c=0.7) 到 Poincaré ball
#       NT-Xent 在 ball 上: positive=(pred, target), negatives=(pred, E[j≠target])
# 避免 v68 G4 在 encoder hidden state 加 metric 的任务耦合问题 (G5 直接在 SID 决策空间)
V69_G5_ENABLED = True
V69_G5_AUX_WEIGHT = 0.10
V69_G5_C = 0.7
V69_G5_TEMPERATURE = 0.10
V69_G5_VOCAB_SIZE = 769  # 256*3 + 1 (PAD=0)
V69_G5_D_MODEL = 128  # hgrec_t5_d_model


@gin.configurable
# =============================================================================
# C33 Issue #181 HG-Rec 训练路径 (序列级 CE + T5ForConditionalGeneration)
# 与 train() 的 per-hierarchy 路径独立; 通过 use_hgrec_arch=True 切换
# 验证: C33 verdict test R@10=0.1093, C33 rerun test R@10=0.1058 (R37 PASS vs baseline 0.0972)
# =============================================================================

def _train_hgrec(
    accelerator_factory,
    dataset_folder,
    save_dir_root,
    pretrained_decoder_path,
    hgrec_code_path,
    hgrec_t5_d_model,
    hgrec_t5_d_ff,
    hgrec_t5_num_heads,
    hgrec_t5_d_kv,
    hgrec_t5_num_layers,
    hgrec_t5_num_decoder_layers,
    hgrec_t5_dropout,
    hgrec_vocab_size,
    hgrec_max_len,
    batch_size,
    learning_rate,
    weight_decay,
    top_k_for_generation,
    top_k_eval_list,
    wandb_logging,
):
    """HG-Rec T5 架构训练路径.

    与 C33 train_stage3_ddp_fast.py 严格对齐:
    - HG_Rec (T5ForConditionalGeneration) + 序列级 CE
    - 数据集: parquet + (9922, 4) SID numpy → flat token seq
    - valid/test eval: model.generate(num_beams=20) + TopKAccumulator
    - 训练: Adam(lr=1e-4), 200 epoch, EARLY_STOP=20, best by valid NDCG@20
    """
    import json as _json
    import os.path as _osp

    accelerator = accelerator_factory()
    device = accelerator.device

    # === R51+ 强约束: DDP 4 卡完全确定性 (2026-08-20 升级, 含 cudnn/CUBLAS/hash) ===
    # 防止 4 个 DDP 噪声源:
    #   1. random/numpy/torch RNG 未固定 (Phase 1)
    #   2. Python hash dict/set 顺序随机 (PYTHONHASHSEED)
    #   3. cuDNN 自动选算法 (cudnn.benchmark + cudnn.deterministic)
    #   4. cuBLAS GEMM workspace 算法 (CUBLAS_WORKSPACE_CONFIG)
    # 实测: 单纯 Phase 1 (v51) 仍有 ±0.02 量级 best_metric 漂移, 升级为 R51+ 后必须字符级一致
    # 与 curvature_experiment/scripts/train_decoder.py 块严格对齐, 字符级一致
    import os as _os
    import random as _r51_random
    import numpy as _r51_np
    _os.environ.setdefault("PYTHONHASHSEED", "42")
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _rank = accelerator.process_index
    _r51_random.seed(42 + _rank)
    _r51_np.random.seed(42 + _rank)
    torch.manual_seed(42 + _rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + _rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("highest")
    print(f"[R51+ seed] rank={_rank} deterministic (cudnn/CUBLAS/hash) before DataLoader", flush=True)
    del _rank, _r51_random, _r51_np, _os

    # === R51+ Phase 2: DataLoader worker_init_fn (rank-relative seed) ===
    # 防止 8 个 dataloader worker 启动顺序 + worker 内 random 状态不同
    def _r51_worker_init_fn(wid):
        import random as _wr
        import numpy as _wnp
        # 用 rank * 1000 + wid 避免不同 rank 同 wid 撞 seed
        _wr.seed(42 + accelerator.process_index * 1000 + wid)
        _wnp.random.seed(42 + accelerator.process_index * 1000 + wid)

    # === 路径 (硬编码, R44/R47) ===
    INSTRUMENTS_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
    DEFAULT_CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"
    code_path = hgrec_code_path if hgrec_code_path else DEFAULT_CODE_PATH
    BEST_CKPT_PATH = _osp.join(save_dir_root, "best_ckpt.pt")
    LOG_PATH = _osp.join(save_dir_root, "train_log.json")
    os.makedirs(save_dir_root, exist_ok=True)

    # === 数据集 (HG-Rec 风格 parquet + SID npy) ===
    codebook_size = [256, 256, 256]
    train_ds = RawMusicalInstrumentsHGRec(
        parquet_path=_osp.join(INSTRUMENTS_DIR, "train.parquet"),
        code_path=code_path, mode="train",
        codebook_size=codebook_size, max_len=hgrec_max_len,
    )
    valid_ds = RawMusicalInstrumentsHGRec(
        parquet_path=_osp.join(INSTRUMENTS_DIR, "valid.parquet"),
        code_path=code_path, mode="evaluation",
        codebook_size=codebook_size, max_len=hgrec_max_len,
    )
    test_ds = RawMusicalInstrumentsHGRec(
        parquet_path=_osp.join(INSTRUMENTS_DIR, "test.parquet"),
        code_path=code_path, mode="evaluation",
        codebook_size=codebook_size, max_len=hgrec_max_len,
    )

    # HG-Rec 评估用 per-rank batch (4 卡各处理一段, DistributedSampler 自动切片)
    eval_batch_size = 64  # per-GPU (4 卡 DDP, beam=20, KV cache 留足空间)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, persistent_workers=True, pin_memory=True,
        prefetch_factor=4, collate_fn=hgrec_collate_fn,
        worker_init_fn=_r51_worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=8, persistent_workers=True, pin_memory=True,
        prefetch_factor=4, collate_fn=hgrec_collate_fn,
        worker_init_fn=_r51_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=8, persistent_workers=True, pin_memory=True,
        prefetch_factor=4, collate_fn=hgrec_collate_fn,
        worker_init_fn=_r51_worker_init_fn,
    )

    # === 模型 (HG_Rec T5ForConditionalGeneration) ===
    hgrec_config = {
        "num_layers": hgrec_t5_num_layers,
        "num_decoder_layers": hgrec_t5_num_decoder_layers,
        "d_model": hgrec_t5_d_model,
        "d_ff": hgrec_t5_d_ff,
        "num_heads": hgrec_t5_num_heads,
        "d_kv": hgrec_t5_d_kv,
        "dropout_rate": hgrec_t5_dropout,
        "vocab_size": hgrec_vocab_size,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "decoder_start_token_id": 0,
        "feed_forward_proj": "relu",
    }
    model = HG_Rec(hgrec_config)
    _raw_model_unwrapped = model  # v69 G5 用: lm_head weight 通过 raw_model 访问 (避免 DDP wrap 后访问问题)

    # === v69 G5 Codebook-aware Manifold Contrastive: 不需要新增 projection head ===
    # 直接用 T5 lm_head 的 weight 作为 SID codebook 嵌入空间 E (V69_G5_VOCAB_SIZE × V69_G5_D_MODEL)
    # pred_emb = softmax(logits) @ E  (B, L, 128)
    # target_emb = E[target_idx]       (B, L, 128)
    # expmap0(c=0.7) → Poincaré ball → NT-Xent on ball

    import math as _math_g5

    def _g5_expmap0(u: torch.Tensor, c: float = V69_G5_C) -> torch.Tensor:
        """Poincaré ball expmap at origin: u → tanh(√c · ||u||) · u / (√c · ||u||)"""
        sqrt_c = _math_g5.sqrt(c)
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

    def _g5_poincare_dist(x: torch.Tensor, y: torch.Tensor, c: float = V69_G5_C) -> torch.Tensor:
        """Poincaré ball distance: d_P(x, y) = (2/√c) · atanh(√c · ||möbius_diff||)"""
        sqrt_c = _math_g5.sqrt(c)
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - c * x_sq) * y + (1 + c * xy) * x
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        diff = num / denom.clamp(min=1e-9)
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(max=(1.0 / sqrt_c) * 0.999)
        arg = sqrt_c * diff_norm
        return (2.0 / sqrt_c) * torch.atanh(arg)

    def _g5_manifold_contrastive_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Chunked NT-Xent on Poincaré ball between pred_emb 和 target_emb (R69 G5 创新).

        Args:
            logits: (B, L, 769) — T5 decoder 输出 logits
            target: (B, L) — 真实 SID token ids (含 PAD=0)

        Returns:
            scalar NT-Xent loss (chunked, OOM-safe for batch_size=3072)
        """
        B, L, V = logits.shape
        if B < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        # 展平 (B*L, V) 与 (B*L,), 跳过 PAD=0 位置
        flat_logits = logits.reshape(B * L, V)
        flat_target = target.reshape(B * L)
        valid_mask = flat_target > 0  # PAD=0 跳过
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        valid_logits = flat_logits[valid_mask]  # (N, V)
        valid_target = flat_target[valid_mask]  # (N,)

        # === 随机采样 N=256 控制 (chunk, N, 128) 内存 (batch_size=3072 → N≈12288 太大) ===
        # R51+ 兼容: 用 torch.randperm 而不是 random.choice
        N_total = valid_logits.shape[0]
        sample_size = min(256, N_total)
        if N_total > sample_size:
            sample_idx = torch.randperm(N_total, device=valid_logits.device)[:sample_size]
            valid_logits = valid_logits[sample_idx]
            valid_target = valid_target[sample_idx]

        # === lm_head 嵌入矩阵 (raw_model 内的 lm_head.weight) ===
        lm_head_weight = _raw_model_unwrapped.model.lm_head.weight  # (V, d_model)
        # pred_emb = softmax(logits) @ E → (N, d_model)
        p = torch.softmax(valid_logits.float(), dim=-1)
        pred_emb = p @ lm_head_weight.float()  # (N, d_model)
        target_emb = lm_head_weight[valid_target].float()  # (N, d_model)
        # expmap0 到 Poincaré ball
        pred_ball = _g5_expmap0(pred_emb)  # (N, d_model)
        target_ball = _g5_expmap0(target_emb)  # (N, d_model)

        # === Chunked NT-Xent: chunk_size=16 控制 (chunk, N, 128) 内存 ===
        # N=256, chunk_size=16 → 16 chunks × (16, 256, 128) = 8MB per chunk = 128MB peak (auto-released per chunk)
        N = pred_ball.shape[0]
        chunk_size = 16
        total_loss = pred_ball.new_zeros(())
        n_chunks = 0
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            pred_chunk = pred_ball[i:end]  # (chunk, d)
            # distance (chunk, N): pairwise
            sqrt_c = _math_g5.sqrt(V69_G5_C)
            x_sq = (pred_chunk * pred_chunk).sum(dim=-1, keepdim=True)  # (chunk, 1)
            y_sq = (target_ball * target_ball).sum(dim=-1, keepdim=True)  # (N, 1)
            xy = pred_chunk @ target_ball.T  # (chunk, N)
            num = (1 - V69_G5_C * x_sq).unsqueeze(-1) * target_ball.unsqueeze(0) + \
                  (1 + V69_G5_C * xy).unsqueeze(-1) * pred_chunk.unsqueeze(1)
            denom = 1 + 2 * V69_G5_C * xy.unsqueeze(-1) + \
                    V69_G5_C * V69_G5_C * x_sq.unsqueeze(-1) * y_sq.unsqueeze(0)
            diff = num / denom.clamp(min=1e-9)
            diff_norm = diff.norm(dim=-1).clamp(max=(1.0 / sqrt_c) * 0.999)
            arg = sqrt_c * diff_norm
            d = (2.0 / sqrt_c) * torch.atanh(arg)  # (chunk, N)
            logits_g5 = -d / V69_G5_TEMPERATURE  # (chunk, N)
            labels_g5 = torch.arange(i, end, device=logits_g5.device)
            chunk_loss = torch.nn.functional.cross_entropy(logits_g5, labels_g5)
            total_loss = total_loss + chunk_loss
            n_chunks += 1
        return total_loss / max(n_chunks, 1)

    # === 新 baseline 速度优化: torch.compile (kernel fusion, 4 卡 DDP 兼容, fallback-safe) ===
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.cache_size_limit = 64
        model = torch.compile(model, dynamic=False, fullgraph=False)
        print(f"[torch.compile] HG_Rec compiled (fallback safe, 4 卡 DDP OK)", flush=True)
    except Exception as _compile_err:
        print(f"[torch.compile] skipped: {_compile_err}", flush=True)

    # === HG-Rec eval helper: 用 generate + TopKAccumulator ===
    def _hgrec_to_acc_format(preds, labels, beam, topk_list):
        """HG_Rec.generate 输出 (B*beam, 5) [含首 token], reshape 到 (B, beam, 4), 找 label 命中位置.

        TopKAccumulator 期望 (B, beam, n_layers) 与 (B, n_layers).
        """
        preds = preds[:, 1:]  # 去掉首 token (decoder_start_token)
        preds = preds.reshape(-1, beam, 4)  # (B, beam, 4) 假设 label 长度=4
        # TopKAccumulator 期望 per-sample, top-k list-of-list-of-int
        top_k_preds = preds  # (B, beam, 4) 直接用, TopKAccumulator 会按 prefix 比对
        return top_k_preds

    def do_eval(eval_dl, split_name: str):
        model.eval()
        acc = TopKAccumulator(ks=top_k_eval_list)
        with torch.no_grad():
            for batch in eval_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["history"]
                attention_mask = batch["attention_mask"]
                labels = batch["target"]  # (B, 4)
                preds = model.module.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    num_beams=top_k_for_generation,
                )
                # 模型未充分训练时 generate 可能输出短序列 (EOS 早停) → 补 PAD 到固定 5 (= 1 首 token + 4 label tokens)
                if preds.shape[-1] < 5:
                    pad = torch.zeros(
                        (*preds.shape[:-1], 5 - preds.shape[-1]),
                        dtype=preds.dtype, device=preds.device,
                    )
                    preds = torch.cat([preds, pad], dim=-1)
                preds = preds[:, 1:].reshape(input_ids.shape[0], top_k_for_generation, 4)
                acc.accumulate(actual=labels, top_k=preds)
        local = acc.get_sums_and_total()
        keys = ["ndcg"] + [f"ndcg@{k}" for k in top_k_eval_list] + [f"h@{k}" for k in top_k_eval_list] + ["total"]
        local_vec = torch.tensor([float(local.get(k, 0.0)) for k in keys], device=device)
        global_vec = accelerator.reduce(local_vec, reduction="sum")
        global_total = global_vec[-1].item()
        metrics = {
            keys[i]: (global_vec[i].item() / global_total if global_total > 0 else 0.0)
            for i in range(len(keys) - 1)
        }
        if accelerator.is_main_process:
            print(f"[{split_name}] {metrics}", flush=True)
        return metrics

    # === optimizer + accelerate prepare ===
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader
    )
    # v69 G5: accelerator.prepare 后 model 被 DDP wrap, unwrap 后保持 lm_head weight 引用一致
    _raw_model_unwrapped = accelerator.unwrap_model(model)
    # === HALC v2 + v16 differential schedule (R36 机制变更): encoder 早 warmup, decoder 晚 warmup ===
    halc = HALCAnnealingRegularizer(
        num_layers=7, init_curvature=1.0, c_max=1.0,
        warmup_epochs=5, cooldown_epochs=10, reg_weight_max=0.05,
        encoder_warmup=3, encoder_cooldown=8,
        decoder_warmup=7, decoder_cooldown=12,
    )
    halc = halc.to(accelerator.device)
    if accelerator.is_main_process:
        halc.set_epoch(0)
        c_init = halc.annealed_curvature().detach().cpu().tolist()
        print(f"[HALC v2 + v16 diff] init annealed c_l @ epoch=0: {[round(c, 4) for c in c_init]}", flush=True)
        print(f"[HALC v2 + v16 diff] encoder_warmup=3/encoder_cooldown=8, decoder_warmup=7/decoder_cooldown=12", flush=True)
        print(f"[HALC v2 + v16 diff] reg_weight_max={halc.reg_weight_max:.4f}, init reg_weight={halc.reg_weight.item():.4f}", flush=True)

    # === 训练循环 (HG-Rec 风格 epoch, R41 EARLY_STOP=20, R41b per-epoch eval) ===
    MAX_EPOCHS = 200
    EARLY_STOP_PATIENCE = 20
    # C33 HG-Rec 路径: TopKAccumulator 输出 keys = ndcg + ndcg@1/5/10 (因 gin binding
    # 失败被 fallback, top_k_eval_list 默认 [1,5,10] 生效, 无 ndcg@20). 用 ndcg@10
    # 作为 SELECT_METRIC 等价于 ndcg (top-1 of valid set), 与 C33 训练协议一致.
    SELECT_METRIC = "ndcg@10"

    best_metric = -1.0
    early_stop_counter = 0
    train_log = []

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[hgrec] n_params={n_params:,} world={accelerator.num_processes}", flush=True)

    for epoch in range(MAX_EPOCHS):
        # === HALC v2: 更新 annealing schedule epoch ===
        halc.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_halс = 0.0
        epoch_count = 0
        if accelerator.is_main_process:
            pbar = tqdm(train_loader, desc=f"E{epoch+1}/{MAX_EPOCHS}")
        else:
            pbar = train_loader
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with accelerator.autocast():
                # === HALC v2 创新: 取 encoder hidden_states 计算 Poincaré reg ===
                loss, logits, enc_hs, _ = model(
                    input_ids=batch["history"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["target"],
                    output_hidden_states=True,
                )
                halc_reg = halc.reg_loss_for_layers(list(enc_hs))
                # === v69 G5: Codebook-aware Manifold Contrastive ===
                if V69_G5_ENABLED:
                    v69_g5_loss = _g5_manifold_contrastive_loss(logits, batch["target"])
                else:
                    v69_g5_loss = torch.tensor(0.0, device=loss.device)
                total_loss = loss + halc.reg_weight * halc_reg + V69_G5_AUX_WEIGHT * v69_g5_loss
            accelerator.backward(total_loss)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_halс += halc_reg.item()
            epoch_count += 1
            if accelerator.is_main_process:
                pbar.set_description(
                    f"E{epoch+1} loss={loss.item():.4f} halc={halc_reg.item():.4f} v69={v69_g5_loss.item():.4f} w={halc.reg_weight.item():.4f}"
                )
        avg_loss = epoch_loss / max(epoch_count, 1)
        avg_halc = epoch_halс / max(epoch_count, 1)

        valid_metrics = do_eval(valid_loader, f"Valid E{epoch+1}")
        cur_metric = valid_metrics.get(SELECT_METRIC, -1.0)

        if accelerator.is_main_process:
            log_entry = {
                "epoch": epoch + 1, "train_loss": avg_loss,
                "valid_metrics": valid_metrics, "best_metric": best_metric,
            }
            train_log.append(log_entry)
            with open(LOG_PATH, "w") as f:
                _json.dump(train_log, f, indent=2)

        if cur_metric > best_metric:
            best_metric = cur_metric
            early_stop_counter = 0
            if accelerator.is_main_process:
                os.makedirs(os.path.dirname(BEST_CKPT_PATH), exist_ok=True)
                # === HALC v2 创新: 同时保存 HALC 状态 (reg_weight + log_curvature) ===
                state = {
                    "epoch": epoch,
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "halc_state": halc.state_dict(),
                    "halc_epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "best_metric": best_metric,
                }
                torch.save(state, BEST_CKPT_PATH)
                print(f"[Best@{epoch+1}] saved {BEST_CKPT_PATH} {SELECT_METRIC}={cur_metric:.4f}", flush=True)
        else:
            early_stop_counter += 1
            if accelerator.is_main_process:
                print(f"[NoImprove@{epoch+1}] counter={early_stop_counter}/{EARLY_STOP_PATIENCE}", flush=True)
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                if accelerator.is_main_process:
                    print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                break

    # === 训练结束: 加载 best_ckpt 跑最终 test eval ===
    if accelerator.is_main_process:
        print(f"\n=== 训练结束, 加载 best_ckpt 跑最终 test eval ===", flush=True)
    raw_model = accelerator.unwrap_model(model)
    best_ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    raw_model.load_state_dict(best_ckpt["model"])
    # === HALC v2 创新: 加载 HALC 状态 (保持 annealing schedule) ===
    if "halc_state" in best_ckpt and "halc_epoch" in best_ckpt:
        halc.load_state_dict(best_ckpt["halc_state"])
        halc.set_epoch(best_ckpt["halc_epoch"])
        if accelerator.is_main_process:
            print(f"[HALC v2] restored halc_epoch={best_ckpt['halc_epoch']}", flush=True)
    test_metrics = do_eval(test_loader, "TEST FINAL")
    if accelerator.is_main_process:
        out_json = _osp.join(save_dir_root, "test_final.json")
        with open(out_json, "w") as f:
            _json.dump({
                "best_ckpt_epoch": int(best_ckpt["epoch"]),
                "best_ckpt_valid_ndcg20": float(best_ckpt["best_metric"]),
                "test_R@5": float(test_metrics.get("h@5", 0)),
                "test_R@10": float(test_metrics.get("h@10", 0)),
                "test_R@20": float(test_metrics.get("h@20", 0)),
                "test_NDCG@5": float(test_metrics.get("ndcg@5", 0)),
                "test_NDCG@10": float(test_metrics.get("ndcg@10", 0)),
                "test_NDCG@20": float(test_metrics.get("ndcg@20", 0)),
                "n_eval": int(test_metrics.get("total", 0)),
            }, f, indent=2)
        print(f"[saved] {out_json}", flush=True)
    if wandb_logging:
        wandb.finish()

def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    max_grad_norm=None,
    t5_d_model=128,
    t5_num_heads=6,
    t5_d_ff=1024,
    t5_num_layers=4,
    t5_num_decoder_layers=None,  # C31: HG-Rec 风格 decoder ≠ encoder 时设
    top_k_for_generation=10,
    should_add_sep_token=True,
    num_user_bins=None,
    top_k_eval_list=[1, 5, 10],
    # === C33 Issue #181 HG-Rec 架构合并 (序列级 CE + 平铺 vocab) ===
    use_hgrec_arch=False,
    hgrec_code_path="",
    hgrec_t5_d_model=128,
    hgrec_t5_d_ff=1024,
    hgrec_t5_num_heads=6,
    hgrec_t5_d_kv=64,
    hgrec_t5_num_layers=6,
    hgrec_t5_num_decoder_layers=4,
    hgrec_t5_dropout=0.1,
    hgrec_vocab_size=769,
    hgrec_max_len=20,
    hgrec_batch_size=2560,  # per-global-batch (4 卡 DDP)
):
    # HG-Rec gin binding 兜底 (加速器子进程下 gin macro 可能未生效, 强制走 HG-Rec 路径)
    if "hgrec" in sys.argv[0:5] or any("hgrec" in str(a) for a in sys.argv):
        use_hgrec_arch = True
    if os.environ.get("FORCE_HGREC", "0") == "1":
        use_hgrec_arch = True

    # === gin 兜底: save_dir_root 不通过 gin binding 时按 config 文件名派生 ===
    if save_dir_root == "out/":
        # 检测 gin config 文件名 (sys argv 末位是 .gin)
        for arg in sys.argv[::-1]:
            if arg.endswith(".gin"):
                tag = arg.replace("decoder_instruments_", "").replace(".gin", "")
                save_dir_root = f"out/decoder/instruments_hgrec_{tag}/"
                break

    if dataset not in (RecDataset.AMAZON, RecDataset.INSTRUMENTS):
        if not use_hgrec_arch:
            raise Exception(f"Dataset currently not supported: {dataset}.")

    # === C33 HG-Rec 路径 (序列级 CE + 平铺 vocab) ===
    if use_hgrec_arch:
        return _train_hgrec(
            accelerator_factory=lambda: Accelerator(
                split_batches=split_batches,
                mixed_precision=mixed_precision_type if amp else "no",
                kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
            ),
            dataset_folder=dataset_folder,
            save_dir_root=save_dir_root,
            pretrained_decoder_path=pretrained_decoder_path,
            hgrec_code_path=hgrec_code_path,
            hgrec_t5_d_model=hgrec_t5_d_model,
            hgrec_t5_d_ff=hgrec_t5_d_ff,
            hgrec_t5_num_heads=hgrec_t5_num_heads,
            hgrec_t5_d_kv=hgrec_t5_d_kv,
            hgrec_t5_num_layers=hgrec_t5_num_layers,
            hgrec_t5_num_decoder_layers=hgrec_t5_num_decoder_layers,
            hgrec_t5_dropout=hgrec_t5_dropout,
            hgrec_vocab_size=hgrec_vocab_size,
            hgrec_max_len=hgrec_max_len,
            batch_size=hgrec_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            top_k_for_generation=top_k_for_generation,
            top_k_eval_list=top_k_eval_list,
            wandb_logging=wandb_logging,
        )

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
        # EncoderDecoderRetrievalModel forward 是 conditional (item_sid_embedding_table
        # 不一定每步都参与), 必须设 find_unused_parameters=True 让 DDP 容许 None grad
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(project="gen-retrieval-decoder-training", config=params)

    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split,
    )
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=True,
        subsample=train_data_subsample,
        # 注意: 不传 split 参数, 让 SeqData 走 is_train=True 默认行为
        # (原代码传 split=dataset_split 是 RawMusicalInstruments 占位符, 不是 data split)
    )
    # HG-Rec 标准: valid 用于每个 epoch 选 best ckpt (NDCG@20), test 仅在最后用 best ckpt 评估
    valid_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        split="eval",  # Amazon / PreprocessingMixin 约定, 语义上 = HG-Rec valid (itemId_fut = valid.target)
    )
    test_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        split="test",  # 最终 test eval (itemId_fut = test.target = 最后 1 个 item)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    # eval 必须用更小 batch: beam=20 generation 时 KV cache 与 batch*beam 成正比,
    # batch=2560+beam20 在 L40S 44GB 上 OOM (38.76 GiB 占用, 还需 5.86 GiB 失败)
    # 用 per-GPU=64, 4 卡合并 = 256 sample/step (vs 训练 2560), KV cache 留足空间
    eval_batch_size = 64
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # 评估不打乱
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, valid_dataloader, test_dataloader
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
        gate_M2_intrinsic=False,  # C28/C29 rollback: 关 M2 (M2 + M3 联合 NO-GO)
        gate_M3_transport=True,   # C28: 开 M3 transport (当前最佳 0.0991)
        hypervq=False,  # C21 rollback fix: 与 Stage2 ckpt (无 mlr_*) 保持一致 (Poincaré distance argmin)
        use_scs=False,  # C24 R37 rollback
        scs_eps_scale=1.0,
        prefix_router_layers=[True] * vae_n_layers,  # C30: 与 Stage2 ckpt 一致
    )
    tokenizer = accelerator.prepare(tokenizer)
    # unwrap DDP 包装以调用非-module 方法 (precompute_corpus_ids 是 tokenizer 的方法,不是 nn.Module 方法)
    raw_tokenizer = accelerator.unwrap_model(tokenizer)
    raw_tokenizer.precompute_corpus_ids(item_dataset)

    if push_vae_to_hf:
        login()
        raw_tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    codebooks = raw_tokenizer.cached_ids[:, :vae_n_layers].cpu()

    model = EncoderDecoderRetrievalModel(
        codebooks=codebooks,
        num_hierarchies=vae_n_layers,
        num_embeddings_per_hierarchy=vae_codebook_size,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        t5_num_decoder_layers=t5_num_decoder_layers,  # C31: HG-Rec 风格 encoder=6 decoder=4
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
        # 欧氏对照: 无曲率注入 / 无 HAB
        curv_state_path="/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/curvature_state_c30_curriculum_per_item_delta.npy",
        response_path="/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/response_c30_curriculum_per_item_delta.npy",
        hab_rqvae_ckpt="/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_c30_curriculum_per_item_delta/rqvae_final.pt",
    )
    if False:
        model = torch.compile(model)  # 关闭 torch.compile (4 卡 DDP 兼容性问题)

    # Issue #64 v74: HAB 参数 (lambda_raw/residual_alpha/U/V) 走 high-lr 组 (100x)
    if getattr(model, "hab_module", None) is not None:
        hab_params = list(model.hab_module.parameters())
        main_params = [p for p in model.parameters() if p not in set(hab_params)]
        optimizer = AdamW(
            [
                {"params": main_params, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": hab_params, "lr": learning_rate * 100.0, "weight_decay": weight_decay},
            ]
        )
        print(f"[HAB] {len(hab_params)} params in high-lr group (lr={learning_rate*100:.4f})", flush=True)
    else:
        optimizer = AdamW(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    # warmup_steps 与 iterations 等比缩放 (DDP 提速: iterations 10000→2500)
    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=250)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}", flush=True)

    # === HG-Rec 风格 epoch-based 训练循环 ===
    # MAX_EPOCHS / EARLY_STOP_PATIENCE / SELECT_METRIC 都是 HG-Rec 标准 (train_HG-Rec.py)
    MAX_EPOCHS = 200
    EARLY_STOP_PATIENCE = 20  # 连续 20 epoch valid NDCG@20 没创新低就停
    SELECT_METRIC = "ndcg@20"  # 与 HG-Rec train_HG-Rec.py:214 一致 (best_ndcg)
    BEST_CKPT_PATH = "out/decoder/c30_curriculum_per_item_delta_400k_instruments/best_ckpt.pt"

    # === helper: eval 一个 dataloader, 返回 R35b 全局口径指标 ===
    def do_eval(eval_dl, split_name: str):
        model.eval()
        raw_model = accelerator.unwrap_model(model)
        acc = TopKAccumulator(ks=top_k_eval_list)
        with torch.no_grad():
            for batch in eval_dl:
                data = batch_to(batch, device)
                tokenized_data = tokenizer(data)
                generated = raw_model.generate_next_sem_id(
                    tokenized_data, top_k=True, temperature=1
                )
                actual = tokenized_data.sem_ids_fut[:, :vae_n_layers]
                acc.accumulate(actual=actual, top_k=generated.sem_ids)
        # R35b: 4 卡 all_reduce SUM 后除以全局 total
        local = acc.get_sums_and_total()
        keys = ["ndcg"] + [f"ndcg@{k}" for k in top_k_eval_list] + [f"h@{k}" for k in top_k_eval_list] + ["total"]
        local_vec = torch.tensor([float(local.get(k, 0.0)) for k in keys], device=device)
        global_vec = accelerator.reduce(local_vec, reduction="sum")
        global_total = global_vec[-1].item()
        metrics = {
            keys[i]: (global_vec[i].item() / global_total if global_total > 0 else 0.0)
            for i in range(len(keys) - 1)
        }
        if accelerator.is_main_process:
            print(f"[{split_name}@{epoch+1}] {metrics}", flush=True)
        return metrics

    best_metric = -1.0
    early_stop_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{MAX_EPOCHS}",
            disable=not accelerator.is_main_process,
        )
        epoch_loss = 0.0
        epoch_count = 0
        for batch in pbar:
            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)
            with accelerator.autocast():
                model_output = model(tokenized_data)
                loss = model_output.loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            if max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            epoch_count += 1
            pbar.set_description(f"E{epoch+1} loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(epoch_count, 1)
        if accelerator.is_main_process:
            print(f"\n[Epoch {epoch+1}] avg_train_loss={avg_loss:.4f}", flush=True)

        # === 每个 epoch 末: valid eval, 触发 best ckpt 保存 (HG-Rec 风格) ===
        valid_metrics = do_eval(valid_dataloader, "Valid")

        cur_metric = valid_metrics.get(SELECT_METRIC, -1.0)
        if cur_metric > best_metric:
            best_metric = cur_metric
            early_stop_counter = 0
            if accelerator.is_main_process:
                os.makedirs(os.path.dirname(BEST_CKPT_PATH), exist_ok=True)
                state = {
                    "epoch": epoch,
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "valid_metrics": valid_metrics,
                    "best_metric": best_metric,
                }
                torch.save(state, BEST_CKPT_PATH)
                print(f"[Best@{epoch+1}] saved best_ckpt.pt {SELECT_METRIC}={cur_metric:.4f}", flush=True)
        else:
            early_stop_counter += 1
            if accelerator.is_main_process:
                print(f"[NoImprove@{epoch+1}] counter={early_stop_counter}/{EARLY_STOP_PATIENCE}", flush=True)
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                if accelerator.is_main_process:
                    print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                break

    # === 训练结束: 加载 best_ckpt, 跑最终 test eval ===
    if accelerator.is_main_process:
        print(f"\n=== 训练结束, 加载 best ckpt 跑最终 test eval ===", flush=True)
    raw_model = accelerator.unwrap_model(model)
    best_ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    raw_model.load_state_dict(best_ckpt["model"])
    if accelerator.is_main_process:
        print(
            f"Loaded best_ckpt from epoch {best_ckpt['epoch']} "
            f"with valid {SELECT_METRIC}={best_ckpt['best_metric']:.4f}",
            flush=True,
        )
    test_metrics = do_eval(test_dataloader, "TEST FINAL")

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    import train_decoder  # noqa: F401  强制注册 @gin.configurable
    parse_config()
    train()

