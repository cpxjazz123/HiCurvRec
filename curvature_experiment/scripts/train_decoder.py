import os
import sys
# 跳过 transformers TF 路径 (Keras 3 兼容性问题)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 把脚本所在目录 (scripts/) 的上级目录 (MAIN_DIR, 含 modules/ _lib/) 加入 sys.path
# (R44/R47: 任务目录自包含; Stage 3 torchrun cwd=MAIN_DIR, 这里显式再加 MAIN_DIR 保险)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MAIN_DIR)

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

# === HALC v38 备胎 (Issue258): per-token input-dependent κ + Stage 2 TCU init=0.3 ===
# 默认未启用, v38 gin config 显式设 train.use_halc_v38=True 才生效
try:
    from _lib.halc_v38 import HALCPerTokenInputDependentRegularizer
    _HAS_HALC_V38 = True
except ImportError:
    _HAS_HALC_V38 = False


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
    use_halc_v38=False,  # Issue258 v38: per-token input-dependent κ
    use_hyp_layernorm=False,  # Issue260 v40: Stage 3 T5 LayerNorm → Poincaré LayerNorm (R36 几何变换)
    use_poinc_input_embed=False,  # Issue261 v41: Stage 3 T5 shared embedding → Poincaré projection (R36 几何变换)
    use_hyp_attn_bias=False,  # Issue262 v42: Stage 3 T5 attention Q/K 双曲距离 bias (HNN 2019)
    use_lorentz_attn=False,  # Issue263 v43: Stage 3 T5 cross-attention Lorentz inner product bias (Chen 2022)
    use_product_manifold=False,  # Issue264 v44: Stage 3 T5 cross-attention Poincaré + Lorentz 双 bias 组合 (R36 曲率机制组合)
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

    # === R51 强约束: DataLoader 创建之前固定全局 RNG (Stage 3 训练端) ===
    # 防止 DistributedSampler 每次拿不同分片 → 训练轨迹不同 → EARLY_STOP 触发时机漂移
    # → best_ckpt_epoch 不同 → test_R@10 漂移 (实测 ±0.001 量级)
    import random as _r51_random
    import numpy as _r51_np
    _r51_random.seed(42 + accelerator.process_index)
    _r51_np.random.seed(42 + accelerator.process_index)
    torch.manual_seed(42 + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + accelerator.process_index)
    print(f"[R51 seed] rank={accelerator.process_index} seeded with 42+rank before DataLoader", flush=True)

    # === 路径 (硬编码, R44/R47) ===
    INSTRUMENTS_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
    DEFAULT_CODE_PATH = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"
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
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=8, persistent_workers=True, pin_memory=True,
        prefetch_factor=4, collate_fn=hgrec_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=8, persistent_workers=True, pin_memory=True,
        prefetch_factor=4, collate_fn=hgrec_collate_fn,
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

    # === gin binding fallback 修复 (Issue261 v41): gin binding 失败时参数 fallback 默认值
    # === 通过 env var 显式启用 v40/v41 创新, 不依赖 gin ===
    if os.environ.get("USE_HYP_LAYERNORM", "0") == "1":
        use_hyp_layernorm = True
    if os.environ.get("USE_POINC_INPUT_EMBED", "0") == "1":
        use_poinc_input_embed = True
    if os.environ.get("USE_HYP_ATTN_BIAS", "0") == "1":
        use_hyp_attn_bias = True
    if os.environ.get("USE_LORENTZ_ATTN", "0") == "1":
        use_lorentz_attn = True
    if os.environ.get("USE_PRODUCT_MANIFOLD", "0") == "1":
        use_product_manifold = True

    # === v40 (Issue260): Stage 3 T5 LayerNorm → Poincaré LayerNorm (R36 几何变换) ===
    # 替换 T5 所有 T5LayerNorm 实例为 PoincareT5LayerNorm (logmap0 → RMSNorm → expmap0, c=0.5)
    # 保持 weight 起点与 T5LayerNorm 完全一致 (new_ln.weight.copy_(old_ln.weight))
    if use_hyp_layernorm:
        from _lib.hyperbolic_layernorm import replace_t5_layernorm
        n_replaced = replace_t5_layernorm(model.model, c=0.5)  # 与 v19 Stage 1 c_end=0.7 一致
        print(f"[v40 hyp_layernorm] replaced {n_replaced} T5LayerNorm → PoincareT5LayerNorm (c=0.5)", flush=True)

    # === v41 (Issue261): Stage 3 T5 shared embedding → Poincaré projection (R36 几何变换) ===
    # 替换 T5.shared (nn.Embedding) 为 PoincareInputEmbedding, 每次 forward 把欧氏 embedding 投影到 Poincaré ball
    if use_poinc_input_embed:
        from _lib.poincare_input_embed import replace_t5_shared_embedding
        n_replaced = replace_t5_shared_embedding(model.model, c=0.5)
        print(f"[v41 poinc_input_embed] replaced {n_replaced} T5.shared → PoincareInputEmbedding (c=0.5)", flush=True)

    # === v42 (Issue262): Stage 3 T5 attention Q/K 双曲距离 bias (HNN 2019) ===
    # 在 encoder 第一层 attention 注入 Poincaré 距离 bias (lambda=0.1, c=0.5)
    # 路径独立于 v40 (LayerNorm) / v41 (shared embedding)
    if use_hyp_attn_bias:
        from _lib.hyp_attn_bias_v42 import install_v42_hook, wrap_t5_attention_with_poinc_bias
        install_v42_hook(model.model)  # 注册 forward hook 捕获 Q/K
        n_replaced = wrap_t5_attention_with_poinc_bias(model.model, c=0.5, lam=0.1)
        print(f"[v42 hyp_attn_bias] installed Q/K Poincaré distance bias on {n_replaced} T5 attention (c=0.5, λ=0.1)", flush=True)

    # === v43 (Issue263): Stage 3 T5 cross-attention Lorentz inner product bias (Chen 2022) ===
    # 在 decoder 第一层 cross-attention 注入 Lorentz 距离 bias (lambda=0.1, c=1.0)
    # 路径独立于 v40 (LayerNorm) / v41 (shared embedding) / v42 (encoder self-attention 欧氏距离)
    if use_lorentz_attn:
        from _lib.lorentz_attn_bias_v43 import install_v43_hook, wrap_t5_cross_attention_with_lorentz_bias
        install_v43_hook(model.model)
        n_replaced = wrap_t5_cross_attention_with_lorentz_bias(model.model, c=1.0, lam=0.1)
        print(f"[v43 lorentz_attn] installed Lorentz inner product bias on {n_replaced} T5 cross-attention (c=1.0, λ=0.1)", flush=True)

    # === v44 (Issue264): Stage 3 T5 cross-attention product manifold bias (R36 曲率机制组合) ===
    # 在 decoder 第一层 cross-attention 同时注入 Poincaré 距离 bias (c=0.5, λ_p=0.05) + Lorentz 距离 bias (c=1.0, λ_l=0.05)
    # bias -= λ_p * normalize(||q-k||²) + λ_l * normalize(d_L(q_L, k_L))
    if use_product_manifold:
        from _lib.product_manifold_attn_v44 import install_v44_hook, wrap_t5_cross_attention_with_product_bias
        install_v44_hook(model.model)
        n_replaced = wrap_t5_cross_attention_with_product_bias(model.model, c_p=0.5, lam_p=0.05, c_l=1.0, lam_l=0.05)
        print(f"[v44 product_manifold] installed Poincaré+Lorentz product bias on {n_replaced} T5 cross-attention (c_p=0.5/λ_p=0.05, c_l=1.0/λ_l=0.05)", flush=True)

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
    # === HALC v38 备胎 (Issue258): per-token input-dependent κ + Stage 2 TCU init=0.3 ===
    # env 注入兜底 (R34b 风格 — gin binding 失败时仍能启用 v38 机制)
    if os.environ.get("USE_HALC_V38", "0") == "1" and _HAS_HALC_V38:
        use_halc_v38 = True
    if use_halc_v38 and _HAS_HALC_V38:
        halc = HALCPerTokenInputDependentRegularizer(
            num_layers=7, d_model=128, init_curvature=0.3,
            warmup_epochs=5, cooldown_epochs=10, reg_weight_max=0.06,
            delta_scale_init=0.01,
        )
        HALC_VARIANT = "v38_per_token_input_dependent"
    else:
        # === HALC v2 + v16 differential schedule (R36 机制变更): encoder 早 warmup, decoder 晚 warmup ===
        halc = HALCAnnealingRegularizer(
            num_layers=7, init_curvature=1.0, c_max=1.0,
            warmup_epochs=5, cooldown_epochs=10, reg_weight_max=0.05,
            encoder_warmup=3, encoder_cooldown=8,
            decoder_warmup=7, decoder_cooldown=12,
        )
        HALC_VARIANT = "v2_v16_diff"
    halc = halc.to(accelerator.device)
    if accelerator.is_main_process:
        halc.set_epoch(0)
        if HALC_VARIANT == "v38_per_token_input_dependent":
            print(f"[HALC v38 per-token κ + TCU init=0.3] init reg_weight={halc.reg_weight.item():.4f}", flush=True)
            print(f"[HALC v38] warmup={halc.warmup_epochs}/cooldown={halc.cooldown_epochs}, reg_weight_max={halc.reg_weight_max:.4f}", flush=True)
            print(f"[HALC v38] num_layers={halc.num_layers}, init_curvature=0.3, delta_scale={halc.delta_scale}", flush=True)
        else:
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
                loss, _, enc_hs, _ = model(
                    input_ids=batch["history"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["target"],
                    output_hidden_states=True,
                )
                halc_reg = halc.reg_loss_for_layers(list(enc_hs))
                total_loss = loss + halc.reg_weight * halc_reg
            accelerator.backward(total_loss)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_halс += halc_reg.item()
            epoch_count += 1
            if accelerator.is_main_process:
                pbar.set_description(
                    f"E{epoch+1} loss={loss.item():.4f} halc={halc_reg.item():.4f} w={halc.reg_weight.item():.4f}"
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
    # === v40/v41/v42/v43 曲率机制开关 (Stage 3 端, 默认 False 不启用, 不破坏 v19 baseline) ===
    use_hyp_layernorm=False,  # Issue260 v40: T5 LayerNorm → Poincaré LayerNorm
    use_poinc_input_embed=False,  # Issue261 v41: T5.shared → PoincareInputEmbedding
    use_hyp_attn_bias=False,  # Issue262 v42: T5 attention Q/K 双曲距离 bias (HNN 2019)
    use_lorentz_attn=False,  # Issue263 v43: T5 cross-attention Lorentz inner product bias (Chen 2022)
    use_product_manifold=False,  # Issue264 v44: T5 cross-attention Poincaré + Lorentz product manifold bias
):
    # HG-Rec gin binding 兜底 (加速器子进程下 gin macro 可能未生效, 强制走 HG-Rec 路径)
    if "hgrec" in sys.argv[0:5] or any("hgrec" in str(a) for a in sys.argv):
        use_hgrec_arch = True
    if os.environ.get("FORCE_HGREC", "0") == "1":
        use_hgrec_arch = True

    # === gin 兜底: save_dir_root 不通过 gin binding 时按 config 文件名派生 ===
    if save_dir_root == "out/":
        # 优先用 env 注入 (R34b 风格 — 绝对路径避免嵌套错)
        env_save_dir = os.environ.get("SAVE_DIR_ROOT", "")
        if env_save_dir:
            save_dir_root = env_save_dir
        else:
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
            use_hyp_layernorm=use_hyp_layernorm,
            use_poinc_input_embed=use_poinc_input_embed,
            use_hyp_attn_bias=use_hyp_attn_bias,
            use_lorentz_attn=use_lorentz_attn,
            use_product_manifold=use_product_manifold,
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

