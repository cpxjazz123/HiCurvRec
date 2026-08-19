"""DDP 4 卡 test eval (加载已训好的 best_ckpt 跑 test.parquet).

修复 #14 race condition: rank 0 先 torch.load best_ckpt, 然后用
torch.distributed.broadcast 把 state_dict 传给其他 3 个 rank — 避免 4 个 rank
同时 load 同一文件的 race condition / 文件系统缓存问题.

R35b: 4 卡分片不重复评估 test 集, all_reduce SUM, 统一除以全局 total.
"""
import os
import sys
import json
import gin
import torch

# 关键: import train_decoder 让 `train` configurable 被 gin 注册
# (decoder_instruments.gin 里有 train.xxx = ... 配置项)
import train_decoder  # noqa: F401

from accelerate import Accelerator
from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.hg_rec import HG_Rec  # C33 Issue #181 合并
from modules.tokenizer.semids import SemanticIdTokenizer
from data.instruments import RawMusicalInstrumentsHGRec, hgrec_collate_fn
# HALC v4/v6 wrap support (R36 曲率机制, 让 test_eval 与 train 用相同 wrap FFN)
try:
    from _lib.halc_v4 import wrap_hgrec_ffn as _wrap_hgrec_ffn_v4
    _HAS_HALC_V4 = True
except ImportError:
    _HAS_HALC_V4 = False
from torch.utils.data import DataLoader


def main():
    # 与 train_decoder.py 同样的 gin config
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    accelerator = Accelerator(split_batches=True, mixed_precision="no")
    device = accelerator.device

    # gin config 必须显式 parse (复用 train_decoder 的 config)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/decoder_instruments.gin"
    try:
        gin.parse_config_file(config_path, skip_unknown=True)
    except Exception as e:
        print(f"[test_eval] gin parse failed: {e}; fallback to FORCE_HGREC=1", flush=True)

    BEST_CKPT_PATH = sys.argv[2] if len(sys.argv) > 2 else "out/decoder/instruments/best_ckpt.pt"

    # === C33 HG-Rec 路径分支 ===
    try:
        use_hgrec = gin.query_parameter("train_decoder.train.use_hgrec_arch")
    except ValueError:
        use_hgrec = False
    if os.environ.get("FORCE_HGREC", "0") == "1":
        return _test_eval_hgrec(accelerator, device, BEST_CKPT_PATH)
    if use_hgrec:
        return _test_eval_hgrec(accelerator, device, BEST_CKPT_PATH)

    # 1) load datasets & tokenizer (走 train 同样的路径)
    item_dataset = ItemData(
        root="dataset/", dataset=RecDataset.INSTRUMENTS, split="instruments",
    )
    test_dataset = SeqData(
        root="dataset/", dataset=RecDataset.INSTRUMENTS,
        is_train=False, subsample=False, split="test",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4,
    )

    tokenizer = SemanticIdTokenizer(
        input_dim=768, hidden_dims=[512, 256, 128], output_dim=32,
        codebook_size=256, n_layers=3, n_cat_feats=0,
        rqvae_weights_path="/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_m2m3/rqvae_final.pt",
        gate_M2_intrinsic=True,
        gate_M3_transport=True,
        rqvae_codebook_normalize=False, rqvae_sim_vq=False,
        hypervq=False,  # C21 rollback fix: 与 Stage2 ckpt 一致
    )
    tokenizer = accelerator.prepare(tokenizer)
    raw_tokenizer = accelerator.unwrap_model(tokenizer)
    raw_tokenizer.precompute_corpus_ids(item_dataset)

    codebooks = raw_tokenizer.cached_ids[:, :3].cpu()
    model = EncoderDecoderRetrievalModel(
        codebooks=codebooks, num_hierarchies=3, num_embeddings_per_hierarchy=256,
        t5_d_model=384, t5_num_heads=6, t5_d_ff=1024, t5_num_layers=4,
        top_k_for_generation=20, should_add_sep_token=True, num_user_bins=None,
    )

    # === 修复 #14 race: rank 0 先 load, 然后用 broadcast_object_list 把 state_dict 传给其他 rank ===
    if accelerator.is_main_process:
        print(f"[rank 0] loading best_ckpt: {BEST_CKPT_PATH}", flush=True)
        state = torch.load(BEST_CKPT_PATH, map_location="cpu", weights_only=False)
        valid_metrics = state.get("valid_metrics", {})
        best_metric = state.get("best_metric", -1.0)
        epoch = state.get("epoch", -1)
        broadcast_payload = [state["model"]]
        print(f"[rank 0] best_ckpt epoch={epoch} valid_ndcg@20={best_metric:.4f}", flush=True)
    else:
        broadcast_payload = [None]
        valid_metrics = {}
        best_metric = -1.0
        epoch = -1

    # 用 accelerate 的广播机制把 state_dict 同步给所有 rank
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(broadcast_payload, src=0)
    model_state = broadcast_payload[0]
    model.load_state_dict(model_state, strict=False)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    raw_model = accelerator.unwrap_model(model)

    # 2) eval (R35b)
    model.eval()
    acc = TopKAccumulator(ks=[1, 5, 10, 20])
    with torch.no_grad():
        for batch in test_dataloader:
            data = batch_to(batch, device)
            tokenized_data = tokenizer(data)
            generated = raw_model.generate_next_sem_id(
                tokenized_data, top_k=True, temperature=1
            )
            actual = tokenized_data.sem_ids_fut[:, :3]
            acc.accumulate(actual=actual, top_k=generated.sem_ids)

    local = acc.get_sums_and_total()
    keys = ["ndcg"] + [f"ndcg@{k}" for k in [1, 5, 10, 20]] + \
           [f"h@{k}" for k in [1, 5, 10, 20]] + ["total"]
    local_vec = torch.tensor([float(local.get(k, 0.0)) for k in keys], device=device)
    global_vec = accelerator.reduce(local_vec, reduction="sum")
    global_total = global_vec[-1].item()
    metrics = {
        keys[i]: (global_vec[i].item() / global_total if global_total > 0 else 0.0)
        for i in range(len(keys) - 1)
    }
    if accelerator.is_main_process:
        print(f"\n=== TEST FINAL (beam=20) ===", flush=True)
        print(f"  loaded best_ckpt epoch={epoch} valid_ndcg@20={best_metric:.4f}", flush=True)
        print(f"  Recall@5:  {metrics.get('h@5', 0):.4f}", flush=True)
        print(f"  Recall@10: {metrics.get('h@10', 0):.4f}", flush=True)
        print(f"  Recall@20: {metrics.get('h@20', 0):.4f}", flush=True)
        print(f"  NDCG@5:    {metrics.get('ndcg@5', 0):.4f}", flush=True)
        print(f"  NDCG@10:   {metrics.get('ndcg@10', 0):.4f}", flush=True)
        print(f"  NDCG@20:   {metrics.get('ndcg@20', 0):.4f}", flush=True)
        print(f"  n_eval:    {int(global_total)}", flush=True)
        print(f"  full: {metrics}", flush=True)
        # 持久化结果
        import json
        out_path = "out/decoder/instruments/test_final.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "best_ckpt_epoch": int(epoch),
                "best_ckpt_valid_ndcg20": float(best_metric),
                "test_R@5": float(metrics.get("h@5", 0)),
                "test_R@10": float(metrics.get("h@10", 0)),
                "test_R@20": float(metrics.get("h@20", 0)),
                "test_NDCG@5": float(metrics.get("ndcg@5", 0)),
                "test_NDCG@10": float(metrics.get("ndcg@10", 0)),
                "test_NDCG@20": float(metrics.get("ndcg@20", 0)),
                "n_eval": int(global_total),
            }, f, indent=2)
        print(f"\nSaved → {out_path}", flush=True)


# =============================================================================
# C33 Issue #181 HG-Rec test eval 路径 (序列级 CE + T5ForConditionalGeneration)
# 加载 best_ckpt.pt (HG-Rec 训练产出), 4 卡 DDP 评估 test.parquet
# =============================================================================

def _test_eval_hgrec(accelerator, device, BEST_CKPT_PATH):
    """HG-Rec 路径 test eval — 与 train_decoder.py:_train_hgrec 的 do_eval 完全一致."""
    import os.path as _osp

    INSTRUMENTS_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
    # 硬编码 (与 train_decoder.hgrec_code_path 一致), 不依赖 gin query
    code_path = "/home/wlia0047/ar57/wenyu/GeneRec/curvature_experiment/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"
    beam_size = 20
    top_k_eval_list = [5, 10, 20]
    max_len = 20

    # === HG-Rec 模型 (与 train 一致) ===
    hgrec_config = {
        "num_layers": 6,
        "num_decoder_layers": 4,
        "d_model": 128,
        "d_ff": 1024,
        "num_heads": 6,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "vocab_size": 769,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "decoder_start_token_id": 0,
        "feed_forward_proj": "relu",
    }
    model = HG_Rec(hgrec_config)

    # === v40 (Issue260): Stage 4 也启用 PoincareT5LayerNorm (与 Stage 3 训练时一致) ===
    # 否则 ckpt state_dict 加载后 forward 行为不一致, R51 baseline 不可比
    try:
        import gin as _gin
        _use_hyp_ln = _gin.query_parameter("train_decoder.train.use_hyp_layernorm")
        if _use_hyp_ln:
            from _lib.hyperbolic_layernorm import replace_t5_layernorm
            n_replaced = replace_t5_layernorm(model.model, c=0.5)
            print(f"[v40 hyp_layernorm] Stage 4 replaced {n_replaced} T5LayerNorm → PoincareT5LayerNorm (c=0.5)", flush=True)
    except (ValueError, Exception) as e:
        # gin.query_parameter raises ValueError if not bound
        pass

    # === v41 (Issue261): Stage 4 也启用 PoincareInputEmbedding (与 Stage 3 训练时一致) ===
    try:
        import gin as _gin
        _use_poinc = _gin.query_parameter("train_decoder.train.use_poinc_input_embed")
        if _use_poinc:
            from _lib.poincare_input_embed import replace_t5_shared_embedding
            n_replaced = replace_t5_shared_embedding(model.model, c=0.5)
            print(f"[v41 poinc_input_embed] Stage 4 replaced {n_replaced} T5.shared → PoincareInputEmbedding (c=0.5)", flush=True)
    except (ValueError, Exception) as e:
        pass

    # === v42 (Issue262): Stage 4 也启用 Q/K 双曲距离 bias (与 Stage 3 训练时一致) ===
    try:
        import gin as _gin
        _use_attn = _gin.query_parameter("train_decoder.train.use_hyp_attn_bias")
        if _use_attn:
            from _lib.hyp_attn_bias_v42 import install_v42_hook, wrap_t5_attention_with_poinc_bias
            install_v42_hook(model.model)
            n_replaced = wrap_t5_attention_with_poinc_bias(model.model, c=0.5, lam=0.1)
            print(f"[v42 hyp_attn_bias] Stage 4 installed Q/K Poincaré distance bias on {n_replaced} T5 attention (c=0.5, λ=0.1)", flush=True)
    except (ValueError, Exception) as e:
        pass

    # === v43 (Issue263): Stage 4 也启用 Lorentz cross-attention bias (与 Stage 3 训练时一致) ===
    try:
        import gin as _gin
        _use_lorentz = _gin.query_parameter("train_decoder.train.use_lorentz_attn")
        if _use_lorentz:
            from _lib.lorentz_attn_bias_v43 import install_v43_hook, wrap_t5_cross_attention_with_lorentz_bias
            install_v43_hook(model.model)
            n_replaced = wrap_t5_cross_attention_with_lorentz_bias(model.model, c=1.0, lam=0.1)
            print(f"[v43 lorentz_attn] Stage 4 installed Lorentz inner product bias on {n_replaced} T5 cross-attention (c=1.0, λ=0.1)", flush=True)
    except (ValueError, Exception) as e:
        pass

    # === v44 (Issue264): Stage 4 也启用 product manifold bias (与 Stage 3 训练时一致) ===
    try:
        import gin as _gin
        _use_pm = _gin.query_parameter("train_decoder.train.use_product_manifold")
        if _use_pm:
            from _lib.product_manifold_attn_v44 import install_v44_hook, wrap_t5_cross_attention_with_product_bias
            install_v44_hook(model.model)
            n_replaced = wrap_t5_cross_attention_with_product_bias(model.model, c_p=0.5, lam_p=0.05, c_l=1.0, lam_l=0.05)
            print(f"[v44 product_manifold] Stage 4 installed Poincaré+Lorentz product bias on {n_replaced} T5 cross-attention (c_p=0.5/λ_p=0.05, c_l=1.0/λ_l=0.05)", flush=True)
    except (ValueError, Exception) as e:
        pass

    test_ds = RawMusicalInstrumentsHGRec(
        parquet_path=_osp.join(INSTRUMENTS_DIR, "test.parquet"),
        code_path=code_path, mode="evaluation",
        codebook_size=[256, 256, 256],
        max_len=max_len,
    )

    # === R34c fix: 固定 DistributedSampler seed=42 让 4 卡 test 分片可重复 ===
    # 默认 DistributedSampler 用 torch.randint 每次生成新 seed, 4 卡分片每次不同
    # → all_reduce SUM 后总数有 ±0.001 量级噪声
    # train_decoder.py 路径让 accelerator.prepare() 自动 wrap DistributedSampler, seed 由 torch RNG 决定
    # → 显式设 torch.manual_seed(42) + numpy/random seed 让 DistributedSampler 拿到固定 seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    import random as _random
    _random.seed(42)
    import numpy as _np
    _np.random.seed(42)

    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4,
        collate_fn=hgrec_collate_fn,
    )

    # === 加载 best_ckpt (rank 0 先 load, broadcast 给其他 rank) ===
    if accelerator.is_main_process:
        print(f"[rank 0] loading best_ckpt: {BEST_CKPT_PATH}", flush=True)
        state = torch.load(BEST_CKPT_PATH, map_location="cpu", weights_only=False)
        valid_metrics = state.get("valid_metrics", {})
        best_metric = state.get("best_metric", -1.0)
        epoch = state.get("epoch", -1)
        broadcast_payload = [state["model"]]
        print(f"[rank 0] best_ckpt epoch={epoch} valid_ndcg@20={best_metric:.4f}", flush=True)
    else:
        broadcast_payload = [None]
        valid_metrics = {}
        best_metric = -1.0
        epoch = -1

    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(broadcast_payload, src=0)
    model_state = broadcast_payload[0]
    model.load_state_dict(model_state, strict=False)
    model, test_loader = accelerator.prepare(model, test_loader)
    raw_model = accelerator.unwrap_model(model)

    # === eval (R35b: DDP 同口径) ===
    model.eval()
    acc = TopKAccumulator(ks=top_k_eval_list)
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["history"]
            attention_mask = batch["attention_mask"]
            labels = batch["target"]
            preds = raw_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                num_beams=beam_size,
            )
            preds = preds[:, 1:].reshape(input_ids.shape[0], beam_size, -1)
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
        print(f"\n=== TEST FINAL (HG-Rec, beam={beam_size}) ===", flush=True)
        print(f"  loaded best_ckpt epoch={epoch} valid_ndcg@20={best_metric:.4f}", flush=True)
        for k in top_k_eval_list:
            print(f"  Recall@{k}:  {metrics.get(f'h@{k}', 0):.4f}", flush=True)
            print(f"  NDCG@{k}:    {metrics.get(f'ndcg@{k}', 0):.4f}", flush=True)
        print(f"  n_eval:    {int(global_total)}", flush=True)
        out_path = _osp.join(_osp.dirname(BEST_CKPT_PATH), "test_final.json")
        os.makedirs(_osp.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "best_ckpt_epoch": int(epoch),
                "best_ckpt_valid_ndcg20": float(best_metric),
                "test_R@5": float(metrics.get("h@5", 0)),
                "test_R@10": float(metrics.get("h@10", 0)),
                "test_R@20": float(metrics.get("h@20", 0)),
                "test_NDCG@5": float(metrics.get("ndcg@5", 0)),
                "test_NDCG@10": float(metrics.get("ndcg@10", 0)),
                "test_NDCG@20": float(metrics.get("ndcg@20", 0)),
                "n_eval": int(global_total),
            }, f, indent=2)
        print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
