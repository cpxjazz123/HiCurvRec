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
    """HG-Rec 路径 test eval — 与 train_decoder.py:_train_hgrec 的 do_eval 完全一致.

    v66: 在 _test_eval_hgrec 主循环内插入 mixed geometry rerank (修复 v64 BLEND_ALPHA=0 失败).
    - V66_RERANK_BLEND_ALPHA=0.5 (混合, vs v64 = 0 纯几何), hist_emb 用 mean pool over input_ids history SID (vs v64 仅 last3).
    - RQ-VAE ckpt (C28 rqvae_final.pt) 提供 3 层 codebook (256, 32) × 3 → 96-dim concat.
    - mixed_score = α * (-rank) + (1-α) * (-norm_dist), 降序重排 beam.
    """
    import os.path as _osp

    INSTRUMENTS_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments"
    # 硬编码 (与 train_decoder.hgrec_code_path 一致), 不依赖 gin query
    code_path = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"
    beam_size = 20
    top_k_eval_list = [5, 10, 20]
    max_len = 20

    # === v66 Stage 4 mixed geometry rerank 配置 (修复 v64 BLEND_ALPHA=0 -38% 失败) ===
    V66_RERANK_ENABLED = True
    V66_RERANK_C = 0.7
    V66_RERANK_RQVAE_CKPT = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_m2m3/rqvae_final.pt"
    V66_RERANK_BLEND_ALPHA = 0.5  # 0=纯几何, 1=纯原序; 0.5 混合保留 T5 logit 主导
    V66_RERANK_HIST_POOL = "mean"  # mean pool over full input_ids history (vs v64 仅 last 3 SID)
    # 3 层距离加权 (默认 1.0 等权)
    _V66_LAYER_W = (1.0, 1.0, 1.0)

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

    test_ds = RawMusicalInstrumentsHGRec(
        parquet_path=_osp.join(INSTRUMENTS_DIR, "test.parquet"),
        code_path=code_path, mode="evaluation",
        codebook_size=[256, 256, 256],
        max_len=max_len,
    )

    # === R51+ 强约束: Stage 4 评估端完全确定性 (单进程版) ===
    # 含 cudnn/CUBLAS/PYTHONHASHSEED (与 Stage 3 R51+ 块严格对齐)
    # 防止 DataLoader worker 启动顺序 / hash dict 顺序 / GEMM 算法选择 引入噪声
    import os as _os
    _os.environ.setdefault("PYTHONHASHSEED", "42")
    _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    import random as _random
    _random.seed(42)
    import numpy as _np
    _np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("highest")
    print("[R51+ seed] eval deterministic (cudnn/CUBLAS/hash)", flush=True)

    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=8,
        persistent_workers=True, pin_memory=True, prefetch_factor=4,
        collate_fn=hgrec_collate_fn,
        worker_init_fn=lambda wid: (__import__("numpy").random.seed(42 + wid), __import__("random").seed(42 + wid)),
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

    # === v66 mixed geometry rerank: 加载 RQ-VAE ckpt 拿 3 层 codebook (在主循环外, 只 load 一次) ===
    if V66_RERANK_ENABLED:
        from modules.hyperbolic import _expmap0_t, _poincare_distance_t
        # rank 0 先 load RQ-VAE ckpt, broadcast 给其他 rank (与 best_ckpt 同模式, 防 file system race)
        if accelerator.is_main_process:
            print(f"[v66] loading RQ-VAE ckpt: {V66_RERANK_RQVAE_CKPT}", flush=True)
            _rq_state = torch.load(V66_RERANK_RQVAE_CKPT, map_location="cpu", weights_only=True)["model"]
            _cb_list = [_rq_state[f"layers.{li}.embedding.weight"].to(device) for li in range(3)]
            _cb_payload = [t.cpu() for t in _cb_list]
            for li, t in enumerate(_cb_list):
                print(f"[v66] layer {li} codebook shape={tuple(t.shape)}", flush=True)
        else:
            _cb_payload = [None, None, None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(_cb_payload, src=0)
        _cb_list = [t.to(device) for t in _cb_payload]
        c_t = torch.tensor(V66_RERANK_C, device=device).view(1, 1)
        rank_score_template = -torch.arange(beam_size, device=device, dtype=torch.float32).view(1, beam_size)  # (1, beam)
        print(f"[v66] mixed rerank ready (alpha={V66_RERANK_BLEND_ALPHA}, hist_pool={V66_RERANK_HIST_POOL})", flush=True)

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

            # === v66 mixed geometry rerank (BLEND_ALPHA=0.5, hist_pool=mean) ===
            if V66_RERANK_ENABLED:
                B = input_ids.shape[0]
                # 1) history_emb: input_ids → SID → 3 层 codebook → concat(96) → mean pool (per-sample)
                hist_mask = input_ids != 0  # (B, H), 滤 PAD
                hist_vocab = input_ids.clamp(min=0)  # safety, 已非负
                hist_lid = ((hist_vocab - 1) // 256).clamp(min=0, max=2).long()  # (B, H) layer idx
                hist_cid = ((hist_vocab - 1) % 256).clamp(min=0, max=255).long()  # (B, H) code idx
                hist_layer_emb = []
                for li in range(3):
                    mask_li = (hist_lid == li) & hist_mask  # (B, H)
                    emb = _cb_list[li][hist_cid]  # (B, H, 32)
                    emb = emb * mask_li.unsqueeze(-1).float()
                    denom = mask_li.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
                    hist_layer_emb.append(emb.sum(dim=1) / denom)  # (B, 32)
                history_emb = torch.cat(hist_layer_emb, dim=-1)  # (B, 96)

                # 2) cand_emb: preds[:, :, :3] → 3 层 codebook → concat(96) per beam
                cand_vocab = preds[:, :, :3].clamp(min=0)  # (B, beam, 3)
                cand_cid = ((cand_vocab - 1) % 256).clamp(min=0, max=255).long()  # (B, beam, 3)
                cand_layer_emb = [_cb_list[li][cand_cid[:, :, li]] for li in range(3)]  # 3 × (B, beam, 32)
                cand_emb = torch.cat(cand_layer_emb, dim=-1)  # (B, beam, 96)

                # 3) Poincaré 距离 (c=0.7, 双曲 expmap0 + poincare_distance)
                history_ball = _expmap0_t(history_emb, c_t)  # (B, 96)
                cand_ball = _expmap0_t(cand_emb, c_t)  # (B, beam, 96)
                dist = _poincare_distance_t(
                    history_ball.unsqueeze(1).expand(-1, beam_size, -1), cand_ball, c_t
                ).squeeze(-1)  # (B, beam)
                # 归一到 [0, 1] 避免与 rank_score 量纲冲突
                dist_max = dist.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
                norm_dist = dist / dist_max  # (B, beam), [0, 1]

                # 4) mixed_score: alpha * (-rank) + (1-alpha) * (-norm_dist), 降序重排
                rank_score = rank_score_template.expand(B, -1)  # (B, beam)
                alpha = V66_RERANK_BLEND_ALPHA
                mixed_score = alpha * rank_score + (1.0 - alpha) * (-norm_dist)  # (B, beam)
                sorted_idx = mixed_score.argsort(dim=-1, descending=True)  # (B, beam)
                preds = torch.gather(
                    preds, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, preds.shape[-1])
                )

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
