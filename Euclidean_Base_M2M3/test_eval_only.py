"""DDP 4 卡 test eval (加载已训好的 best_ckpt 跑 test.parquet).

修复 #14 race condition: rank 0 先 torch.load best_ckpt, 然后用
torch.distributed.broadcast 把 state_dict 传给其他 3 个 rank — 避免 4 个 rank
同时 load 同一文件的 race condition / 文件系统缓存问题.

R35b: 4 卡分片不重复评估 test 集, all_reduce SUM, 统一除以全局 total.
"""
import os
import sys
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
from modules.tokenizer.semids import SemanticIdTokenizer
from torch.utils.data import DataLoader


def main():
    # 与 train_decoder.py 同样的 gin config
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    accelerator = Accelerator(split_batches=True, mixed_precision="no")
    device = accelerator.device

    # gin config 必须显式 parse (复用 train_decoder 的 config)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/decoder_instruments.gin"
    gin.parse_config_file(config_path)

    BEST_CKPT_PATH = sys.argv[2] if len(sys.argv) > 2 else "out/decoder/instruments/best_ckpt.pt"

    # 1) load datasets & tokenizer (走 train 同样的路径)
    item_dataset = ItemData(
        root="dataset/", dataset=RecDataset.INSTRUMENTS, split="instruments",
    )
    test_dataset = SeqData(
        root="dataset/", dataset=RecDataset.INSTRUMENTS,
        is_train=False, subsample=False, split="test",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2,
        persistent_workers=True, pin_memory=True,
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


if __name__ == "__main__":
    main()
