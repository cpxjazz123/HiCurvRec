import os
import random as _random
import numpy as _np
import gin
import torch
import wandb

# R51+ 6 确定性约束 (Stage 3 训练, R47 联动)
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from _eval_metrics.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


@gin.configurable
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
    top_k_for_generation=10,
    should_add_sep_token=True,
    num_user_bins=None,
    top_k_eval_list=[1, 5, 10],
):
    if dataset not in (RecDataset.AMAZON, RecDataset.INSTRUMENTS):
        raise Exception(f"Dataset currently not supported: {dataset}.")

    # R51+: Stage 3 训练端, accelerator 创建之前显式 seed (per-rank 偏移保证 4 卡不同 seed)
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _random.seed(42 + _local_rank)
    _np.random.seed(42 + _local_rank)
    torch.manual_seed(42 + _local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + _local_rank)

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
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )
    if False:
        model = torch.compile(model)  # 关闭 torch.compile (4 卡 DDP 兼容性问题)

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
    BEST_CKPT_PATH = "out/decoder/instruments/best_ckpt.pt"

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
    parse_config()
    train()
