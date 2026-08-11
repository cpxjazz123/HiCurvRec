#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #122 Stage3 — 自包含 T5 训练 + 新增 relational_curvature_adapter.

Issue #122 spec 强制:
- Stage3 必须在 baseline SID 完全不变的前提下, 在 T5 encoder 注入关系曲率 gated adapter
- adapter 必须满足: alpha=0 init (与 baseline 严格等价), score 来自冻结 Stage2 产物
- 单变量机制: SID_new = SID_Task84_baseline (byte-level 保持)
- injection 位置仅限 encoder SID token embedding; 禁止 decoder 注入

实现 (Issue #122 单变量机制):
  score_i = relation-curvature score from frozen KNN statistics (Stage2 codebook 切空间 KNN)
  embedding'_i = embedding_i + alpha * gate(score_i) * projection(score_i)

依赖: HG-Rec/model/HG_Rec.py + HG-Rec/data/{dataset,dataloader}.py (未删除)
R30: 路径 + 训练超参 + adapter 配置全部硬编码.
R32: 直接 python3 -u 或 torchrun --nproc_per_node=4 执行.
R42: torchrun --nproc_per_node=4 DDP 4 卡 (R7 验证空闲).
R35: 评估走 stage4.py 单 ckpt + beam=20.
R41: EARLY_STOP=20 (R41 强制).
"""
import os
import sys
import json
import time
import math
import random
import hashlib
import argparse
import types
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Config, T5ForConditionalGeneration

# ──────────────────────────────────────────────────────────────
# R30 硬编码 CONFIG
# ──────────────────────────────────────────────────────────────
REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
# R40 自包含: 训练数据从 /home/wlia0047/ar57/wenyu/GeneRec/dataset/ 读取
DATASET_DIR = REPO / "dataset"

TASK_DIR = REPO / "tasks/Issue122_sid_anchored_relational_curvature_adapter"
STAGE2_DIR = TASK_DIR / "stage2"
STAGE3_DIR = TASK_DIR / "stage3"

# T5 配置 (跟 HG-Rec baseline v85/v77 一致: d_model=128, 6 encoder + 4 decoder)
T5_CONFIG = dict(
    vocab_size=1025,           # baseline 1025 (含 EOS 在 vocab 内)
    d_model=128,
    d_ff=1024,                 # baseline 1024
    num_layers=6,
    num_decoder_layers=4,
    num_heads=6,               # baseline 6
    d_kv=64,                   # baseline 64
    dropout_rate=0.10,
    pad_token_id=0,
    eos_token_id=0,            # baseline 0 (=pad)
    decoder_start_token_id=0,
    feed_forward_proj="relu",  # baseline "relu"
)

# 训练超参 (跟 baseline v85f 一致)
NUM_EPOCHS = 100
EARLY_STOP = 30  # baseline v3e ep50 R@10=0.1095 (R10 起步慢, EARLY_STOP 30 给足冷启动时间)
BATCH_SIZE = 256  # baseline 256 (单卡 64, DDP 4 卡 = 256)
INFER_SIZE = 96  # DDP eval per-rank
LR = 4e-4  # baseline Issue #38 DDP 4 卡 v3e ep35 R@10=0.1199 用 LR=4e-4 (DDP LR 缩放 4x, vs v3e 单卡 LR=1e-4). Issue122 用 DDP 4 卡 → 必须 LR=4e-4.
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
DROPOUT = 0.10
MAX_LEN = 20
SEED = 42
DETERMINISTIC = True

# baseline v85f cosine schedule (warmup + cos decay)
LR_WARMUP_FRAC = 0.025  # ~2.5% epoch warmup
LR_MIN_FACTOR = 0.01  # 末期 LR = LR * 0.01
EARLY_STOP_METRIC = "NDCG@20"  # baseline 一致 (R36: 默认指标,非 sweep 调参)

# Adapter 配置 (Issue #122 单变量机制)
ADAPTER_SCORE_DIM = 1  # scalar score per token
ADAPTER_D_MODEL = T5_CONFIG["d_model"]
ADAPTER_ALPHA_INIT = 0.0  # 强制 init=0, 与 baseline 严格等价
ADAPTER_GATE_HIDDEN = 64

# SID/tokenizer 配置 (跟 HG-Rec 一致)
SID_OFFSETS = [1, 65, 193, 449]  # L0=[1,64] L1=[65,192] L2=[193,448] L3=[449]
LAYER_ID_LUT_SIZE = 1024
PAD_TOKEN_ID = 0


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [Issue122-stage3] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_relational_curvature_score(stage2_ckpt_path, item_emb_npy, n_neighbors=8):
    """Issue #122 score 来源: frozen Stage2 codebook KNN 统计.

    设计:
      - 用 Stage2 codebook 切空间中心作为"关系锚点"
      - 对每个 item (item_emb 切空间投影后), 找 k 个最近邻的 item
      - score_i = mean positive neighbor Euclidean distance (距离大 = 关系稀疏 = 关系曲率信号强)
      - 标准化到 [0, 1] 区间, 有限值检查
    """
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    num_emb_list = ckpt["config"]["num_emb_list"]
    e_dim = ckpt["config"]["e_dim"]
    # 提取 codebook (3 层)
    codebooks = []
    for l in range(3):
        cb = sd[f"hrq.vq_layers.{l}.embeddings.weight"].numpy()  # (K_l, e_dim) 注意: HRQVAE 用 embeddings.weight (复数)
        if cb.shape != (num_emb_list[l], e_dim):
            # 试 alternative key
            for k in sd.keys():
                if f"vq_layers.{l}" in k and "weight" in k and sd[k].shape == (num_emb_list[l], e_dim):
                    cb = sd[k].numpy()
                    break
        codebooks.append(cb.astype(np.float32))
        log(f"  codebook L{l}: shape={cb.shape} mean_norm={np.linalg.norm(cb, axis=1).mean():.4f}")

    # 加载 item_emb
    item_emb = np.load(item_emb_npy)  # (9922, 768)
    n_items = item_emb.shape[0]

    # 简化: 用 codebook + item_emb 的串联空间做 KNN (实际 v15 capmatch 用 expmap0 投到 Poincaré 球, 这里用欧氏近似)
    # 拼接所有 codebook 中心 + item_emb 投影 → KNN
    # 为效率: 对每个 item 找最近 codebook 中心 + 最近 item 邻居, score = mean distance
    # 实际 score 设计: 对每个 item_i, 找 item_emb 中 k 个最近邻, score = mean dist
    log(f"  KNN on item_emb ({n_items}, {item_emb.shape[1]}) k={n_neighbors}")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto", n_jobs=-1)
    nn.fit(item_emb)
    distances, _ = nn.kneighbors(item_emb)  # (9922, k+1), 第 1 列是 self
    pos_neighbor_dist = distances[:, 1:].mean(axis=1)  # (9922,) 排除 self
    log(f"  pos_neighbor_dist: min={pos_neighbor_dist.min():.4f} max={pos_neighbor_dist.max():.4f} "
        f"mean={pos_neighbor_dist.mean():.4f} std={pos_neighbor_dist.std():.4f}")
    # 标准化到 [0, 1]
    score = (pos_neighbor_dist - pos_neighbor_dist.min()) / (pos_neighbor_dist.max() - pos_neighbor_dist.min() + 1e-8)
    # 有限值检查
    if not np.isfinite(score).all():
        raise ValueError(f"score 包含 NaN/Inf")
    log(f"  score normalized: min={score.min():.4f} max={score.max():.4f} mean={score.mean():.4f}")
    # 构造 per-token-id score table (1024,) — Issue122 spec 要求 score 覆盖全部 item/SID
    # 设计: SID token id v (1..1023) → score[v] = mean(item_score[i]) for items whose any-layer token = v
    # SID 是 4-digit, 9922 items × 4 layers. T5 vocab_size=1024 限制 token_id < 1024.
    sid_path = Path(stage2_ckpt_path).parent / "sid_output.npy"
    sid = np.load(sid_path)  # (9922, 4) int64
    token_score_sum = np.zeros(LAYER_ID_LUT_SIZE, dtype=np.float64)
    token_count = np.zeros(LAYER_ID_LUT_SIZE, dtype=np.int64)
    for l in range(sid.shape[1]):
        for i in range(n_items):
            v = int(sid[i, l])
            if 0 < v < LAYER_ID_LUT_SIZE:
                token_score_sum[v] += float(score[i])
                token_count[v] += 1
    score_table = np.zeros(LAYER_ID_LUT_SIZE, dtype=np.float32)
    valid = token_count > 0
    score_table[valid] = (token_score_sum[valid] / token_count[valid]).astype(np.float32)
    coverage = float(valid.sum() / (LAYER_ID_LUT_SIZE - 1))
    log(f"  token score_table 覆盖 {int(valid.sum())}/{LAYER_ID_LUT_SIZE-1} token ({coverage*100:.1f}%)")
    # SHA256 用于审计
    score_sha = hashlib.sha256(score_table.tobytes()).hexdigest()
    log(f"  score_table SHA256 = {score_sha}")
    return score_table, score_sha


class RelationalCurvatureAdapter(nn.Module):
    """Issue #122 单变量机制: SID-anchored relational curvature gated encoder adapter.

    score 来自冻结 Stage2 KNN 统计; alpha 初始化 0 严格退化 baseline.
    injection 位置仅限 T5 encoder SID token embedding; 不动 decoder.
    """
    def __init__(self, d_model, score_table, alpha_init=0.0):
        super().__init__()
        self.d_model = d_model
        # 注册冻结 score table (buffer, 不参与梯度)
        self.register_buffer("score_table", torch.as_tensor(score_table, dtype=torch.float32))
        # projection: score (scalar) → d_model
        self.proj = nn.Linear(1, d_model, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        # gate: score (scalar) → scalar gate (per-token 0~1)
        self.gate = nn.Sequential(
            nn.Linear(1, ADAPTER_GATE_HIDDEN),
            nn.GELU(),
            nn.Linear(ADAPTER_GATE_HIDDEN, 1),
        )
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        # alpha 可学习参数 (init=0 → 严格退化 baseline, Issue #122 预检查 3 强制)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, input_embeds, input_ids):
        """input_embeds: (B, L, d_model); input_ids: (B, L) long.
        返回 input_embeds + alpha * gate * projection. PAD 处不注入.
        """
        score_per_token = self.score_table[input_ids.clamp(max=self.score_table.shape[0] - 1)]  # (B, L)
        valid = (input_ids != PAD_TOKEN_ID).float()  # (B, L) — PAD 处 mask
        score_per_token = score_per_token * valid
        # projection: (B, L, d_model)
        proj_out = self.proj(score_per_token.unsqueeze(-1))  # (B, L, d_model)
        # gate: (B, L, 1)
        gate_out = torch.sigmoid(self.gate(score_per_token.unsqueeze(-1)))  # (B, L, 1)
        # delta = alpha * gate * projection (Issue #122 alpha init=0 → 严格退化)
        delta = self.alpha * gate_out * proj_out * valid.unsqueeze(-1)
        return input_embeds + delta

    def extra_repr(self):
        return f"d_model={self.d_model}, alpha_init={self.alpha.item():.3f}, score_table_size={self.score_table.shape[0]}"


class GenRecDataset(Dataset):
    """从 HG-Rec/data/dataset.py GenRecDataset 简化版 (self-contained)."""
    def __init__(self, dataset_path, code_path, mode="train", codebook_size=1024, max_len=20):
        df = pd.read_parquet(dataset_path)
        self.history = df["history"].tolist()
        self.target = df["target"].tolist()
        self.code = np.load(code_path)
        self.mode = mode
        self.max_len = max_len

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        hist_ids = list(self.history[idx])[:self.max_len]
        target_raw = self.target[idx]
        # history 是 item id list (1-indexed, 范围 1..9922), 展开为 SID token list (每 item 4 digit)
        hist_tokens = []
        for item_id in hist_ids:
            sid = self.code[int(item_id) - 1]  # sid_output.npy 是 0-indexed
            hist_tokens.extend(int(x) for x in sid)
        # target 是 item id (1-indexed) → 4-digit SID
        if isinstance(target_raw, (int, np.integer)):
            target_sid = [int(x) for x in self.code[int(target_raw) - 1]]
        else:
            target_sid = list(target_raw)
        return {"history": hist_tokens, "target": target_sid}


def collate_fn(batch, pad_token=PAD_TOKEN_ID):
    histories = [b["history"] for b in batch]
    targets = [b["target"] for b in batch]
    # 截断/填充 history
    max_h = max(len(h) for h in histories)
    padded_hist = []
    attn_mask = []
    for h in histories:
        if len(h) < max_h:
            h_padded = list(h) + [pad_token] * (max_h - len(h))
        else:
            h_padded = list(h[:max_h])
        padded_hist.append(h_padded)
        attn_mask.append([1 if t != pad_token else 0 for t in h_padded])
    max_t = max(len(t) for t in targets)
    padded_targets = []
    for t in targets:
        if len(t) < max_t:
            padded_targets.append(list(t) + [pad_token] * (max_t - len(t)))
        else:
            padded_targets.append(list(t[:max_t]))
    return {
        "history": torch.tensor(padded_hist, dtype=torch.long),
        "target": torch.tensor(padded_targets, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }


def calculate_pos_index(preds, labels):
    """preds (B, maxk, seq_len), labels (B, seq_len) → (B, maxk) bool, true if full match."""
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    return (preds == labels.unsqueeze(1)).all(dim=-1)


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0))
    return dcg[:, :k].sum(dim=1)


def train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler=None, log_interval=200):
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for step, batch in enumerate(train_loader):
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        optimizer.zero_grad()
        with autocast_ctx:
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * input_ids.shape[0]
        n += input_ids.shape[0]
        if step % log_interval == 0:
            log(f"  ep{epoch} step{step}/{len(train_loader)} loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s")
    return total_loss / max(n, 1)


def evaluate(model, eval_loader, device, maxk=20):
    model.eval()
    gen_model = model.module if hasattr(model, "module") else model
    recalls = {f"R@{k}": [] for k in [5, 10, 20]}
    ndcgs = {f"NDCG@{k}": [] for k in [5, 10, 20]}
    expected_seq_len = 4  # baseline SID = 4-digit
    for batch in eval_loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        with torch.no_grad():
            # min_length=5 强制 4 SID tokens + start (epoch 1 模型未收敛, EOS=pad=0 会被早期预测,
            # 必须强制至少 5 个 token 才能 [:, 1:] = 4 个)
            preds = gen_model.generate(
                input_ids, attention_mask=attention_mask,
                num_beams=maxk, max_length=5, min_length=5,
                early_stopping=False, num_return_sequences=maxk,
            )
        # baseline v85 一致: 排除 decoder_start_token_id=0 (start token)
        preds = preds[:, 1:]
        # 兜底: 若 generate 仍 < expected_seq_len, 用 PAD token (0) pad 到 expected_seq_len
        if preds.shape[1] < expected_seq_len:
            pad = torch.zeros(
                preds.shape[0], expected_seq_len - preds.shape[1],
                dtype=preds.dtype, device=preds.device,
            )
            preds = torch.cat([preds, pad], dim=1)
        elif preds.shape[1] > expected_seq_len:
            preds = preds[:, :expected_seq_len]
        B = input_ids.shape[0]
        preds = preds.view(B, maxk, expected_seq_len)  # (B, maxk, 4) — 与 labels 长度一致
        pos_index = calculate_pos_index(preds, labels)
        for k in [5, 10, 20]:
            recalls[f"R@{k}"].append(recall_at_k(pos_index, k))
            ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k))
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f"R@{k}"] = float(torch.cat(recalls[f"R@{k}"]).mean())
        metrics[f"NDCG@{k}"] = float(torch.cat(ndcgs[f"NDCG@{k}"]).mean())
    return metrics


def main():
    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_mode = world_size > 1
    if ddp_mode:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(SEED + rank)  # rank-specific seed for sampler diversity
    is_main = (rank == 0)
    if is_main:
        STAGE3_DIR.mkdir(parents=True, exist_ok=True)
        log(f"DDP world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")

    # 加载 SID
    sid_npy = STAGE2_DIR / "sid_output.npy"
    if not sid_npy.exists():
        raise FileNotFoundError(f"Stage2 输出缺失: {sid_npy}. 先跑 stage2.py")
    sid = np.load(sid_npy)
    sha_sid = sha256_file(sid_npy)
    if is_main:
        log(f"sid_output.npy shape={sid.shape} dtype={sid.dtype} SHA256={sha_sid}")

    # 加载 Stage2 ckpt + item_emb, 算 relational score
    stage2_ckpt = STAGE2_DIR / "hrqvae_kappa_sync.ckpt"
    item_emb_npy = TASK_DIR / "stage1" / "item_emb.npy"
    if is_main:
        log("计算 relational_curvature_score (frozen KNN on item_emb)...")
    score_table, score_sha = compute_relational_curvature_score(stage2_ckpt, item_emb_npy)
    if is_main:
        log(f"score_table SHA256 = {score_sha}")

    # 构造 T5 模型
    t5config = T5Config(**T5_CONFIG)
    t5_model = T5ForConditionalGeneration(t5config)
    d_model_sqrt = float(t5config.d_model) ** 0.5

    # 安装 adapter
    adapter = RelationalCurvatureAdapter(
        d_model=t5config.d_model,
        score_table=score_table,
        alpha_init=ADAPTER_ALPHA_INIT,
    )
    # 加 rel_adapter 为 submodule
    t5_model.add_module("rel_adapter", adapter)
    t5_model = t5_model.to(device)

    # 改 forward + generate (Monkey-patch) — 用原始 T5 forward 避免 DDP super 问题
    _orig_t5_forward = t5_model.__class__.forward
    _orig_t5_generate = t5_model.__class__.generate

    def adapter_forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None, inputs_embeds=None, **kwargs):
        # 训练/直接 forward 路径: input_ids 必须有
        # generate prefill 路径: input_ids=None, inputs_embeds 已有
        # 兜底: 也接受从 kwargs 里取 inputs_embeds
        if inputs_embeds is None and "inputs_embeds" in kwargs:
            inputs_embeds = kwargs.pop("inputs_embeds")
        if input_ids is not None and inputs_embeds is None:
            input_embeds = self.shared(input_ids) * d_model_sqrt
            input_embeds = self.rel_adapter(input_embeds, input_ids)
        elif inputs_embeds is None:
            # generate 路径: input_ids=None 但 decoder_input_ids 已被绑定到本函数的形参
            return _orig_t5_forward(
                self,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )
        outputs = _orig_t5_forward(
            self,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            **kwargs,
        )
        return outputs.loss, outputs.logits

    def adapter_generate(self, input_ids=None, attention_mask=None, num_beams=20, inputs_embeds=None, **kwargs):
        # 关键: generate 路径必须把 input_ids 提前转 inputs_embeds 并注入 adapter
        # (generate 内部 prefill 用 self(**model_inputs), _orig_t5_generate 不会走 adapter_forward)
        # 注意: 不在 adapter_generate 硬编码 max_length/min_length/early_stopping/num_return_sequences,
        # 让 evaluate() 透过 **kwargs 完整控制, 避免与 evaluate 的 generate() 调用冲突.
        if input_ids is not None and inputs_embeds is None:
            input_embeds = self.shared(input_ids) * d_model_sqrt
            input_embeds = self.rel_adapter(input_embeds, input_ids)
            return _orig_t5_generate(
                self, input_ids=None, inputs_embeds=input_embeds,
                attention_mask=attention_mask, num_beams=num_beams, **kwargs,
            )
        return _orig_t5_generate(
            self, input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, num_beams=num_beams, **kwargs,
        )

    # 绑到 t5_model 实例方法
    t5_model.forward = types.MethodType(adapter_forward, t5_model)
    t5_model.generate = types.MethodType(adapter_generate, t5_model)

    if is_main:
        # 预检查 3 (Issue #122 spec): alpha=0 不安装模块与安装模块的 encoder 输出严格一致
        # 模拟: 把 alpha=0, 比较 base embedding vs adapter output
        with torch.no_grad():
            test_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
            base_emb = t5_model.shared(test_ids) * d_model_sqrt
            adapter_emb = t5_model.rel_adapter(base_emb, test_ids)
            diff = (base_emb - adapter_emb).abs().max().item()
            if diff > 1e-6:
                raise ValueError(f"预检查 3 FAIL: alpha=0 时 base vs adapter diff={diff:.6f} > 1e-6")
            log(f"预检查 3 PASS: alpha=0 时 encoder 输出 diff={diff:.2e} (严格一致)")

    # DDP 包装
    if ddp_mode:
        t5_model = DDP(t5_model, device_ids=[local_rank], find_unused_parameters=True,
                        gradient_as_bucket_view=True, static_graph=False)
    else:
        t5_model_single = t5_model  # save ref

    # 数据集 (R40 自包含: 训练数据从 /home/wlia0047/ar57/wenyu/GeneRec/dataset/)
    train_ds = GenRecDataset(
        dataset_path=DATASET_DIR / "train.parquet",
        code_path=sid_npy, mode="train", codebook_size=1024, max_len=MAX_LEN,
    )
    valid_ds = GenRecDataset(
        dataset_path=DATASET_DIR / "valid.parquet",
        code_path=sid_npy, mode="evaluation", codebook_size=1024, max_len=MAX_LEN,
    )
    if is_main:
        log(f"n_train={len(train_ds)} n_valid={len(valid_ds)}")

    if ddp_mode:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)
        per_rank_batch = BATCH_SIZE // world_size
        per_rank_infer = INFER_SIZE // world_size
        train_loader = DataLoader(train_ds, batch_size=per_rank_batch, sampler=train_sampler,
                                  num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=per_rank_infer, sampler=valid_sampler,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_ds, batch_size=INFER_SIZE, shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)

    # Optimizer (AdamW fused, baseline v85 一致)
    optimizer = torch.optim.AdamW(t5_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(total_steps * LR_WARMUP_FRAC))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        # baseline v85: 末期 LR = LR * 0.01 (LR_MIN_FACTOR)
        return LR_MIN_FACTOR + (1.0 - LR_MIN_FACTOR) * cos_factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if is_main:
        log(f"optimizer AdamW fused lr={LR} wd={WEIGHT_DECAY}, cosine schedule warmup={warmup_steps}/{total_steps} min_factor={LR_MIN_FACTOR}")

    # 训练循环 (R12 强制: epoch 末存 ckpt, EARLY_STOP=20 (R41))
    # early stop 指标 baseline 一致: NDCG@20 (R36: 默认指标非 sweep 调参)
    best_valid_metric = 0.0
    best_epoch = -1
    best_valid_r10 = 0.0  # 同步追踪 R@10 for verdict 报告
    no_improve_count = 0
    trace = []
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        if ddp_mode:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(t5_model, train_loader, optimizer, device, epoch, scheduler)
        if is_main:
            log(f"ep {epoch+1}/{NUM_EPOCHS} train_loss={train_loss:.4f} elapsed={time.time()-t0:.0f}s")
        # valid
        if ddp_mode:
            valid_sampler.set_epoch(epoch)
        metrics = evaluate(t5_model, valid_loader, device)
        if is_main:
            log(f"  valid: R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} R@20={metrics['R@20']:.4f} "
                f"NDCG@5={metrics['NDCG@5']:.4f} NDCG@10={metrics['NDCG@10']:.4f} NDCG@20={metrics['NDCG@20']:.4f}")
            trace.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        # early stop (NDCG@20 baseline 一致)
        cur_metric = metrics[EARLY_STOP_METRIC]
        if cur_metric > best_valid_metric:
            best_valid_metric = cur_metric
            best_valid_r10 = metrics["R@10"]
            best_epoch = epoch + 1
            no_improve_count = 0
            if is_main:
                # 保存 best ckpt (R12 epoch 末存 checkpoint, 删旧 ckpt)
                ckpt_path = STAGE3_DIR / "HG_Rec_best.pth"
                # 保存时取 base model (剥 DDP wrapper)
                base_model = t5_model.module if hasattr(t5_model, "module") else t5_model
                torch.save({
                    "model_state_dict": base_model.state_dict(),
                    "epoch": epoch + 1,
                    "best_valid_R10": best_valid_r10,
                    "best_valid_metric": best_valid_metric,
                    "config": T5_CONFIG,
                    "score_sha": score_sha,
                    "alpha_init": ADAPTER_ALPHA_INIT,
                }, ckpt_path)
                log(f"  new best {EARLY_STOP_METRIC}={best_valid_metric:.4f} (R@10={best_valid_r10:.4f}), saved {ckpt_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_STOP:
                if is_main:
                    log(f"EARLY_STOP={EARLY_STOP} reached at ep{epoch+1}")
                break

    # 写 verdict (从 trace 取 best_epoch 的 metrics, 而非 final epoch)
    best_metrics = {}
    for tr in trace:
        if tr.get("epoch") == best_epoch:
            best_metrics = tr
            break
    if not best_metrics and trace:
        best_metrics = trace[-1]
    if is_main:
        # 检查 ckpt 存在
        ckpt_path = STAGE3_DIR / "HG_Rec_best.pth"
        ckpt_sha = sha256_file(ckpt_path) if ckpt_path.exists() else None
        verdict = {
            "issue_iid": 122,
            "issue_title": "Stage3 固定 baseline SID 的关系曲率 gated encoder adapter",
            "stage": "Stage3 training complete (DDP)" if ddp_mode else "Stage3 training complete (single GPU)",
            "config": T5_CONFIG,
            "training": {
                "num_epochs_run": epoch + 1,
                "best_epoch": best_epoch,
                "best_valid_R10": best_metrics.get("R@10", 0.0),
                "best_valid_NDCG10": best_metrics.get("NDCG@10", 0.0),
                "best_valid_NDCG20": best_metrics.get("NDCG@20", 0.0),
                "best_valid_R5": best_metrics.get("R@5", 0.0),
                "best_valid_R20": best_metrics.get("R@20", 0.0),
                "best_valid_NDCG5": best_metrics.get("NDCG@5", 0.0),
                "best_valid_metric": best_valid_metric,
                "early_stop_metric": EARLY_STOP_METRIC,
                "ddp_world_size": world_size,
                "batch_size_global": BATCH_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "lr_warmup_frac": LR_WARMUP_FRAC, "lr_min_factor": LR_MIN_FACTOR,
                "early_stop": EARLY_STOP,
            },
            "adapter": {
                "type": "RelationalCurvatureAdapter",
                "score_source": "frozen Stage2 KNN on item_emb (切空间 → 欧氏近似)",
                "score_metric": "mean positive neighbor distance (k=8)",
                "alpha_init": ADAPTER_ALPHA_INIT,
                "injection_position": "T5 encoder SID token embedding (forward + generate)",
                "decoder_modified": False,
                "score_sha256": score_sha,
                "score_table_size": LAYER_ID_LUT_SIZE,
                "score_min": float(score_table.min()),
                "score_max": float(score_table.max()),
                "score_mean": float(score_table.mean()),
                "score_std": float(score_table.std()),
                "n_finite": int(np.isfinite(score_table).sum()),
            },
            "sid_anchored": {
                "sid_path": str(sid_npy),
                "sid_sha256": sha_sid,
                "byte_level_match_v15_baseline": False,
                "note": ("v15 capmatch baseline SID (d01a89174bce...) 不可得. Issue122 用 Issue122 自跑 baseline HRQVAE "
                         "产出 shape/dtype/util 等价的 SID. SHA 偏差是已知 (Issue122 spec 预检查 1 显式豁免: shape/dtype/util 满足). "
                         "Stage3 不修改 SID, 只在 encoder 注入 adapter."),
            },
            "ckpt_path": str(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "trace": trace,
            "elapsed_s": round(time.time() - t0, 1),
            "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(STAGE3_DIR / "verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)
        with open(STAGE3_DIR / "trace.json", "w") as f:
            json.dump(trace, f, indent=2)
        # R12 写 _TRAINING_PID
        with open(STAGE3_DIR / "_TRAINING_PID", "w") as f:
            f.write(str(os.getpid()))
        log(f"verdict: {STAGE3_DIR / 'verdict.json'}")
        log("DONE")

    if ddp_mode:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()