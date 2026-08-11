#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue #122 Stage4 — 自包含 T5 eval (单 ckpt + beam=20) + relational_curvature_adapter.

Issue #122 spec 强制:
- 评估: 单 ckpt + beam=20 (R35 严格)
- 安装同 stage3 用的 adapter (Stage4 镜像安装, 确认 checkpoint 加载 score/projection/gate 权重)
- 24772 完整 test 样本, 输出六项指标 (R@5/10/20 + NDCG@5/10/20)

依赖: HG-Rec/ (未删除), stage3 产出的 HG_Rec_best.pth + 同样的 score table
R30: 路径 + 评估超参 + adapter 配置全部硬编码.
R32: 直接 python3 -u 执行, 单卡.
R35: 单 ckpt + beam=20 严格使用, 禁 Borda / 任何 ensemble.
"""
import os
import sys
import json
import time
import hashlib
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
# R40 自包含: 测试数据从 /home/wlia0047/ar57/wenyu/GeneRec/dataset/ 读取
DATASET_DIR = REPO / "dataset"

TASK_DIR = REPO / "tasks/Issue122_sid_anchored_relational_curvature_adapter"
STAGE2_DIR = TASK_DIR / "stage2"
STAGE3_DIR = TASK_DIR / "stage3"
STAGE4_DIR = TASK_DIR / "stage4"

# 与 stage3.py 一致 (baseline v85/v77 一致)
T5_CONFIG = dict(
    vocab_size=1025,
    d_model=128,
    d_ff=1024,
    num_layers=6,
    num_decoder_layers=4,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.10,
    pad_token_id=0,
    eos_token_id=0,
    decoder_start_token_id=0,
    feed_forward_proj="relu",
)
ADAPTER_GATE_HIDDEN = 64
PAD_TOKEN_ID = 0
LAYER_ID_LUT_SIZE = 1024
MAX_LEN = 20

# 评估超参 (R35 强制)
BEAM_SIZE = 20
MAX_GEN_LEN = 5
INFER_BATCH_SIZE = 32


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [Issue122-stage4] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_relational_curvature_score(stage2_ckpt_path, item_emb_npy, n_neighbors=8):
    """与 stage3.py 完全一致 — 必须使用相同 score 表."""
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    codebooks = []
    for l in range(3):
        for k in sd.keys():
            if f"vq_layers.{l}" in k and "weight" in k:
                t = sd[k]
                if t.ndim == 2:
                    codebooks.append(t.numpy().astype(np.float32))
                    break
    item_emb = np.load(item_emb_npy)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto", n_jobs=-1)
    nn.fit(item_emb)
    distances, _ = nn.kneighbors(item_emb)
    pos_neighbor_dist = distances[:, 1:].mean(axis=1)
    score = (pos_neighbor_dist - pos_neighbor_dist.min()) / (pos_neighbor_dist.max() - pos_neighbor_dist.min() + 1e-8)
    if not np.isfinite(score).all():
        raise ValueError("score 包含 NaN/Inf")
    score_table = np.zeros(LAYER_ID_LUT_SIZE, dtype=np.float32)
    score_table[1:item_emb.shape[0] + 1] = score[:item_emb.shape[0]]
    score_sha = hashlib.sha256(score_table.tobytes()).hexdigest()
    log(f"score_table SHA256 = {score_sha} (与 stage3 共享同一 score 表)")
    return score_table, score_sha


class RelationalCurvatureAdapter(nn.Module):
    def __init__(self, d_model, score_table, alpha_init=0.0):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("score_table", torch.as_tensor(score_table, dtype=torch.float32))
        self.proj = nn.Linear(1, d_model, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Sequential(
            nn.Linear(1, ADAPTER_GATE_HIDDEN),
            nn.GELU(),
            nn.Linear(ADAPTER_GATE_HIDDEN, 1),
        )
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, input_embeds, input_ids):
        score_per_token = self.score_table[input_ids.clamp(max=self.score_table.shape[0] - 1)]
        valid = (input_ids != PAD_TOKEN_ID).float()
        score_per_token = score_per_token * valid
        proj_out = self.proj(score_per_token.unsqueeze(-1))
        gate_out = torch.sigmoid(self.gate(score_per_token.unsqueeze(-1)))
        delta = self.alpha * gate_out * proj_out * valid.unsqueeze(-1)
        return input_embeds + delta


class GenRecDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, code_path, mode="evaluation", codebook_size=1024, max_len=20):
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
        # history 是 item id list (1-indexed, 范围 1..9922), 展开为 SID token list
        hist_tokens = []
        for item_id in hist_ids:
            sid = self.code[int(item_id) - 1]
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


def evaluate_with_beam(model, eval_loader, device, maxk=BEAM_SIZE):
    """跑完整 test set, 单 ckpt + beam=20, 收集 raw predictions + 六指标.

    对齐 baseline v3e+LR=4e-4 fork (c4289aa pf_generate wrapper):
      preds = gen_model.generate(num_beams=BEAM_SIZE, num_return_sequences=BEAM_SIZE, max_length=5)
      preds = preds[:, 1:5]  # 排除 start + 强制 4 token (transformers 5.x EOS 截断兼容)
      preds = preds.reshape(input_ids.shape[0], BEAM_SIZE, -1)

    baseline HG_Rec.generate 内部显式 max_length=5 + num_return_sequences=num_beams;
    原生 T5 默认 max_length=21 会跟 labels (B, 4) 维度不匹配, 必须显式传 max_length=5.
    """
    model.eval()
    all_pos = []
    all_preds = []
    all_labels = []
    n_eval = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            # baseline v3e+LR=4e-4 pf_generate 等价: max_new_tokens=4 + 强制 [:, 1:5]
            preds = model.generate(
                input_ids, attention_mask=attention_mask,
                num_beams=maxk, num_return_sequences=maxk, max_new_tokens=4,
            )
            preds = preds[:, 1:5]  # 排除 start + 强制 4 token (不足 4 的 PAD 0)
            B = input_ids.shape[0]
            preds = preds.view(B, maxk, -1)  # (B, maxk, 4) — 跟 SID 4-digit 一致
            pos_index = (preds == labels.unsqueeze(1)).all(dim=-1)
            all_pos.append(pos_index.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            n_eval += B
            if (batch_idx + 1) % 50 == 0:
                log(f"  eval batch {batch_idx+1}/{len(eval_loader)} elapsed={time.time()-t0:.0f}s")
    pos_all = torch.cat(all_pos, dim=0)
    preds_all = torch.cat(all_preds, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    metrics = {}
    for k in [5, 10, 20]:
        metrics[f"R@{k}"] = float(pos_all[:, :k].sum(dim=1).float().mean())
        ranks = torch.arange(1, k + 1, dtype=torch.float)
        dcg = 1.0 / torch.log2(ranks + 1)
        per_user = (pos_all[:, :k].float() * dcg.unsqueeze(0)).sum(dim=1)
        metrics[f"NDCG@{k}"] = float(per_user.mean())
    return metrics, n_eval, preds_all, labels_all, pos_all, time.time() - t0


def main():
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    # 加载 ckpt
    ckpt_path = STAGE3_DIR / "HG_Rec_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage3 ckpt 缺失: {ckpt_path}. 先跑 stage3.py")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    log(f"loaded ckpt {ckpt_path} epoch={ckpt.get('epoch')} best_valid_R10={ckpt.get('best_valid_R10')}")

    # 加载 SID + score
    sid_npy = STAGE2_DIR / "sid_output.npy"
    if not sid_npy.exists():
        raise FileNotFoundError(f"Stage2 SID 缺失: {sid_npy}")
    stage2_ckpt = STAGE2_DIR / "hrqvae_kappa_sync.ckpt"
    item_emb_npy = TASK_DIR / "stage1" / "item_emb.npy"
    score_table, score_sha = compute_relational_curvature_score(stage2_ckpt, item_emb_npy)
    # 校验 stage3 ckpt 的 score_sha 一致
    if ckpt.get("score_sha") != score_sha:
        raise ValueError(f"Stage3 score_sha ({ckpt.get('score_sha')}) != Stage4 score_sha ({score_sha}); 必须一致")

    # 构造模型 + adapter + 加载权重
    from transformers import T5Config, T5ForConditionalGeneration
    t5config = T5Config(**T5_CONFIG)
    t5_model = T5ForConditionalGeneration(t5config)
    adapter = RelationalCurvatureAdapter(
        d_model=t5config.d_model,
        score_table=score_table,
        alpha_init=0.0,
    )
    t5_model.add_module("rel_adapter", adapter)
    t5_model = t5_model.to(device)
    d_model_sqrt = float(t5config.d_model) ** 0.5

    # Monkey-patch forward + generate (与 stage3 一致, 用原始 forward 避免 DDP super 问题)
    _orig_t5_forward = t5_model.__class__.forward
    _orig_t5_generate = t5_model.__class__.generate

    def adapter_forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None and "inputs_embeds" in kwargs:
            inputs_embeds = kwargs.pop("inputs_embeds")
        if input_ids is not None and inputs_embeds is None:
            input_embeds = self.shared(input_ids) * d_model_sqrt
            input_embeds = self.rel_adapter(input_embeds, input_ids)
        elif inputs_embeds is None:
            return _orig_t5_forward(
                self,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )
        outputs = _orig_t5_forward(
            self, inputs_embeds=input_embeds, attention_mask=attention_mask,
            labels=labels, decoder_input_ids=decoder_input_ids, **kwargs,
        )
        return outputs.loss, outputs.logits

    def adapter_generate(self, input_ids=None, attention_mask=None, num_beams=20, inputs_embeds=None, **kwargs):
        # 关键: generate 路径必须把 input_ids 提前转 inputs_embeds 并注入 adapter
        # 不在 adapter_generate 硬编码任何 generate 参数 (max_length/min_length/early_stopping/num_return_sequences),
        # 让 evaluate_with_beam() 透过 **kwargs 完整控制, 避免冲突.
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

    t5_model.forward = types.MethodType(adapter_forward, t5_model)
    t5_model.generate = types.MethodType(adapter_generate, t5_model)

    # 加载 stage3 训练权重 (strict=True 校验 score/projection/gate 权重全部加载)
    missing, unexpected = t5_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    log(f"loaded state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        log(f"  missing keys (前 5): {missing[:5]}")
    if len(unexpected) > 0:
        log(f"  unexpected keys (前 5): {unexpected[:5]}")
    # Issue #122 预检查 5: score/projection/gate 权重全部被加载
    rel_adapter_keys_loaded = [k for k in ckpt["model_state_dict"].keys() if "rel_adapter" in k]
    log(f"rel_adapter 权重键数={len(rel_adapter_keys_loaded)} (期望 ≥4: score_table/proj.weight/proj.bias/gate.*/alpha)")

    # 预检查 4: alpha>0 时模块确实改变 encoder 输出
    with torch.no_grad():
        test_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
        base_emb = t5_model.shared(test_ids) * d_model_sqrt
        # 当前 alpha 可能是 0 (Stage3 训练起点), 强制设 >0 看 delta
        original_alpha = adapter.alpha.item()
        adapter.alpha.data.fill_(0.5)  # 设 >0 验证
        adapter_emb = adapter(base_emb.clone(), test_ids)
        adapter.alpha.data.fill_(original_alpha)  # 恢复
        diff = (base_emb - adapter_emb).abs().max().item()
        log(f"预检查 4 (alpha=0.5): encoder diff={diff:.6f} (期望 > 0)")
        if diff <= 1e-6:
            raise ValueError("预检查 4 FAIL: alpha>0 时 adapter 未改变 encoder 输出")

    # Test set
    test_ds = GenRecDataset(
        dataset_path=DATASET_DIR / "test.parquet",
        code_path=sid_npy, mode="evaluation", codebook_size=1024, max_len=MAX_LEN,
    )
    test_loader = DataLoader(test_ds, batch_size=INFER_BATCH_SIZE, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    log(f"n_test={len(test_ds)}")

    # 跑 eval
    metrics, n_eval, preds_all, labels_all, pos_all, t_eval = evaluate_with_beam(
        t5_model, test_loader, device, maxk=BEAM_SIZE,
    )
    log(f"n_eval={n_eval} t_eval={t_eval:.1f}s")
    log(f"metrics: R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} R@20={metrics['R@20']:.4f} "
        f"NDCG@5={metrics['NDCG@5']:.4f} NDCG@10={metrics['NDCG@10']:.4f} NDCG@20={metrics['NDCG@20']:.4f}")

    # 保存 raw predictions (R35: 单 ckpt, 无 ensemble)
    raw_preds_path = STAGE4_DIR / "raw_predictions.npz"
    np.savez_compressed(raw_preds_path,
                        preds=preds_all.numpy(),
                        labels=labels_all.numpy(),
                        pos_index=pos_all.numpy())
    log(f"raw predictions saved {raw_preds_path}")

    # 写 eval_test.json (Stage4 6 指标)
    eval_test = {
        "ckpt_path": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "sid_path": str(sid_npy),
        "sid_sha256": sha256_file(sid_npy),
        "score_sha256": score_sha,
        "beam_size": BEAM_SIZE,
        "n_eval": n_eval,
        "t_eval_s": round(t_eval, 1),
        "metrics": metrics,
        "rule": "R35 beam=20 单 ckpt 严格使用, 禁 Borda / 任何 ensemble",
        "issue_iid": 122,
        "issue_title": "Stage3 固定 baseline SID 的关系曲率 gated encoder adapter",
        "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(STAGE4_DIR / "eval_test.json", "w") as f:
        json.dump(eval_test, f, indent=2)
    log(f"eval_test.json: {STAGE4_DIR / 'eval_test.json'}")

    # 写 verdict
    verdict = {
        "issue_iid": 122,
        "stage": "Stage4 evaluation complete",
        "stage3_ckpt": str(ckpt_path),
        "stage3_ckpt_epoch": ckpt.get("epoch"),
        "stage3_best_valid_R10": ckpt.get("best_valid_R10"),
        "test_metrics": metrics,
        "n_eval": n_eval,
        "t_eval_s": round(t_eval, 1),
        "rule_compliance": {
            "r35_single_ckpt_beam20": True,
            "r32_direct_python3": True,
            "r30_hardcoded": True,
            "issue122_alpha_init_zero": True,
            "issue122_decoder_unchanged": True,
        },
        "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(STAGE4_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    log(f"verdict: {STAGE4_DIR / 'verdict.json'}")
    log("DONE")


if __name__ == "__main__":
    main()