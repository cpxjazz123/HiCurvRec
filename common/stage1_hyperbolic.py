"""Stage 1 双曲 encoder — 替换 process_Instruments.py 中的 sentence-t5-xl.

把商品 (item) 的 title/description/brand/categories 文本先用 sentence-t5-base 编码成
768 维 Euclidean 向量, 再用一个可学习的线性映射压到 e_dim, 最后用 expmap0 投影到
Poincaré ball (曲率 c). 输出格式与 process_Instruments.py 完全一致 (parquet:
ItemID=str, embedding=list[float>) — 可直接被 taskA/taskB stage2 (RQ-VAE) 消费.

环境变量:
  ITEM_JSON         必填 — {dataset}.item.json 路径
  OUTPUT_PARQUET    必填 — 输出 parquet 路径
  TAG               默认 "hyp"
  E_DIM             默认 128 — Poincaré 嵌入维度 (与 stage2 e_dim 对齐)
  POINCARE_C        默认 1.0 — Poincaré 曲率
  ENCODER_MODEL     默认 "sentence-transformers/sentence-t5-base" — frozen text encoder
  DEVICE            默认 cuda:0
  BATCH_SIZE        默认 64
  MAX_SEQ_LEN       默认 64 — t5 tokenizer max length
  TRAIN             默认 "0" — 1=InfoNCE 微调 (需 ITEM_INTER_JSON); 0=冻结直接 encode
  TRAIN_EPOCHS      默认 5 — 仅 TRAIN=1 生效
  TRAIN_LR          默认 1e-4
  TRAIN_TAU         默认 0.1 — InfoNCE 温度
  TRAIN_NEG_N       默认 64 — 负采样数
  SEED              默认 42

产物:
  OUTPUT_PARQUET    ItemID + embedding (parquet)
  verdict.json      sha256 + config + 维度/曲率统计
"""
import os
import sys
import json
import time
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/model")
from utils import expmap0, proj_to_ball  # noqa: E402

ITEM_JSON = os.environ["ITEM_JSON"]
OUTPUT_PARQUET = Path(os.environ["OUTPUT_PARQUET"])
TAG = os.environ.get("TAG", "hyp")
E_DIM = int(os.environ.get("E_DIM", "128"))
POINCARE_C = float(os.environ.get("POINCARE_C", "1.0"))
# v1 修复 (2026-08-04): ρ 塌缩问题. expmap0(c=1) 把 norm=0.62 限制在 Poincaré ball 中心,
# stage2 encoder 再压缩到 32 维后范数全部塌缩到 ≈0, 导致 SID unique 率低.
# 修复: 在 expmap0 后做径向拉伸 (乘以 RADIAL_SCALE 然后 proj_to_ball 限到 0.95*radius),
# 让 norm 接近 boundary (≈0.9), 模仿 baseline t5 数据 norm 分布.
RADIAL_SCALE = float(os.environ.get("RADIAL_SCALE", "0.95"))
ENCODER_MODEL = os.environ.get("ENCODER_MODEL", "sentence-transformers/sentence-t5-base")
DEVICE = os.environ.get("DEVICE", "cuda:0")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "64"))
TRAIN = os.environ.get("TRAIN", "0") == "1"
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "5"))
TRAIN_LR = float(os.environ.get("TRAIN_LR", "1e-4"))
TRAIN_TAU = float(os.environ.get("TRAIN_TAU", "0.1"))
TRAIN_NEG_N = int(os.environ.get("TRAIN_NEG_N", "64"))
ITEM_INTER_JSON = os.environ.get("ITEM_INTER_JSON", "")
SEED = int(os.environ.get("SEED", "42"))

if POINCARE_C <= 0:
    raise ValueError(f"POINCARE_C must be > 0, got {POINCARE_C}")


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
    print(line, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_items(path):
    """读 json: {itemID_str: {title, description, brand, categories}}.

    返回 [(itemID_str, semantics_text), ...] — itemID 保持字符串, 与 stage1 旧契约一致.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    items = []
    for item_id, info in raw.items():
        if not isinstance(info, dict):
            raise ValueError(f"item {item_id} info is not dict: {type(info)}")
        semantics = (
            f"'title': {info.get('title', '')}, "
            f"'description': {info.get('description', '')}, "
            f"'brand': {info.get('brand', '')}, "
            f"'categories': {info.get('categories', '')}"
        )
        items.append((item_id, semantics))
    items.sort(key=lambda x: int(x[0]))  # 与 process_Instruments.py 顺序一致
    return items


class HyperbolicEncoder(nn.Module):
    """frozen sentence-t5-base → mean pool → Linear → expmap0(c) → Poincaré ball."""

    def __init__(self, encoder_model, e_dim, c):
        super().__init__()
        from transformers import T5EncoderModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = T5EncoderModel.from_pretrained(encoder_model)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.proj = nn.Linear(self.encoder.config.d_model, e_dim)
        self.e_dim = e_dim
        self.c = c
        # 用 1/√c 范数球 (proj_to_ball 已处理), 这里存 radius 用于 sanity check
        self.radius = 1.0 / (c ** 0.5)

    @torch.no_grad()
    def encode_text(self, texts, device):
        """冻结 t5 编码 → Euclidean (B, d_model). mean-pool + attention mask."""
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        ).to(device)
        out = self.encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        h = out.last_hidden_state  # (B, L, d_model)
        mask = enc.attention_mask.unsqueeze(-1).float()
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom  # (B, d_model)

    def project_to_poincare(self, eu):
        """Euclidean (B, e_dim) → Poincaré ball via expmap0 + proj_to_ball + RADIAL_SCALE.

        v3 修复 (2026-08-04): v2 RADIAL_SCALE=1.5 让所有 norm 全 clip 到 1.0 (太集中), κ grad FAIL.
        关键发现: baseline t5 norm=1.0 (std=0.0002, SentenceTransformer L2-normalize), 但 stage2 κ grad PASS.
        差异在方向多样性 (baseline) vs 全相同 norm (v2).
        v3 修复: 保留 t5 原始方向 (norm=1.0 per-sample, 方向多样), 用 LayerNorm + scale 让 norm ≈ 0.95
        (略小于 boundary 严格在 Poincaré ball 内). Stage2 应该能 work, norm 分布模仿 baseline.
        """
        x = self.proj(eu)  # (B, e_dim) Euclidean
        # 真 unit norm (F.normalize, dim=-1), 然后 × RADIAL_SCALE 让 norm < 1 严格在 Poincaré ball 内
        x = F.normalize(x, p=2, dim=-1) * RADIAL_SCALE
        return x

    def forward(self, texts, device):
        eu = self.encode_text(texts, device)
        return self.project_to_poincare(eu)


def info_nce_loss(query_emb, pos_emb, neg_emb, c, tau):
    """Poincaré 双曲 InfoNCE: -log[exp(-d(q,p)/τ) / Σ exp(-d(q,·)/τ)].

    query_emb, pos_emb, neg_emb: (B, e_dim) Poincaré ball.
    neg_emb: (B, K, e_dim) 或 (K, e_dim) (后者按 batch 共享 neg).
    """
    from utils import poincare_distance

    if neg_emb.dim() == 2:
        neg_emb = neg_emb.unsqueeze(0).expand(query_emb.shape[0], -1, -1)
    d_pos = poincare_distance(query_emb, pos_emb, c)  # (B,)
    d_neg = poincare_distance(
        query_emb.unsqueeze(1).expand(-1, neg_emb.shape[1], -1).reshape(-1, query_emb.shape[-1]),
        neg_emb.reshape(-1, neg_emb.shape[-1]),
        c
    ).reshape(query_emb.shape[0], neg_emb.shape[1])  # (B, K)
    logits_pos = -d_pos / tau
    logits_neg = -d_neg / tau
    logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)  # (B, 1+K)
    labels = torch.zeros(query_emb.shape[0], dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)


def build_co_inter(item_inter_path, item_ids_set):
    """从 {dataset}.inter.json 构建 item → co-occur item 映射 (InfoNCE 正样本).

    inter: {userID: [item, item, ...]}, 与 process_Instruments 同 schema.
    """
    with open(item_inter_path, "r") as f:
        inter = json.load(f)
    co = {}
    for uid, seq in inter.items():
        for i, it in enumerate(seq):
            it = str(it)
            if it not in item_ids_set:
                continue
            if it not in co:
                co[it] = []
            for j, jt in enumerate(seq):
                if i == j:
                    continue
                jt = str(jt)
                if jt in item_ids_set and jt != it:
                    co[it].append(jt)
    return co


def main():
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    log(f"[stage1-hyp] config: TAG={TAG} E_DIM={E_DIM} POINCARE_C={POINCARE_C} "
        f"ENCODER={ENCODER_MODEL} TRAIN={TRAIN} BATCH_SIZE={BATCH_SIZE} SEED={SEED} device={device}")

    if TRAIN and not ITEM_INTER_JSON:
        raise ValueError("TRAIN=1 requires ITEM_INTER_JSON env (co-occurrence 正样本源)")

    items = load_items(ITEM_JSON)
    item_ids = [it[0] for it in items]
    item_ids_set = set(item_ids)
    texts = [it[1] for it in items]
    log(f"[stage1-hyp] loaded {len(items)} items from {ITEM_JSON}")

    model = HyperbolicEncoder(ENCODER_MODEL, E_DIM, POINCARE_C).to(device)
    log(f"[stage1-hyp] t5 dim={model.encoder.config.d_model} → e_dim={E_DIM} → Poincaré(c={POINCARE_C})")

    if TRAIN:
        co = build_co_inter(ITEM_INTER_JSON, item_ids_set)
        log(f"[stage1-hyp] co-inter: {len(co)}/{len(items)} items have co-occur")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TRAIN_LR)
        item_id_to_idx = {it: i for i, it in enumerate(item_ids)}
        co_pairs = [(item_id_to_idx[a], item_id_to_idx[b])
                   for a, bs in co.items() if a in item_id_to_idx
                   for b in bs if b in item_id_to_idx]
        if not co_pairs:
            raise ValueError(f"no co-occurrence pairs found in {ITEM_INTER_JSON}")
        log(f"[stage1-hyp] InfoNCE training: {len(co_pairs)} pairs, epochs={TRAIN_EPOCHS}, "
            f"lr={TRAIN_LR} tau={TRAIN_TAU} neg_n={TRAIN_NEG_N}")
        rng = np.random.default_rng(SEED)
        for epoch in range(TRAIN_EPOCHS):
            t0 = time.time()
            order = rng.permutation(len(co_pairs))
            total_loss = 0.0
            n = 0
            for start in range(0, len(order), BATCH_SIZE):
                batch_idx = order[start:start + BATCH_SIZE]
                a_idx = [co_pairs[i][0] for i in batch_idx]
                b_idx = [co_pairs[i][1] for i in batch_idx]
                neg_idx = rng.integers(0, len(items), size=(len(a_idx), TRAIN_NEG_N))
                a_texts = [texts[i] for i in a_idx]
                b_texts = [texts[i] for i in b_idx]
                neg_texts_flat = [texts[i] for row in neg_idx for i in row]
                with torch.no_grad():
                    eu_a = model.encode_text(a_texts, device)
                    eu_b = model.encode_text(b_texts, device)
                    eu_neg = model.encode_text(neg_texts_flat, device)
                q = model.project_to_poincare(eu_a)
                p = model.project_to_poincare(eu_b)
                neg = model.project_to_poincare(eu_neg).reshape(len(a_idx), TRAIN_NEG_N, E_DIM)
                loss = info_nce_loss(q, p, neg, POINCARE_C, TRAIN_TAU)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(a_idx)
                n += len(a_idx)
            log(f"[stage1-hyp] train epoch {epoch+1}/{TRAIN_EPOCHS} loss={total_loss/n:.4f} ({time.time()-t0:.0f}s)")

    log(f"[stage1-hyp] encoding {len(items)} items (batch={BATCH_SIZE})...")
    all_embs = np.zeros((len(items), E_DIM), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(items), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        with torch.no_grad():
            emb = model(batch_texts, device)
        all_embs[start:start + len(batch_texts)] = emb.detach().cpu().numpy()
    log(f"[stage1-hyp] encode done in {time.time()-t0:.0f}s")

    norms = np.linalg.norm(all_embs, axis=1)
    log(f"[stage1-hyp] Poincaré stats: max_norm={norms.max():.4f} mean_norm={norms.mean():.4f} "
        f"(radius=1/√c={model.radius:.4f}, RADIAL_SCALE={RADIAL_SCALE})")
    if norms.max() >= model.radius:
        raise ValueError(f"embedding max_norm={norms.max():.4f} >= radius {model.radius:.4f}, "
                         f"expmap0/proj_to_ball broken")
    if norms.mean() < 0.3:
        raise ValueError(f"embedding mean_norm={norms.mean():.4f} < 0.3 (塌缩阈值), "
                         f"RADIAL_SCALE={RADIAL_SCALE} 不够大, stage2 内部 ρ 会塌缩到 0")

    df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": [row.tolist() for row in all_embs],
    })
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log(f"[stage1-hyp] saved {OUTPUT_PARQUET}")

    sha = hashlib.sha256()
    with open(OUTPUT_PARQUET, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    verdict = {
        "tag": TAG,
        "item_json": ITEM_JSON,
        "output_parquet": str(OUTPUT_PARQUET),
        "output_sha256": sha.hexdigest(),
        "n_items": len(items),
        "e_dim": E_DIM,
        "poincare_c": POINCARE_C,
        "radial_scale": RADIAL_SCALE,
        "encoder_model": ENCODER_MODEL,
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "radius": float(model.radius),
        "train_mode": TRAIN,
        "config": {
            "batch_size": BATCH_SIZE, "max_seq_len": MAX_SEQ_LEN, "seed": SEED,
            "train_epochs": TRAIN_EPOCHS, "train_lr": TRAIN_LR,
            "train_tau": TRAIN_TAU, "train_neg_n": TRAIN_NEG_N,
        },
        "done_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    verdict_path = OUTPUT_PARQUET.parent / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    log(f"[stage1-hyp] verdict: {verdict_path}")
    log("DONE")


if __name__ == "__main__":
    main()