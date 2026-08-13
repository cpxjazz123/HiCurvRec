"""Stage 1 双曲 encoder — frozen sentence-t5-base → MLP backbone (方向) + radius head (半径) [BASELINE].

设计: HG-Rec baseline Stage 1 — R_MODE="fixed" 退化为 v4 行为 (全部商品同半径, 无 per-item 半径).
- 输出**欧氏基础向量 v_i = r_i × d_i** (norm = r_i < 1, 严格在欧氏单位球内)
  - 方向 d_i = F.normalize(MLP(t5_emb)) (unit 向量, 语义方向)
  - 半径 r_i = R_MAX (固定, R_MODE=fixed, 全部商品同半径)
- stage2 v15 capmatch (per-layer learnable κ_l) 接收 baseline Stage1 输出
- baseline Stage1 输出 SHA = 1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc
  → v15 capmatch Stage2 训练时 item_emb_sha256 也是 1a42341f (R10 一致性)

环境变量:
  ITEM_JSON         必填 — {dataset}.item.json 路径
  OUTPUT_PARQUET    必填 — 输出 parquet 路径 (ItemID + embedding=list<float> 768 维欧氏基础向量)
  TAG               默认 "baseline"
  E_DIM             默认 768 — 欧氏基础向量维度 (与 stage2 e_dim 对齐)
  R_MAX             默认 0.99 — 半径上限 (<1 给 expmap0 留 arctanh 余量)
  R_MODE            heuristic / fixed (默认 heuristic)
                      heuristic: r_i = R_MAX × sigmoid(3 × (||t5_emb|| - 0.7)) 由 t5 norm 启发
                      fixed:     r_i = R_MAX (退化为 v4 固定半径行为)
  ENCODER_MODEL     默认 sentence-transformers/sentence-t5-base — frozen text encoder
  DEVICE            默认 cuda:0
  BATCH_SIZE        默认 64
  MAX_SEQ_LEN       默认 64
  SEED              默认 42

产物:
  OUTPUT_PARQUET    ItemID + embedding (parquet, 128 维欧氏基础向量 v_i = r_i × d_i)
  verdict.json      sha256 + config + v/r 维度/半径/方向统计 + 半径 histogram
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue157_intrinsic_curvature_residual/control/_lib")  # R44 baseline 自包含: HG_Rec/dataset/dataloader/utils/fsq_quantizer/hrqvae + hyperbolic_attention_bias + poincare_*
from utils import expmap0, proj_to_ball  # noqa: E402

# === R30: 所有配置硬编码 (无 os.environ.get; 变体复制脚本改常量) ===
ITEM_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments.item.json"  # R44 baseline 自包含: 引用共享 dataset/ 顶层 (md5 一致 HG-Rec/dataset/Instruments/)
# baseline Stage1 输出: parquet SHA = 1a42341f01537d6db5f622e4831e0fc1a2039dc8b1288291dcc4963deee000cc
# 与 v15 capmatch Stage2 训练时 item_emb_sha256 (1a42341f...) 一致 → R10 一致性
OUTPUT_PARQUET = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue157_intrinsic_curvature_residual/control/stage1/item_emb.parquet")  # R44 baseline 自包含: 产物写本任务 stage1/
TAG = "baseline"
E_DIM = 768  # Issue #71 v82 (2026-08-07): 必须与 Stage2 EMB_DIM 对齐 (768), 不然 np.load shape mismatch
# R_MAX 必须 < 1 — 给 expmap0(c=κ) 留 arctanh(r) 余量, 防 arctanh(1)=inf 数值爆炸
R_MAX = 0.99  # Issue #71 v82 (2026-08-12): 回归历史 baseline (issue96 v74_repro 真实训练) R_MAX=0.99 + R_MODE=heuristic. 历史 v_norm 多样 (0.5-0.66) → Stage2 c 学到 c>1 健康值. 当前 R_MODE=fixed R_MAX=1.0 → v_norm 全 1.0 → Stage2 κ 触底 GATE 2 FAIL.
# baseline 模式: R_MODE=heuristic (sigmoid 缩放 t5 norm) — 回归历史 baseline R5 真实训练配置
R_MODE = "heuristic"  # Issue #71 v82 (2026-08-12): 改回 heuristic 修复 Stage2 GATE 2 FAIL. 历史 issue96 v74_repro 真实跑这个模式 (v_norm 0.5-0.66).
# Issue #71 v82 (2026-08-07): radius 对 t5 norm 的敏感度 (越大越锐利过渡)
SIGMOID_TEMP = 3.0
SIGMOID_CENTER = 0.7  # sigmoid 中心点 (||t5_emb|| ≈ 0.7 时 r = R_MAX/2)
ENCODER_MODEL = "sentence-transformers/sentence-t5-base"
DEVICE = "cuda:0"  # GPU 由 CUDA_VISIBLE_DEVICES 决定 (R30 例外: torch 标准接口)
BATCH_SIZE = 64
MAX_SEQ_LEN = 64
SEED = 42

if not (0.0 < R_MAX <= 1.0):
    raise ValueError(f"R_MAX must be in (0, 1], got {R_MAX}")  # v15+v74 baseline 允许 R_MAX=1.0 (sentence-t5 mean-pool F.normalize norm=1.0)
if R_MODE not in ("heuristic", "fixed"):
    raise ValueError(f"R_MODE must be heuristic or fixed, got {R_MODE}")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}", flush=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_items(path):
    """{itemID: {title, description, brand, categories}} → [(itemID, semantics_text), ...]."""
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
    items.sort(key=lambda x: int(x[0]))
    return items


class HyperbolicEncoder(nn.Module):
    """frozen sentence-t5-base → MLP backbone (方向) + radius head (半径) → v = r × d.

    方向 d_i: unit 向量 (语义方向)
    半径 r_i: per-item scalar ∈ (0, R_MAX) (径向位置)
    基础向量 v_i = r_i × d_i: norm=r_i<1 严格在欧氏单位球内
    """

    def __init__(self, encoder_model, e_dim, r_max, r_mode, sigmoid_temp=3.0, sigmoid_center=0.7):
        super().__init__()
        from transformers import T5EncoderModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = T5EncoderModel.from_pretrained(encoder_model)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        d_model = self.encoder.config.d_model
        # MLP backbone: 768 → 256 → e_dim, 输出方向 d (unit)
        self.backbone = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, e_dim),
        )
        self.e_dim = e_dim
        self.r_max = r_max
        self.r_mode = r_mode
        # Issue #71 v82 (2026-08-07): radius 对 t5 norm 的敏感度 (越大越锐利)
        self.sigmoid_temp = float(sigmoid_temp)
        self.sigmoid_center = float(sigmoid_center)

    @torch.no_grad()
    def encode_text(self, texts, device):
        """frozen t5 → (B, d_model). mean-pool + attention mask."""
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        ).to(device)
        out = self.encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        h = out.last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom  # (B, d_model)

    def predict_radius(self, eu):
        """(B, d_model) → (B,) ∈ (0, R_MAX).

        heuristic: r_i = R_MAX × sigmoid(3 × (||t5_emb|| - 0.7)) (无训练, t5 norm 启发)
            标定: norm=0.3 → r≈R_MAX×0.18, norm=0.7 → r=R_MAX×0.5, norm=1.5 → r≈R_MAX×0.95
            t5 mean-pool norm 通常 0.3-1.5, 不同商品有差异 → 不同半径
        fixed: 全 R_MAX (退化为 v4 行为, 全部商品同半径)
        """
        if self.r_mode == "fixed":
            # v15+v74 baseline: 跳过 radius 调制, 直接返回 ||eu|| (≈1.0, sentence-t5 mean-pool norm 自然 ≈1.0)
            # v15 capmatch Stage 2 训练时实测输入 norm=1.0000, std=0.0002, unique=11/9922
            return eu.norm(dim=-1)  # (B,) ∈ (0, R_MAX)
        # heuristic (Issue #71 v82: sigmoid_temp / sigmoid_center 从 self 读)
        norms = eu.norm(dim=-1)  # (B,)
        return self.r_max * torch.sigmoid(self.sigmoid_temp * (norms - self.sigmoid_center))  # (B,) ∈ (0, R_MAX)

    def forward(self, texts, device):
        """texts → (v, r, d).
        v: (B, e_dim) 欧氏基础向量, norm=r (heuristic: r<1, fixed: r=1)
        r: (B,) 半径 ∈ (0, R_MAX)
        d: (B, e_dim) 单位方向
        """
        eu = self.encode_text(texts, device)  # (B, d_model) no_grad
        # Issue #71 v82 (2026-08-12): 回归历史 baseline R_MODE=heuristic 行为
        # 历史 issue96 v74_repro 真实训练: r = R_MAX × sigmoid(3 × (||eu|| - 0.7)), v = r × d
        # 当前 v_norm 多样 (0.5-0.66) → Stage2 c 学到 c>1 健康值 (vs R_MODE=fixed 全 1.0 → 触底 FAIL)
        d = F.normalize(eu, p=2, dim=-1)  # (B, d_model) unit direction
        r = self.predict_radius(eu)  # (B,) heuristic: R_MAX × sigmoid(...); fixed: ||eu||
        v = r.unsqueeze(-1) * d  # (B, d_model) v norm = r < R_MAX
        return v, r, d


def main():
    # Issue #71 v82 (2026-08-07): argparse 默认值硬编码 (R30 允许), 兼容旧实验 + v82 命令行覆盖
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius_max", type=float, default=R_MAX, help="per-item radius 上限 (<1, v82: 0.99→0.95)")
    ap.add_argument("--sigmoid_temp", type=float, default=SIGMOID_TEMP, help="heuristic sigmoid 温度 (v82: 3.0→5.0)")
    ap.add_argument("--sigmoid_center", type=float, default=SIGMOID_CENTER, help="heuristic sigmoid 中心")
    ap.add_argument("--tag_suffix", type=str, default="", help="输出路径 tag 后缀 (v82 用 _v82_r095_t5)")
    # Issue #105 (2026-08-10): SHIE — Stage1 Hyperbolic Item Encoding
    ap.add_argument("--shie_encoding", action="store_true", default=False,
                    help="Issue #105: 启用 SHIE exp_map_0 Stage1 encoding (输出 hyperbolic 而非 Euclidean)")
    ap.add_argument("--shie_c", type=float, default=1.0, help="Issue #105: SHIE exp_map_0 曲率 c (默认 1.0)")
    args = ap.parse_args()

    global OUTPUT_PARQUET
    if args.tag_suffix:
        OUTPUT_PARQUET = OUTPUT_PARQUET.parent / f"item_emb_baseline_{args.tag_suffix}.parquet"
    # Issue #105: SHIE 启用时改 output tag 后缀, 避免与 baseline parquet 冲突
    if args.shie_encoding:
        OUTPUT_PARQUET = OUTPUT_PARQUET.parent / "item_emb_shie.parquet"

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    log(f"[stage1-hyp] config: TAG={TAG} E_DIM={E_DIM} R_MAX={args.radius_max} R_MODE={R_MODE} "
        f"sigmoid_temp={args.sigmoid_temp} sigmoid_center={args.sigmoid_center} "
        f"ENCODER={ENCODER_MODEL} SEED={SEED} device={device}")

    items = load_items(ITEM_JSON)
    item_ids = [it[0] for it in items]
    item_ids_set = set(item_ids)
    texts = [it[1] for it in items]
    log(f"[stage1-hyp] loaded {len(items)} items from {ITEM_JSON}")

    model = HyperbolicEncoder(ENCODER_MODEL, E_DIM, args.radius_max, R_MODE,
                              sigmoid_temp=args.sigmoid_temp,
                              sigmoid_center=args.sigmoid_center).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[stage1-hyp] t5 dim={model.encoder.config.d_model} → MLP backbone → e_dim={E_DIM} → "
        f"v = r × d (||v||=r<{args.radius_max}), backbone params={n_params:,}")
    log(f"[stage1-hyp] stage2 expmap0(c_l) 严格在 Poincaré ball 内 (||v||<1 ⟹ norm<radius)")

    log(f"[stage1-hyp] encoding {len(items)} items (batch={BATCH_SIZE})...")
    all_v = np.zeros((len(items), E_DIM), dtype=np.float32)
    all_r = np.zeros((len(items),), dtype=np.float32)
    t0 = time.time()
    model.eval()
    # Issue #105 (2026-08-10): SHIE — exp_map_0 应用于 Stage1 输出 (Euclidean → hyperbolic)
    # 仅在 --shie_encoding 启用时执行; 默认保持 v15 Euclidean 输出 (||v||=1.0).
    if args.shie_encoding:
        log(f"[Issue105] SHIE encoding: exp_map_0 applied (c={args.shie_c})")
    for start in range(0, len(items), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        with torch.no_grad():
            v, r, d = model(batch_texts, device)
        # Issue #105: SHIE — apply exp_map_0 to v (Euclidean → hyperbolic ball)
        if args.shie_encoding:
            v = expmap0(v, args.shie_c)  # v ∈ Poincaré ball, ||v||_ball = tanh(√c · ||v||)/√c
        all_v[start:start + len(batch_texts)] = v.detach().cpu().numpy()
        all_r[start:start + len(batch_texts)] = r.detach().cpu().numpy()
    log(f"[stage1-hyp] encode done in {time.time()-t0:.0f}s")

    # sanity check
    norms = np.linalg.norm(all_v, axis=1)
    log(f"[stage1-hyp] 欧氏基础向量 v stats: max_norm={norms.max():.4f} mean_norm={norms.mean():.4f} "
        f"std_norm={norms.std():.4f} (R_MAX={R_MAX})")
    log(f"[stage1-hyp] 半径 r stats: min={all_r.min():.4f} max={all_r.max():.4f} "
        f"mean={all_r.mean():.4f} std={all_r.std():.4f}")
    if norms.max() > 1.0 + 1e-6:
        raise ValueError(f"v max_norm={norms.max():.4f} > 1.0, R_MAX={R_MAX} 过大")  # v15+v74 baseline 允许 norm=1.0
    if all_r.max() >= R_MAX + 1e-6:
        raise ValueError(f"r max={all_r.max():.4f} > R_MAX={R_MAX}")
    # 半径多样性检查
    unique_r = len(np.unique(np.round(all_r, 4)))
    log(f"[stage1-hyp] 半径唯一值数={unique_r}/{len(items)} (R_MODE={R_MODE})")
    if R_MODE != "fixed" and unique_r < len(items) * 0.5:
        log(f"[stage1-hyp] WARN: 半径多样性低, 考虑换 R_MODE")

    df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": [row.tolist() for row in all_v],
    })
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log(f"[stage1-hyp] saved {OUTPUT_PARQUET}")

    sha = hashlib.sha256()
    with open(OUTPUT_PARQUET, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    r_hist, r_bin_edges = np.histogram(all_r, bins=10, range=(0, R_MAX))
    verdict = {
        "tag": TAG,
        "item_json": ITEM_JSON,
        "output_parquet": str(OUTPUT_PARQUET),
        "output_sha256": sha.hexdigest(),
        "n_items": len(items),
        "e_dim": E_DIM,
        "r_max": R_MAX,
        "r_mode": R_MODE,
        "encoder_model": ENCODER_MODEL,
        "v_stats": {
            "max_norm": float(norms.max()),
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "median_norm": float(np.median(norms)),
        },
        "r_stats": {
            "min": float(all_r.min()),
            "max": float(all_r.max()),
            "mean": float(all_r.mean()),
            "std": float(all_r.std()),
            "median": float(np.median(all_r)),
            "unique_buckets": int(unique_r),
            "histogram": [int(c) for c in r_hist],
            "bin_edges": [float(e) for e in r_bin_edges],
        },
        "config": {
            "batch_size": BATCH_SIZE, "max_seq_len": MAX_SEQ_LEN, "seed": SEED,
        },
        "design_notes": {
            "philosophy": "v = r × d, ||v||=r<1, stage2 用每层 κ_l 做 expmap0 投到 Poincaré ball",
            "advantages": [
                "不同商品可有不同半径 (per-item 径向位置, t5 norm 启发)",
                "三层使用不同曲率 (stage2 现有 per-layer κ_l 自动生效)",
                "曲率变化后自动重新映射 (stage2 forward 每步重算 expmap0, 无需重训 stage1)",
                "不易超 Poincaré ball 边界 (||v||<1 ⟹ expmap0 输出 norm<radius)",
            ],
            "stage2_required_changes": "none (KappaAwareVectorQuantization 已实现 per-layer expmap0)",
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