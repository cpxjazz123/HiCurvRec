"""Issue #97 Stage 1: 复现 TIGER (Rajput et al., NeurIPS 2023) 在 Musical_Instruments 上的 R@10。

TIGER = RQ-VAE 语义 ID tokenization + T5 自回归生成 (beam search 解码)。
本复现复用 LETTER 官方仓库 (Wang et al., CIKM 2024) 内嵌的 TIGER 实现 + RQ-VAE tokenizer。

流程:
1. RQ-VAE 语义 ID tokenization (Instruments_llama7b_768 / instruments_t5base_v4)
2. TIGER 解码器训练 (T5-base)
3. beam=20 评估 (recall@10 口径, 与 HG-Rec 论文 Table 1 一致)
"""
import os
import subprocess
import sys

REPO = "/fs04/ar57/wenyu/LETTER/LETTER-TIGER"
RQVAE_CKPT = "/fs04/ar57/wenyu/LETTER/RQ-VAE/ckpt/Instruments_llama7b_768"
INDEX = "/fs04/ar57/wenyu/LETTER/data/Instruments/Instruments.index.json"


def main():
    assert os.path.isdir(REPO), f"LETTER repo missing: {REPO}"
    assert os.path.isdir(RQVAE_CKPT), f"RQ-VAE ckpt missing: {RQVAE_CKPT}"
    assert os.path.isfile(INDEX), f"index missing: {INDEX}"
    print("[Stage1 TIGER] precheck PASS: repo + RQ-VAE ckpt + index 齐全")
    print(f"  RQ-VAE ckpt: {RQVAE_CKPT}")
    print(f"  index: {INDEX}")


if __name__ == "__main__":
    main()
