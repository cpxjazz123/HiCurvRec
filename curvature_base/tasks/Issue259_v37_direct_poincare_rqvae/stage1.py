"""v37 Stage 1 — Direct Poincaré RQ-VAE 训练 (DDP 4 卡, RQ-VAE 全链路 Poincaré 几何化).

启动方式:
  python3 stage1.py  (脚本内部 torchrun --nproc_per_node=4)

v37 创新点 (R36 框架变更, R18 4 维度对比):
- D1 spec: Stage 1 RQ-VAE 全链路强制 Poincaré ball 几何 (encoder 输出 / codebook / 距离)
- D2 实施:
  * RqVae.encode 末尾加 expmap0(c=1) 把 encoder output 投影到单位 Poincaré ball
  * codebook embedding 同样 expmap0(c=1) 强制在 ball 内
  * quantize.distance 改用 poincare_dist_sq (替换 Euclidean L2)
- D3 Gate 1 失败机制: expmap0 数值不稳 → NaN-Inf → 立即 R37 FAIL
- D4 文献: Bécigneul & Ganea 2019 (Riemannian Adaptive), Chami 2019 (HGCN)

vs v19 (Stage 1 几何变换, baseline 0.1113):
  v19 只在 loss 函数级加 c_end 缓和, encoder 仍输出 Euclidean → v37 直接 Poincaré 输出
vs v22 (Möbius 减法):
  v22 只在 encoder 层用 Möbius, codebook 仍 Euclidean → v37 codebook 也 Poincaré
vs v26 (per-item κ codebook):
  v26 codebook 几何化但 quantize distance 仍 Euclidean → v37 quantize 距离也双曲

输入:
  - 数据集: /home/wlia0047/ar57/wenyu/GeneRec/dataset/Instruments/

产物:
  - out/rqvae/v37_poincare/rqvae_final.pt

R37 决策点 (Stage 4 之后):
  test_R@10 ≥ 0.11130798969072164 (v19 baseline + R51 锁定) → R37 PASS
  test_R@10 < 0.1113 → R37 FAIL + R50 revert

R36: 新曲率机制, 不调 LR/dropout/wd/batch_size.
R42: torchrun --nproc_per_node=4 DDP 4 卡.
R30/R43: 超参硬编码, 无 CLI 数值超参.
"""
import os
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"


def main():
    print(f"[v37 stage1] launching Direct Poincaré RQ-VAE training (DDP 4 card)")
    print(f"[v37 stage1] TODO: 实施 _lib/poincare_quantizer.py + 改 train_rqvae_instruments.py")
    print(f"[v37 stage1] 关键路径:")
    print(f"  1. RqVae.encode 末尾加 expmap0(c=1)")
    print(f"  2. codebook 用 expmap0 投影")
    print(f"  3. quantize.distance 用 poincare_dist_sq")
    print(f"  4. Stage 2 KMeans 用 Poincaré 距离 (R41c: fixed random_state)")
    print(f"[v37 stage1] 预期产物: out/rqvae/v37_poincare/rqvae_final.pt")
    print(f"[v37 stage1] Stage 1 训练预计 ~3-4 小时 (DDP 4 卡)")


if __name__ == "__main__":
    main()