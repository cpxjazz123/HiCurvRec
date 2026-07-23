# Task #76 — ETEGRec 128d paper_exact 400 epoch 复跑 (baseline 重现)

> **任务目的**: 复跑 Task #73 paper_exact 配置 (cycle=2, warmup=8000, warm_epoch=10, early_stop=15, 400 epoch) 验证 baseline 可重现性 + 与 Task #74 paper-cycle 直接对比 R3 假设.
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #73 paper_exact 已完成 (R@10=0.0253). Task #74 paper-cycle 是变量修改版本. Task #76 复跑**完全相同** paper_exact 配置作为 R3 假设验证的对照组.

**对比矩阵**:
| 配置 | Task #73 (✅完成) | Task #74 (🟢 在跑) | Task #76 (🆕 本任务) |
|------|------------------|-------------------|---------------------|
| warmup_steps | 8000 | 0 | **8000** |
| warm_epoch | 10 | 1 | **10** |
| early_stop | 15 | 30 | **15** |
| cycle | 2 | 2 | **2** |

**假设 R6**: Task #73 paper_exact R@10=0.0253 可重现, Task #74 paper-cycle 是真实"改善" vs "退化" 通过直接对比可定论.

## 2. 实验设计

**完全复制 Task #73 paper_exact**: cycle=2, warmup=8000, warm_epoch=10, early_stop=15, epochs=400, 128d SASRec-only, code_length=4, lr_rec=5e-3, lr_id=1e-4, batch_size=128+grad_accum=4.

**启动命令**:
```bash
# 独立 yaml 副本避免覆盖 Task #74 当前 yaml
cp /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments.yaml \
   /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments_paper_exact.yaml
# 改回 paper_exact (warmup=8000, warm_epoch=10, early_stop=15)
sed -i 's/^warmup_steps: 0$/warmup_steps: 8000/' \
    /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments_paper_exact.yaml
sed -i 's/^warm_epoch: 1$/warm_epoch: 10/' \
    /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments_paper_exact.yaml
sed -i 's/^early_stop: 30$/early_stop: 15/' \
    /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments_paper_exact.yaml

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec
export CUDA_VISIBLE_DEVICES=2
python3 main.py \
    --config ./config/musical_instruments_paper_exact.yaml \
    --lr_rec=0.005 --lr_id=0.0001 \
    --batch_size=128 --gradient_accumulation_steps=4 \
    --warmup_steps=8000 --warm_epoch=10 --early_stop=15 --cycle=2
```

## 3. 决策阈值

| R@10 | 决策 |
|------|------|
| 0.025-0.026 | ✅ 与 Task #73 (0.0253) 一致, 复现成功 |
| > 0.03 | ⭐ 比 Task #73 更好 (自然 variance) |
| < 0.020 | ⚠️ 与 Task #73 不一致, 调查 variance 来源 |

## 4. 预算

~6-8h (400 epoch 上限).

## 5. 风险

- 与 Task #73 paper_exact 高度重复, ROI 中等 (但 reproducibility 是科学基础)
- 若 GPU 0/1/2 资源竞争, 降低优先级