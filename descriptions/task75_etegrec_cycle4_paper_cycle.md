# Task #75 — ETEGRec 128d cycle=4 (paper Section 4.1.5 备选 cycle 验证)

> **任务目的**: Paper Section 4.1.5 明确 cycle ∈ {2, 4}. Task #74 验证 cycle=2, Task #75 验证 cycle=4. 同样 paper-cycle 配置 (warmup=0, warm_epoch=1, early_stop=30).
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #74 (paper-cycle cycle=2): epoch 1 val_R@10=0.01494, epoch 3=0.01880. KL loss 持续下降 (81→48) 健康. 但与 Task #73 paper_exact (cycle=2 + warmup=8000 + warm_epoch=10) early epoch 0.0226 vs 0.0149 (-34%) 偏弱.

**假设 R5**: cycle=4 让每个 cycle 内 ID/REC 切换更频繁, 可能让 REC 学到更稳定的生成模式.

## 2. 实验设计

**变量**: cycle 2 → **4** (paper Section 4.1.5 备选值)
**保持不变**: paper-cycle 其他配置 (warmup=0, warm_epoch=1, early_stop=30), 128d SASRec-only, code_length=4, lr_rec=5e-3, lr_id=1e-4, batch_size=128+grad_accum=4.

**启动命令**:
```bash
# 用独立 yaml 副本避免覆盖 Task #74
cp /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments.yaml \
   /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments_cycle4.yaml
# 修改 cycle=4
sed -i 's/^cycle: 2$/cycle: 4/' /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec/config/musical_instruments_cycle4.yaml

cd /home/wlia0047/ar57/wenyu/GeneRec/ETEGRec
export CUDA_VISIBLE_DEVICES=1
python3 main.py \
    --config ./config/musical_instruments_cycle4.yaml \
    --cycle=4 \
    --warmup_steps=0 \
    --warm_epoch=1 \
    --early_stop=30 \
    --lr_rec=0.005 --lr_id=0.0001 \
    --batch_size=128 --gradient_accumulation_steps=4
```

## 3. 决策阈值

| R@10 | 决策 |
|------|------|
| ≥ 0.0624 | ✅ paper-EXACT cycle=4 突破 |
| 0.03-0.0624 | 🟡 cycle=4 vs cycle=2 优势确认 |
| 0.0253-0.03 | 🟡 cycle=2 vs cycle=4 等价 |
| < 0.0253 | ⚠️ cycle=4 退化, 改回 cycle=2 |

## 4. 预算

~4-6h (cycle=4 总 epoch 数 = Task #74 × 2).

## 5. 风险

- cycle=4 让 val_R@10 V-shape 更频繁 → early_stop=30 patience 容忍
- 总训练时间翻倍, 需确保 4-6h 内收敛