# Task #34 — phonism/genrec TIGER 200 epochs × seed=7 (run #3/4)

> **任务目的**: 与 Task #32/#33 并行跑，验证 phonism 200 epochs 上限在多 seed 下稳定
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

> 同系列任务: Task #32 (seed=42), Task #33 (seed=123), Task #35 (seed=2024)
> GPU 分配: cuda:2

---

## 1-2. 背景 + 实验设计

见 Task #32。**变量**: seed = 7。

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/genrec
PYTHONPATH="/home/wlia0047/ar57/wenyu/genrec:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=2 \
python -m genrec.trainers.tiger_trainer \
    --config config/tiger/amazon/tiger.gin \
    --split toys \
    --gin "train.epochs=200" \
    --gin "train.wandb_logging=False" \
    --gin "train.eval_test_every_epoch=2" \
    --gin "train.save_dir_root='out/tiger/amazon/toys/seed7/'" \
    --gin "train.pretrained_rqvae_path='out/tiger/amazon/toys/rqvae/checkpoint_19999.pt'" \
    --gin "train.seed=7"
```

## 3. 决策触发

见 Task #32 §3。

## 4. 预算

~5-6h（4 seed 并行）。

## 5. 完成度跟踪

- [ ] 启动 (cuda:2)
- [ ] 训练完成 + test eval
- [ ] verdict 写入 `verdicts/task34_phonism_seed7_result.md`
