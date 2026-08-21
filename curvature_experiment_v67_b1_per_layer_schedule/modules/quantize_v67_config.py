"""v67 B1 Per-Layer Schedule 配置.

v67 设计: Stage 1 RQ-VAE 端 B1 Time-varying Per-Layer Schedule
- 3 层用 3 种不同 schedule, per-layer c_l(t) 动态变化 (R36b 满足)
- L0: cosine cyclic [0.05, 0.7]  (v65 D1 同款, 周期循环)
- L1: linear warmup [0.5, 1.0]  (单调增, 训练全程动态)
- L2: step decay [1.0 → 0.7]   (单调减, 在 step=curriculum_steps/2 时切换)
- 每层 schedule 不同 → 3 个独立曲率动态, 引入层间异质性
- 不引入可学习参数 (R36 范围内)
- 扩展 v65 D1 单标量 c cosine cyclic, 增加 B1 per-layer 维度
"""
V67_B1_ENABLED = True
V67_B1_C_START_PER_LAYER = (0.05, 0.5, 1.0)
V67_B1_C_END_PER_LAYER = (0.7, 1.0, 0.7)
V67_B1_SCHEDULE_PER_LAYER = ("cosine_cyclic", "linear_warmup", "step_decay")
# step_decay 在 step=curriculum_steps/2 时切换 (与 v65 cycle_period=curriculum_steps 对齐)
V67_B1_STEP_DECAY_HALF = 0.5