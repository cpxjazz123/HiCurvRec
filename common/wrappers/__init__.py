"""共享 wrapper 模块 (taskA 方向A / taskB 方向B 各架构的 HG_Rec wrapper).

每个模块是薄转发: exec 归档在对应 stage/_archive/ 的原始训练脚本, 导出 wrapper 类,
避免复制大段类定义 (R2: 无 fallback, 类必须能从归档脚本加载).
"""
