## C11：一次诊断派发在 GPU 启动前被拒绝

失败命令 `python3 -m tools.issue252.launch_optimized_profile`，exit1：`Stage2 needs 4 visible cards for four-rank DDP and this host shows 0`。证据 `PROFILE_LAUNCH_FAILURE.md`。

根因不是已证实的硬件缺失：控制器错误地用了系统 `/apps/anaconda/2024.02-1/bin/python3`；实际依赖诊断 `CONTROLLER_INTERPRETER.json` 为 torch_available=false。`launcher._visible_cards` 在控制器 import torch 失败时返回0，尚未启动指定的子解释器。成功的原四卡 profile 使用的是仓库规定的 Stage2 Python。

纠正为固定 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue252.launch_optimized_profile`。不绕过四卡守卫、不改配方；这一拒绝没有产生 GPU 优化器步，更不是正式训练/多种子。新实测结果待回写。
