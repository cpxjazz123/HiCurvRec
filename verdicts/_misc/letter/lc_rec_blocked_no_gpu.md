# LETTER-LC-Rec 路径 R18 论证 + GPU 阻塞 NO-GO

## 背景

LETTER paper (CIKM 2024) 提供 **2 个 instantiation**:
1. **LETTER-TIGER**: t5-base + LETTER semantic token aggregation (已完成 5 轮 NO-GO)
2. **LETTER-LC-Rec**: LLaMA-2 7B + LoRA + collaborative PEFT (完全未尝试)

按 R18 4 维度对比,LC-Rec 与 TIGER **D1 spec / D2 实施核心 / D3 Gate 1 / D4 引用** 全不同:

| 维度 | TIGER | LC-Rec |
|---|---|---|
| D1 spec (paper instantiation) | "TIGER-style sequential rec with semantic tokens" | "LC-Rec-style with LoRA PEFT" |
| D2 实施核心 | t5-base + Sinkhorn-Knopp RQ-VAE + TIGER arch | LLaMA-2 7B + LoRA + standard generation |
| D3 Gate 1 失败机制 | T5 训练饱和 @ 25 epoch, beam=50 反而变差 | 未验证 (LoRA 训练动态与全参不同) |
| D4 引用文献 | LETTER paper Section 4.1 | LETTER paper Section 4.2 (paper 独立章节) |

按 R18 兜底:**"任何不同 → 必须做实验, 不能凭路径同构 NO-GO"**。
但 LC-Rec 实施需要 LLaMA-2 7B (13GB) + LoRA fine-tune + evaluation。

## 当前 GPU 阻塞

`nvidia-smi` 实时状态 (2026-08-08):
- GPU 0 (唯一可用): util 100% / mem 6.6GB / 9 个 genrec_env python 进程占用
- 估计是其他 agent 的 DDP 训练任务

按 R7 (GPU 必须完全空闲) + R19 (禁止抢卡),无法启动 LC-Rec 训练。

## LC-Rec 实施依赖清单

需检查的环境依赖:
- LLaMA-2 7B safetensors (已下载 huggyllama/llama-7b 替代, 13GB)
- peft 库 (LC-Rec 用 `peft.LoraConfig`)
- bitsandbytes (8-bit 训练)
- accelerate (DDP)
- deepspeed (可选)
- flash_attn (LC-Rec 用 `replace_llama_attn_with_flash_attn` monkey patch)

这些是 LETTER paper 要求的 torch 1.13.1+cu117, 我们之前用 deepke env (torch 1.11+cu102) 不一定能跑。

## 决策

**Blocked NO-GO**:
- LC-Rec 路径 **理论上 R18 必须实验**, 但 GPU 阻塞无法启动
- 与 TIGER 5 轮 NO-GO 不同, LC-Rec 是真正的不同 instantiation
- 待 GPU 空闲 + 验证 deepke/comfyui_v2 env 是否能跑 LC-Rec 后再启动

## Why

- 满足 R18 论证义务 (D1-D4 全不同必须实验)
- 满足 R7 (GPU 空闲约束)
- 满足 R28 兜底 (用户硬约束 0.11 vs TIGER 0.06 上限, 需尝试 LC-Rec)
- 不能立即启动 → 落 "blocked" 状态 verdict 而非直接 NO-GO

## How to apply

- GPU 空闲 + env 验证后, 立即启动 LC-Rec 实验
- 优先级: LC-Rec > TIGER (LC-Rec 是 R18 未尝试维度)
- 评估 LC-Rec 是否能突破 TIGER 上限 0.06

## 后续工作

1. **环境验证**: 测试 deepke env 能否 import peft/bitsandbytes/flash_attn
2. **GPU 等待**: 监控 GPU 0 util < 10% 时启动 LC-Rec 训练
3. **RQ-VAE 复用**: LC-Rec 用同样 RQ-VAE 输出 codes, 可复用 TIGER 的 InRamaindex-sk4-sk.json
4. **LoRA 超参**: r=8 / alpha=16 / target_modules=q_proj,v_proj (LC-Rec 默认)
5. **Eval**: 用 LETTER-LC-Rec/evaluate.py 算 R@10, 对比 TIGER 0.0581

## 参考资源

- 入口: `/home/wlia0047/ar57/wenyu/LETTER/LETTER-LC-Rec/lora_finetune.py`
- 评估: `/home/wlia0047/ar57/wenyu/LETTER/LETTER-LC-Rec/evaluate.py`
- 索引: `/home/wlia0047/ar57/wenyu/LETTER/data/Instruments/Instruments.llamaindex-sk4-sk.json` (复用)
- RQ-VAE ckpt: `/home/wlia0047/ar57/wenyu/LETTER/RQ-VAE/ckpt/instruments_t5base_v4/Aug-07-2026_23-21-03/best_collision_model.pth` (复用)
- LLaMA-7B 替代: `/fs04/ar57/wenyu/LETTER/ckpt/LLaMA-7b/`