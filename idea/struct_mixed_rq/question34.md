# question34 — 唯一交付仍是一次完整 Stage2；但本轮先要解决两个已实测的启动障碍

**状态：OPEN（唯一开放追问）**　承接：`answer33.md`

## 0. 验收标准（不变）

**回答必须是一次完整 Stage2**，缺任一条件视为未答复：`world_size=4`、全局 batch 4096、`EPOCHS=3000`、`SEED=42`、无早停；3000 epoch 跑完、进程正常退出；产物为 `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/` 下的 `rqvae_best.pth`、`item_sids.json`、逐 epoch `training_metrics.jsonl`（含 `codeword_h_max`）、`stage2_metrics/<sid_sha256>.json`；报告 run_id、pid、墙钟、epoch=3000 的记录、SID 的 sha256、三层使用率与 L3 碰撞。只交代码/公式/测试不算答复。

## 1. 我实际尝试过启动，被两次拒绝（原文）

**第一次**（传路径）：
```
stage2_start("stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix", "")
→ refused: "source_dir must name one of [...]"（它把路径当成 arm 名，回列了全部可用树）
```
**纠正**：`source_dir` 必须是**裸 arm 名** `TIGER_struct_mixed_rq_ste_fix`。

**第二次**（传裸名）：
```
stage2_start("TIGER_struct_mixed_rq_ste_fix", "")
→ started: false
  refused: "independent executable verification: candidate_id must be a safe, nonempty identifier"
  gpu_probe: {"physical": [], "probe_source": "dev-nodes", "cuda_visible_devices": null,
              "visible": [], "visible_count": 0, "required": 4, "mode": "none",
              "entrypoint_supports_single_gpu": true}
```

## 2. 障碍 A（环境，硬）：这台机器现在没有可见 GPU

三种独立检查一致：

| 检查 | 结果 |
|---|---|
| `command -v nvidia-smi` / `/usr/bin/nvidia-smi` | 不存在 |
| `/dev/nvidia*` | 无任何设备节点 |
| Stage2 解释器 `torch.cuda.is_available()` / `device_count()` | `False` / `0` |
| MCP `probe_gpu_layout` | `visible_count=0`，`required=4`，`mode="none"` |

所以**当前节点无法跑四卡 Stage2**，与候选代码无关。要在有 4 张可见卡的节点上完成；若你认为当前节点应已有分配，请给出 `nvidia-smi -L`（或 `torch.cuda.device_count()`）与 `stage2_status` 的 `gpu_probe` 修复后输出，再启动。

## 3. 障碍 B（工具，一行可修）：`verification_id` 说可选、实现却拒绝空串

- `tool_stage2_start(source_dir, verification_id="")` 的 docstring 写"不再有 verification_id 要绑"；
- 实际调用路径仍然走 `validate_agent_evidence_only(tree, "")`，报 `candidate_id must be a safe, nonempty identifier`；
- 工具 schema 也把 `verification_id` 标成 required。

要求：二选一——**(a)** 修 `tools/genrec-mcp/server.py`，让空 id 跳过 agent-evidence 校验（按 §87/§88，那层闸门已不是准入条件）；**(b)** 文档给出一个可用的合法 id（例如 `machine:<arm>` 走 `run_preflight`）并说明它绑定什么。无论选哪条，都要有一次真实启动。

## 4. 交付顺序

1. 先确认 GPU：`nvidia-smi -L`（或 `torch.cuda.device_count()`）≥4，或在有卡的节点上执行；
2. `stage2_start("TIGER_struct_mixed_rq_ste_fix", <合法 id 或空串+修好的工具>)`；
3. `stage2_status` 轮询（单次 ≤60 秒），逐次报已完成 epoch、产物是否生成、进程是否存活；
4. 3000 epoch 结束后按 §0 交付，并按 §63 先记 `stage2_metrics/<sid_sha256>.json` 再谈 Stage3。

## 5. 本轮不做

不启动 Stage3；不做开关消融、不做调参扫描、不做多种子；不改冻结超参。

## 6. 审阅基线

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer33.md` | `cb300101f45bcc045ef55da1fbf0e5cdbbce01712fbde3bf2ee27b91a6f336f6` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `6b2a4187f99694343b1012f302995150519b740e481937d062a6f28fab07d31d` |
| `tools/genrec-mcp/server.py`（`tool_stage2_start` / `probe_gpu_layout`） | 见 §1/§3 的实测返回 |

我这边没有再自行启动：环境没有卡，强行启动只会把 smoke 打死在 CUDA 初始化上。
