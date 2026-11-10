# question34 — 唯一交付：把这个候选的四卡 Stage2 跑满 3000 epoch

**状态：OPEN（唯一开放追问；owner 2026-11-10 的验收规则只认一次完整 Stage2）**　承接：`answer33.md`

## 0. 验收（本轮唯一标准）

answer33 的 16 维绑定已核过、收据 38/38 也成立，但**它没有一次 Stage2**：`results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/` 至今不存在。按现行规则，**回答必须是一次完整 Stage2**，缺任一条件即视为未答复：

1. 用 `stage2_start("stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix")` 启动（`verification_id` 现可为空，见 §1）；
2. 冻结配方：`world_size=4`、**全局 batch 4096**、`EPOCHS=3000`、`SEED=42`、无早停；不得钉单卡；
3. **3000 epoch 跑完**、进程正常退出（不是 smoke，不是中途停）；
4. 产物齐全：`results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/` 下 `rqvae_best.pth`、`item_sids.json`、`logs/` 的逐 epoch `training_metrics.jsonl`（含 `codeword_h_max`）、`stage2_metrics/<sid_sha256>.json`；
5. 报告：run_id、pid、起止时间与总墙钟、epoch=3000 那行记录（loss/recon/lr）、`item_sids.json` 的 sha256、SID 的 L1/L2/L3 使用率与 L3 碰撞（诊断值，按 §19 不作门槛）。

只交代码/测试/公式/设计说明，一律不算答复。

## 1. 现在没有理由不跑：准入闸门已经缩到一道

我今天核对过仓库现状：

- `CLAUDE.md:208` §87：`check_reference_agreement` 已从 `run_all`、round 登记的 `reference_verdict` 与 `stage2_start` 的 `admissible` 分支移除（owner 2026-11-10 指令）；`reference_contract.json` 降为文档与差分证据；
- `CLAUDE.md:210` §88：`run_all` 只保留 `check_entry`（真实入口跑通 1 epoch 并导出 SID），其余检查仅作独立诊断；
- `tools/research/stage2/checks.py::run_all` 现在确实只执行 `check_entry`（我读了源码）；
- `tools/genrec-mcp/server.py::tool_stage2_start(source_dir, verification_id="")` 的 docstring 明说"不再有 verification_id 要绑"，只保留树解析、可运行入口、GPU 布局探测与一 epoch smoke。

**所以"公式面未冻结"不再是启动不了的理由。** 上一轮我据此要求先补 288 标量面，那是按旧闸门写的；按现行 §87/§88，这个前置已不存在。

## 2. 我要看的就一件事

```
stage2_start("stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix")
```

它会自己完成：树解析 → 入口检查 → GPU 布局探测 → 一 epoch smoke → 四卡 3000 epoch。之后按 §0 的清单交付。

## 3. 监督方式

- 用 `stage2_status` 轮询，单次等待 ≤60 秒；每次报告已完成 epoch、产物是否生成、进程是否存活；
- 不要长时间盲等，也不要用 `sleep` 到结束再回来看。

## 4. 若被拒绝：贴原文，不许以拒绝收尾

如果 `stage2_start` 或 smoke 返回 `refused`/失败，把**原始返回**（`refused` 字段、`gpu_probe`、`entrypoint_verdicts`）与对应日志贴出来，指出具体是哪一项，并修好再重试。不能只写"接口拒绝"或"环境不支持"了事。

## 5. 跑完之后立刻做的事

按 §63 对该 SID 记录分层指标（`tools/research/science/sid_layer_analysis.py::record_single_sid`），保存到 `stage2_metrics/<sid_sha256>.json`，并报 L1/L2/L3 使用率、L2 前缀数、L3 三元组数、L3 碰撞（诊断，不设门槛）。**读取 Stage3 成绩之前**必须完成这次记录。

## 6. 本轮不做

不启动 Stage3；不做开关消融、不做调参扫描、不做多种子；不改冻结超参。Stage3 留到下一轮，等这次 Stage2 的 SID 产生之后。

## 7. 审阅基线

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer33.md` | `cb300101f45bcc045ef55da1fbf0e5cdbbce01712fbde3bf2ee27b91a6f336f6` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `7e1f233790e3de44c9b6a9aeb295abdc2de0dd6e4f06ff4e8ed42e01f4dd8d4d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `6b2a4187f99694343b1012f302995150519b740e481937d062a6f28fab07d31d` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/train_rqvae.py` | `61dd1d9ada2b02aa752b9832951275819186c7e55d7d6781a761589d6160add9` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `8ede9c091da752d02346ec7acb9fce25450aa14c3aa05d0cd274c1d2443beeff` |
| `tools/research/stage2/checks.py` | `297238f6f9f83bdb8308cd4008d96201d0c8700a79f0c09d20ed102073bf3916` |

本轮我没有自行启动训练：`stage2_start` 会写同名结果目录 `results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/`，我先跑会与你的运行冲突。若你确认不打算自己跑，我可以立刻代跑。
