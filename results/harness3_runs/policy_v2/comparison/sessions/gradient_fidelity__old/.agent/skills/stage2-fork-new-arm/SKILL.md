---
name: stage2-fork-new-arm
description: Use when creating a new Stage2 experiment arm — copy stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE to TIGER_<new mechanism name>, rewire the copy's identity (arm name, guard, mechanism), then launch it through the genrec MCP. Paths need no edits because the tree derives them from its own directory name.
---

# 复制 TIGER 树造新机制臂

把一个 Stage2 臂做成新机制的标准流程：**复制 `stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE` → 改名 `TIGER_<新机制名>` → 改身份与机制 → 自检 → 经 MCP 启动**。原树保持只读。

**路径不需要改**：源树已把输入/输出/日志路径改成「由本文件所在目录名自动解析」（见 `train_rqvae.py` 的「路径自动解析」段），复制改名后 `_OUT` / `LOG_DIR` / `CHECKPOINT` 自动跟到新树，不会写回源树。复制后**唯一**要改的是「身份」和「机制」。

## 0. 机制名怎么取（每次都要新的）

- 形状：`^[a-z][a-z0-9_]*$`，snake_case，2–4 个词，描述**机制本身**而不是「第几次尝试」（`residual_norm_calibrated` ✔，`resample2`、`test1`、`new_arm` ✘）。
- 禁止包含 `__`：Stage3 的变体名是 `<树名>__<机制名>`，双下划线是分隔符。
- 查重三处：

```bash
ls -1 stage2_RQ-VAE/ && ls -1 results/stage2_RQ-VAE/ && git log --oneline --all | grep -i '<关键词>'
```

## 1. 复制

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE
cp -a TIGER_BEHAVIOR_SAMPLE_RQ-VAE TIGER_<名>
rm -rf TIGER_<名>/logs TIGER_<名>/__pycache__ TIGER_<名>/model/__pycache__ TIGER_<名>/scripts/__pycache__
find TIGER_<名> -name '*.pyc' -delete
```

`logs/` 与 `__pycache__` 必须清掉：带过去就等于把父树历史日志当成本臂证据。

## 2. 改身份与机制（只在副本里改）

1. **机制名**：`EXPERIMENT_ARM = "<名>"`，并在文件头加一行出处说明（由哪棵树复制、本树只跑哪个机制）。
2. **自锁守卫**：把守卫改成只允许本树自己的机制，其它名字导入即失败：
   ```python
   if EXPERIMENT_ARM != "<名>":
       raise RuntimeError(
           f"only the '<名>' mechanism may run in this tree; got {EXPERIMENT_ARM!r}"
       )
   ```
3. **机制本体**：实现放副本自己的模块（与 `anti_collapse_losses.py` / `geom_losses.py` 同级），在副本 `train_rqvae.py` 里替换调用点（当前是 `behavior_contrastive_loss(residuals, source_ids, target_ids, successor_indicator)`），并把被替换掉的旧机制函数删掉（别留两套机制）。
4. **metrics 里的 `mechanism="..."` 字符串**改成新机制名，否则证据链会说谎。
5. **脚本**：`scripts/grad_check*.py` 已从 `trainer._OUT` 取 ckpt（复制后自动正确），但它们**调用的是旧机制函数名**，随第 3 步一起改名。

自检没有遗漏：

```bash
grep -rn 'TIGER_BEHAVIOR_SAMPLE_RQ-VAE' TIGER_<名> --include='*.py'   # 只应剩出处注释
```

**机制写法约定**：新 loss 用**模块级公共函数**（不要 `_foo`）并在 trainer 里 import 进来——MCP 的单 epoch smoke 会包装树内模块的公共函数做逐项梯度探针，私有函数探不到。签名接受张量入参、返回标量张量，保持在原调用点的同一位置，别动数据管线与优化器契约。

**别急着做的清理**：历史臂表（`_ARM_BEHAVIOR_WEIGHT` / `_ARM_CROWDING2` / `_ARM_ANCHOR_WEIGHT` / `_CAPACITY_ARMS` …）与各 `EXPERIMENT_ARM == "..."` 分支只作数据保留，不再影响路径；整批删除会牵动十几处引用（源树试过，直接 NameError 回滚）。要删就单独一轮做，并以「import 自检 + 一次单 epoch smoke」收尾。

## 3. 启动前必须种一个探针 ckpt

新树没有任何产物，而 MCP 的 smoke 要求用**真实已训练权重**探梯度；它先在 `results/stage2_RQ-VAE/TIGER_<名>/<名>/rqvae_best.pth` 找，再找 `artifact_archive/TIGER_<名>_*/`，都没有就 fail-closed 拒绝启动。所以先种一份父树权重（同架构，仅作探针）：

```bash
mkdir -p results/stage2_RQ-VAE/TIGER_<名>/<名>
cp -p results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth \
      results/stage2_RQ-VAE/TIGER_<名>/<名>/rqvae_best.pth
```

`stage2_start` 会先归档它，smoke 通过后删除，训练从冷启动写出全新产物；回包 `smoke.checkpoint_source` 会标明这次探的是 `result_dir` 还是 `archive:...`。

## 4. 自检（不过不启动）

```bash
cd stage2_RQ-VAE/TIGER_<名>
/usr/bin/python3 -m py_compile train_rqvae.py <新机制模块>.py
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -c "
import sys; sys.path.insert(0, '.')
import train_rqvae as T
assert T.EXPERIMENT_ARM == '<名>', T.EXPERIMENT_ARM
assert T._OUT.parent.name == T.TREE_NAME == 'TIGER_<名>', T._OUT      # 路径自动跟目录名
assert T._OUT.name == '<名>' and T.LOG_DIR.parent.name == 'logs', T.LOG_DIR
print('arm', T.EXPERIMENT_ARM, '| out', T._OUT, '| log', T.LOG_DIR)
"
```

再用只读 MCP 状态工具确认服务器认得这棵树（应回 `idle` 而不是 `unknown`）：

```
write xd://mcp__genrec_stage2_status
{"source_dir": "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_<名>"}
```

## 5. 启动与监控

按 skill `stage2-start-mcp` 走：`stage2_start(source_dir=.../TIGER_<名>)` → `smoke_pending` → 每 60 秒 `stage2_status` 到 `running`，再按该 skill 的判据盯到 `finished`。

注意：启动前若改过 `tools/genrec-mcp/server.py`，需先 `/mcp reconnect genrec`，否则挂载进程仍在跑旧代码。
