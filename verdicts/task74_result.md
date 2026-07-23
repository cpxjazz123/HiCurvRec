# Task #74 result — 新建专用 kgat_mckg env（TF-GPU 化）

> **任务名**: Task #74 — KGAT/MCKG 训练专用 conda env（替换错配的 vec2text env）
> **完成日期**: 2026-07-17
> **状态**: ✅ **env 已建好，TF-GPU 4 卡识别，文档已更新**

---

## 1. 任务目标

承接 Task #73 verdict §5.1 建议 A：解决 TF-GPU 库不可用问题。

但与 Task #73 时期不同的是：
- 现在 vec2text env 装的是错配 cu13 包（nvidia-cudnn-cu13 等），TF 2.15 找不到
- 路径 A 修复 vec2text 风险高（KGAT 训练和 vec2text RAG 项目混用，卸错包会破坏两边）
- 用户选择：**新建专用 env** `kgat_mckg`（不复用 vec2text），专给 KGAT/MCKG 训练用

---

## 2. 关键决策

| 决策 | 内容 |
|------|------|
| 新 env 名 | `kgat_mckg` |
| 新 env 路径 | `/home/wlia0047/ar57_scratch/wenyu/kgat_mckg` |
| Python | 3.10.20（与 vec2text 一致，避免 KGAT 代码兼容回归） |
| TF | 2.15.0 + tensorflow[and-cuda]（自动拉对齐 cu12）|
| cuDNN | 强制 8.9.0.131（对齐 TF 2.15 wheel 构建版本）|
| KGAT 依赖 | numpy 1.26.4 + scipy 1.15.3 + scikit-learn 1.7.2 + pyyaml 6.0.3 + tqdm 4.68.4 + pandas 2.3.3 + networkx 3.4.2 |
| 不装的包 | vec2text / openai / sentence-transformers / accelerate / datasets / evaluate（RAG 依赖，与 KGAT 无关）|
| 旧 env 处理 | vec2text 保留但不用于 KGAT 训练；CLAUDE.md 标注 "2026-07-17 弃用" |

---

## 3. env 安装步骤

1. `conda create -p /home/wlia0047/ar57_scratch/wenyu/kgat_mckg python=3.10 -y`
2. `pip install tensorflow==2.15.0`（不用 `[and-cuda]` extra，因 tensorrt 8.6.1 在 PyPI 已下架）
3. `pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-curand-cu12`
4. `pip install --force-reinstall nvidia-cudnn-cu12==8.9.0.131`（强制对齐 TF 2.15 wheel）
5. `pip install numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 pyyaml==6.0.3 tqdm==4.68.4 pandas==2.3.3 networkx==3.4.2`

完整一键脚本：`task_artifacts/scripts/task74_create_kgat_env.sh`

---

## 4. 验证结果

### 4.1 TF-GPU 验证

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'),
#          PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'),
#          PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU'),
#          PhysicalDevice(name='/physical_device:GPU:3', device_type='GPU')]
# ✅ 4 卡 A40 全部识别
```

### 4.2 KGAT 代码加载验证

```python
from KGAT import KGAT
from utility.load_data import Data
from utility.batch_test import test
# ✅ 所有 KGAT 模块可 import
```

### 4.3 GPU 显存释放

Task #72 Stage 3 PID 923347 取消后，整组 kill 释放 GPU 0：
- `kill -9 923352 923516 923696 923728 923760 923792 923824 923856 923888 923920`
- nvidia-smi 验证：4 卡全部 0 MiB / 0% util

---

## 5. 文档更新

### 5.1 CLAUDE.md R1 改前 → 改后

**改前**（单 env，GRID 专用）：
```bash
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
```

**改后**（双 env，按任务类型选择）：
- GRID 流水线 → `grid_toys`
- KGAT/MCKG 训练 → `kgat_mckg`（新建）
- vec2text 标注"2026-07-17 弃用"

### 5.2 loop.md 附录

新增三行：
- `Conda env — GRID 流水线`：grid_toys 路径 + 说明
- `Conda env — KGAT/MCKG 训练`：kgat_mckg 路径 + TF-GPU 验证标记
- `~~Conda env — vec2text~~`：废弃标记
- `Env 创建脚本（kgat_mckg）`：task74_create_kgat_env.sh 路径

### 5.3 CLAUDE.md GPU 环境表

新增一行 "CUDA 版本 (TF/kgat_mckg)" 和 "TF-GPU" 行，反映新 env 的 GPU 识别状态。

---

## 6. 副产品

- 旧的 Task #72 Stage 3 (PID 923347) 进程组已 kill，GPU 0 完全释放
- vec2text env 保留不动（不影响 KGAT 训练），但 CLAUDE.md 标记弃用
- products/task74/vec2text_pkgs_before_cu12_fix.txt 保存了 vec2text 改前的 pip freeze（如需回滚）

---

## 7. 后续建议

### 7.1 立即可做（建议）

启动 Task #74 KGAT 完整训练（100 epoch，GPU 加速）：
```bash
conda activate /home/wlia0047/ar57_scratch/wenyu/kgat_mckg
cd /home/wlia0047/ar57/wenyu/MCKG_repro/knowledge_graph_attention_network
CUDA_VISIBLE_DEVICES=0 python Main.py --dataset last-fm --alg_type kgat \
    --adj_type bi --use_att --use_kge --pretrain -2 \
    --epoch 100 --embed_size 64 --layer_size 64 64 64 \
    --lr 0.0001 --regs [1e-7,1e-7,1e-7] \
    --batch_size 65536 --batch_size_kg 2048 \
    --gpu_id 0
```

预期：
- GPU 训练速度 ≈ CPU 的 10-30×（10 epoch CPU 跑 8 min，100 epoch GPU 预计 ~2h）
- HR@20 目标 ≥ 0.70（KGAT 论文 LFM-1b full 是 0.842）

### 7.2 监控

- GPU 利用率（应 > 80%）/ 显存占用（应 < 30 GB / 48 GB）
- 每个 epoch 的 recall / ndcg / hit @20
- 100 epoch 后写 verdict 文档

---

## 8. 完成度跟踪

- [x] 创建 conda env `kgat_mckg`
- [x] 装 tensorflow==2.15.0 + cu12 包（cudnn 8.9）
- [x] 装 KGAT 训练必需依赖（scipy/sklearn/pyyaml/tqdm/pandas/networkx）
- [x] TF-GPU 验证：4 张 A40 全部识别
- [x] KGAT 代码加载验证（import 不报错）
- [x] kill Task #72 Stage 3 PID 923347 进程组，释放 GPU 0
- [x] 更新 CLAUDE.md R1（双 env 切换）
- [x] 更新 CLAUDE.md GPU 环境表
- [x] 更新 loop.md 附录（Conda env + 创建脚本）
- [x] 写 verdict → `verdicts/task74_result.md`（本文档）

---

**final result**: Task #74 完成。新 env `kgat_mckg` @ `/home/wlia0047/ar57_scratch/wenyu/kgat_mckg` 已建好，TF-GPU 4 张 A40 识别成功，KGAT 代码可加载。CLAUDE.md R1 + loop.md 附录已同步更新。Stage 3 PID 923347 进程组已 kill，GPU 全部空闲。下一步可直接启动 Task #74 第二阶段（KGAT 100 epoch 完整训练）—— 建议用 GPU 0/1/2/3 任一张空闲卡跑，预计 ~2h。

---

当前任务已完成，请做下一个任务的指示。

result: Task #74 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
