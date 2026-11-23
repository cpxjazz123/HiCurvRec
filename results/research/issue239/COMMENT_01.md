## 节点 1：验收清单冻结

- 正文 sha256：`7a1ee963ef2b0432888b7587b31cd5ac42a603ed3136d98c4b7894065abebfc1`
- 条数：**16 条（C1–C16）**
- 清单：`results/research/issue239/RUN_LOG.md`

### 是否需要训练：**需要，且只训一次**

依据「评估与验收」：`K_q`、硬 q 的曲率保真、三码碰撞与 Stage3 Recall 只能由新训练的 Stage2 权重产生。
预算：3000 epochs、四卡、全局 batch 4096、seed 42、其余超参同冻结 TIGER，**一次**，不扫参、不多种子（C6）。

### 与 #238 的差别（清单里单列了一节）

| 项 | #238 | #239 |
|---|---|---|
| 量化器代价 | 混合距离（进 Sinkhorn） | **欧氏**（与 TIGER 逐字节相同，C2 用 sha256 比对） |
| 混合距离用途 | Sinkhorn 代价 + L_curv | **仅 L_curv**（C4） |

所以本 issue 天然避开 #238 的阻塞点（`center_distance` 守卫），但 C7 仍要求先交付根因证据。

### 实施顺序（按 issue「先检查阻塞根因，再训练」）

1. infra 复核（#238 的哈希是否仍有效）
2. **C7 根因诊断**：第 0/1/2 步的 encoder / 残差 / 码本 / 混合距离 / 梯度 的有限性、动态范围、参数更新量 → 逐量数值表
3. 实现（TDD）+ C8（欧氏代价与基线一致）+ C9（STE 路由与梯度）
4. smoke → 正式 Stage2 → 评估表 → Stage3

*（`results/research/issue239/COMMENT_01.md`）*
