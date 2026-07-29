# HG-Rec Patches Directory

> **2026-07-29 创建 (Task #274)** — 用户授权 "don't ask me any question just do by yourself" (R11.1 自主决策)

本目录存放 HG-Rec upstream framework 的本地修复 patch 文件.

由于 `.gitignore` line 244 排除整个 `HG-Rec/` 目录, 任何本地修复都是 disk-only local fix, 新 clone 不会自动获得. 本目录 + `scripts/apply_hgrec_patches.sh` 提供 **patch 文件化 + 一键 apply** 解决方案, 不修改 .gitignore 也不 fork 上游.

## 当前 patches

| patch 文件 | 修复内容 | 关联任务 |
|---|---|---|
| `hrqvae_trainer_l322_unbound_local_error_fix.patch` | 修 `import glob, os` nested-scope 覆盖 → `import glob` 单 import. 让 step2 monitor 真打印 per-layer utilization | Task #265 (Issue #17 Gate 1) + Task #274 (patch 文件化) |

## 使用方式

```bash
cd $repo_root
bash scripts/apply_hgrec_patches.sh  # 一键 apply 所有 patches (idempotent)
```

apply 脚本会先 git diff 检测是否已 apply, 已 apply 跳过避免冲突.

## 设计原则

- **不修改 .gitignore**: 保持 HG-Rec/ 完全排除上游变更跟踪
- **不直接 git track 任何 HG-Rec/ 文件**: 避免 fork 上游
- **patch 文件 hunk 精确**: 每个 patch 只动 1 行 (或最小变更), 冲突概率最小
- **apply 脚本幂等**: 同一 patch apply 多次结果一致
- **每个 patch 必带 verdict 引用**: 决策透明 + 数字可审计
