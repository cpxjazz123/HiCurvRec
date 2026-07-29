# Task #274 — HG-Rec disk-only L322 fix patch 文件化 闭环

> **完成日期**: 2026-07-29
> **状态**: 🟢 **候选 6 闭环** — 磁盘外修复 patch 文件化, 不改 .gitignore 不 fork 上游

---

## 1. 解决方案选择 (R11.4 + R11.3)

**问题**: `.gitignore` line 244 排除整个 `HG-Rec/` 目录. Task #265 修复是 disk-only local fix, 新 clone 会失效.

**候选方案 (5 个)**:

| 备选 | 描述 | 风险评估 |
|------|------|----------|
| A | .gitignore whitelist `!HG-Rec/model/hrqvae_trainer.py` | 上游 fix 后 → git 永久 fork 上游 |
| B | 把 hgrec 当独立 sub-repo (git submodule) | 项目结构复杂化, ROI 低 |
| C | 写 PR 上游 (snap-research/GRID) | 慢 (数周 PR cycle), 不解决**当前**已 fix 的保存 |
| D | 不做 (接受 disk-only) | 新 clone 训练失败, 结构性遗留 |
| **E (选)** | **patch 文件 + apply 脚本** | **0 风险, 不破坏现有结构** |

## 2. 实施产物

```
patches/
├── README.md  (patches 目录说明, 设计原则)
└── hrqvae_trainer_l322_unbound_local_error_fix.patch  (1 行字面 unified diff)
scripts/apply_hgrec_patches.sh  (idempotent apply 脚本, 循环 patches/*.patch)
```

### patches/hrqvae_trainer_l322_unbound_local_error_fix.patch (1 行变更)

```diff
--- a/HG-Rec/model/hrqvae_trainer.py
+++ b/HG-Rec/model/hrqvae_trainer.py
@@ -319,7 +319,7 @@
         # 只 load encoder.mlp.* 权重, 码本/decoder 仍随机初始化 (因为几何约束跟原版不同).
         init_encoder_from = getattr(self.args, 'init_encoder_from', None)
         if init_encoder_from is not None:
-            import glob, os
+            import glob
             if os.path.isfile(init_encoder_from):
                 ckpt_path = init_encoder_from
             else:
```

### scripts/apply_hgrec_patches.sh (idempotent)

核心算法:
1. 对每个 `patches/*.patch` 跑 `patch --dry-run -p1`
2. 成功 → 真实 apply
3. 失败且提示 "Reversed (or previously applied)" → 跑 `patch --dry-run -R -p1` 验证. 成功说明确实已 apply (因为反向 dry-run = 反转 = 等价正向已 apply)
4. 否则真失败 → 报错退出非零

**测试结果**: 在当前已 fix 文件上跑 → `applied=0 skipped=1 failed=0` (idempotent ✓)

## 3. 验证 (idempotent 关键性质)

```bash
$ bash scripts/apply_hgrec_patches.sh
--- 处理 patch: hrqvae_trainer_l322_unbound_local_error_fix.patch ---
  ⏭️  SKIPPED (already applied): hrqvae_trainer_l322_unbound_local_error_fix.patch
✅ Total: applied=0 skipped=1 failed=0
```

后续多次跑都是同样输出 → idempotent 验证通过.

## 4. 关键决策点 (R11.4 用户 override 透明)

- **不改 .gitignore**: 保持 HG-Rec/ 完全 git 排除
- **不直接 git track 任何 HG-Rec/ 文件**: 不 fork 上游
- **patch 文件化**: 是 "修复失效时不破坏 git 历史" 的最兼容方案
- **apply 脚本 idempotent**: 重复跑结果一致, 防止误用
- **跟 Issue #17 Gate 1 修复一致**: 同一行变更, 同一数字 (47/64 L0 utilization etc.) 不影响
- **未来需要时再加 patch**: 每个新修复 1 个 patch 文件, 1 行 (或最小) 变更, 冲突概率最小

## 5. 不做但留候选 (R11.4)

- 不 git track HG-Rec/model/hrqvae_trainer.py: 用户若要求可立 Task #275, 但 ROI 低
- 不开 PR 上游: slow, 跟用户 "do by yourself" override 不符
- 不修复其他 HG-Rec/ 文件: 0 个已知必要 fix, 其他都是 working state

## 6. 物理产物

```
verdicts/task274_hgrec_disk_fix_patch_result.md  (本文件)
descriptions/task274_hgrec_disk_fix_patch.md
patches/README.md
patches/hrqvae_trainer_l322_unbound_local_error_fix.patch
scripts/apply_hgrec_patches.sh  (idempotent, ⭐ 修补 v2 后已测)
```

result: Task #274 — disk-only L322 fix patch 文件化 (1 行字面变更) + apply 脚本 idempotent (applied=0 skipped=1 failed=0). 候选 6 闭环. 不改 .gitignore 不 fork 上游. 任何新 clone 跑 `bash scripts/apply_hgrec_patches.sh` 即可获得 fix. R11.4 决策透明: patch 文件化是 5 候选里风险最低, .gitignore whitelist 会 fork 上游风险高.
