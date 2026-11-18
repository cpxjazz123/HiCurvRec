# 仓库级保护审计

`audit_repo.py` 对仓库里每一棵 Stage2 源码树做**只读**探测(不安装、不改文件),回答一个问题:这棵树现在能不能在不坍缩的前提下训练。

## 审计前

```
TIGER_Uniform_KNN_RQ                NOT PROTECTED   the floor constant is missing
TIGER_Uniform_KNN_control           NOT PROTECTED   the floor constant is missing
TIGER_af3_geodesic_residual_rq      NOT PROTECTED   the floor constant is missing
TIGER_behavior_edge_contrastive_rq  NOT PROTECTED   the floor constant is missing
TIGER_graphlet_codebook_global_rq   NOT PROTECTED   the floor constant is missing
TIGER_graphlet_codebook_rq          NOT PROTECTED   the floor constant is missing
F0_seed42                           NOT PROTECTED   the floor constant is missing
F0_seed42_nomonitor                 NOT PROTECTED   the floor constant is missing
F0_seed42_nomonitor2                NOT PROTECTED   the floor constant is missing
TIGER_RQ-VAE                        READ-ONLY       the frozen copy source
TIGER_BEHAVIOR_SAMPLE_RQ-VAE        READ-ONLY       the frozen copy source

0/11 trees protected
```

也就是说:保护机制本身是完整的,但**仓库里没有任何一棵树装过它**。这正是“机制写完 ≠ 仓库受保护”的差距。

## 安装

对 9 棵树各跑一次 `selfcheck.py`(它安装、探测、审计,并给出 `PROTECTED` / `NOT PROTECTED`)。两棵冻结基线被拒绝改写(§19b),保持逐字节不变。

## 审计后

```
TIGER_Uniform_KNN_RQ                PROTECTED   VQLayer 8/8/8  EMAVQLayer 8/8/8
TIGER_Uniform_KNN_control           PROTECTED   VQLayer 8/8/8  EMAVQLayer 8/8/8
TIGER_af3_geodesic_residual_rq      PROTECTED   VQLayer 8/8/8  EMAVQLayer 8/8/8
TIGER_behavior_edge_contrastive_rq  PROTECTED   VQLayer 8/8/8  EMAVQLayer 8/8/8
TIGER_graphlet_codebook_global_rq   PROTECTED   MixedVQLevel 8/8/8
TIGER_graphlet_codebook_rq          PROTECTED   MixedVQLevel 8/8/8
F0_seed42                           PROTECTED   MixedVQLevel 8/8/8
F0_seed42_nomonitor                 PROTECTED   MixedVQLevel 8/8/8
F0_seed42_nomonitor2                PROTECTED   MixedVQLevel 8/8/8
TIGER_RQ-VAE                        READ-ONLY
TIGER_BEHAVIOR_SAMPLE_RQ-VAE        READ-ONLY

9/11 trees protected
```

## 安装没有破坏训练

对两族各跑一次真实入口(1 epoch + 导出):

```
TIGER_graphlet_codebook_rq  PASS   levels 256/256/256
TIGER_Uniform_KNN_RQ        PASS   levels 256/256/256
```

## 复现

```bash
<genrec_env_v2>/python3.9 results/research/qda/round13/audit_repo.py
```

只读,不修改任何树;要安装就对单棵树跑 `tools/research/stage2/anticollapse/selfcheck.py <tree>`。
