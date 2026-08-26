"""Stage 4 — HG-Rec T5 test eval (DDP 4 卡, beam=20).

主目录默认入口: 用 Stage 3 v19 best_ckpt.pt 跑 test 集评估.
输出 test_R@10, test_R@20, test_NDCG@10/20.

启动方式 (任选其一):
  1) python3 stage4_beam20.py
  2) CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 \
       /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/stage4_beam20.py

R35: 单 ckpt + beam=20, 禁 Borda Rank Fusion.
R35b: DDP 4 卡各自分片不重复评估 test 集, all_reduce SUM.
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.

⚠️ 重要: 必须设 FORCE_HGREC=1 环境变量!
test_eval_only.py line 54 gin.query_parameter 抛 ValueError
(use_hgrec_arch 非 @gin.configurable formal param),
fallback to use_hgrec=False 走错路径 (EncoderDecoderRetrievalModel
加载 HG-Rec 格式 best_ckpt strict=False 静默失败, 模型随机初始化 → Recall ≈ 0).
FORCE_HGREC=1 bypass 这个 bug, 直接走 _test_eval_hgrec() 分支.

输入:
  - Stage 3 产物: best_ckpt.pt (R34b fix 后默认路径见 resolve_stage3_ckpt)
  - gin config:    /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/configs/
                  decoder_instruments_hgrec_v19.gin

产物:
  - test_final.json (与 best_ckpt.pt 同目录)

预期 test_R@10 ≈ 0.110 (vs Issue239 commit 0.1102, -0.001 noise 内).
R37 决策点:
  test_R@10 ≥ 0.110 → R37 PASS (v19 历史最佳)
  test_R@10 < 0.110 → R37 FAIL (回退 v18)
"""
import os
import subprocess
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE4_TORCHRUN_BASE = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29778",
]


def build_stage4_cmd(ckpt_path):
    """构造 stage4 torchrun 命令 (ckpt 在 main() 时 resolve)."""
    return STAGE4_TORCHRUN_BASE + [
        os.path.join(MAIN_DIR, "test_eval_only.py"),
        os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v19.gin"),
        ckpt_path,
    ]

# R53 v3.8: Stage 3 best_ckpt 路径从 curvature_config 硬编码 derive
# 不再写死 v19 绝对路径 (历史 R34b fix 残留, 已废弃)
from curvature_config import BEST_CKPT_PATH as _BEST_CKPT_PATH_DEFAULT
STAGE3_CKPT_CANDIDATES = [_BEST_CKPT_PATH_DEFAULT]


def resolve_stage3_ckpt():
    """自动发现 Stage 3 best_ckpt 路径 (兼容 R34b 前后两种输出布局)."""
    for p in STAGE3_CKPT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Stage 3 best_ckpt 不存在. 检查路径:\n" + "\n".join(f"  - {p}" for p in STAGE3_CKPT_CANDIDATES)
        + f"\n请先跑 stage3.py 训练 HG-Rec T5."
    )


def check_stage3_artifact():
    """R40: Stage 3 best_ckpt 必须存在 (R34b fix 后支持路径自动发现)."""
    ckpt_path = resolve_stage3_ckpt()
    print(f"[stage4] Stage 3 best_ckpt OK: {ckpt_path}")
    return ckpt_path


def main():
    ckpt_path = check_stage3_artifact()

    # R53 v3.8: CUDA_VISIBLE_DEVICES 硬编码自 curvature_config.py, USE_HGREC_ARCH 硬编码为 True
    # 不传 env= 让 subprocess 继承父进程 env (curvature_config.py 已设置全部 env var)
    from curvature_config import CUDA_VISIBLE_DEVICES as _CUDA_VISIBLE_DEVICES

    cmd = build_stage4_cmd(ckpt_path)
    print(f"[stage4] launching DDP 4-card HG-Rec T5 test eval (beam=20)")
    print(f"[stage4] command: {' '.join(cmd)}")
    print(f"[stage4] CUDA_VISIBLE_DEVICES={_CUDA_VISIBLE_DEVICES} (硬编码自 curvature_config.py)")
    print(f"[stage4] USE_HGREC_ARCH=True (硬编码自 curvature_config.py, 替代 FORCE_HGREC=1)")
    print(f"[stage4] best_ckpt: {ckpt_path}")
    print(f"[stage4] expected test_R@10 ≈ 0.110 (Issue239 baseline)")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[stage4] FAIL exit={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # 解析 test_final.json 打印 R37 决策点 (路径随 ckpt 路径动态推导)
    out_dir = os.path.dirname(ckpt_path)
    out_json = os.path.join(out_dir, "test_final.json")
    if os.path.exists(out_json):
        import json
        with open(out_json) as f:
            metrics = json.load(f)
        r10 = metrics.get("test_R@10", 0)
        print(f"\n[stage4] test_R@10 = {r10:.4f}")
        if r10 >= 0.110:
            print(f"[stage4] R37 PASS (test_R@10 ≥ 0.110 baseline)")
        else:
            print(f"[stage4] R37 FAIL (test_R@10 < 0.110 baseline)")
    print(f"[stage4] done → {out_json}")


if __name__ == "__main__":
    main()