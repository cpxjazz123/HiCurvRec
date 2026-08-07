"""Issue #73 (Issue F) — 双曲体积增长可视化 (volume growth curve).

目的
----
为论文 §1.2「曲率 = 空间展开速度」提供直觉可视化: 把双曲空间体积增长率
    V_c(r) ∝ exp((d - 1) · sqrt(c) · r)
画成曲线族, 让读者直接看到 **κ 越大, 同 r 范围内能容纳的体积增长越快**.

数学背景
--------
曲率 -c (c = κ > 0) 的 d 维双曲空间中, 半径 r 测地球的精确体积为
    V_c(r) = S_{d-1} · ∫_0^r ( sinh(sqrt(c)·t) / sqrt(c) )^{d-1} dt
其中 S_{d-1} = 2·π^{d/2} / Γ(d/2) 为单位 (d-1)-球面面积.
当 sqrt(c)·r >> 1 时 sinh(x) ≈ e^x / 2, 积分被端点主导, 于是
    V_c(r) ~ exp((d - 1) · sqrt(c) · r)
这就是 issue 指定的比例式 (asymptotic proportionality), 本脚本主曲线即用它.

数值溢出处理 (关键)
-------------------
指数 (d-1)·sqrt(c)·r 在 d=128, c=5, r=5 时约为
    127 × 2.2360679... × 5 ≈ 1419.9
即 V ≈ e^1420 ≈ 10^616, 远超 float64 上限 (~1.8e308).
因此**绝不调用 np.exp 再取 log**, 而是**直接在 log10 域解析计算**:
    log10 V_c(r) = (d - 1) · sqrt(c) · r / ln(10)
纵轴即为 log10 V_c(r) (relative), 无任何溢出风险.

比值标注
--------
κ=5 相对 κ=0.1 在 r = ANNOT_R 处的体积比 (issue 硬要求, 必须由公式实际算出):
    ratio = exp( (d - 1) · (sqrt(5) - sqrt(0.1)) · r )
同样在 log10 域给出:
    log10 ratio = (d - 1) · (sqrt(5) - sqrt(0.1)) · r / ln(10)

产物
----
figure6_volume_growth.png / .pdf   Figure 6: 1×3 子图 (d ∈ D_LIST), 每子图 5 条 κ 曲线
volume_growth_numbers.json         全部数值结果 (log10 V 采样点 + log10 ratio)

运行 (R32: 直接 python3 执行, 无 .sh 包装; 0 GPU)
    python3 -u /fs04/ar57/wenyu/GeneRec/common/analysis/issue73_volume_growth.py
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# 常量区 (R30: 全部超参硬编码, 严禁 os.environ 读取)
# ============================================================================
D_LIST = [32, 64, 128]                    # RQ-VAE 常见隐空间维度
KAPPA_LIST = [0.1, 0.5, 1.0, 2.0, 5.0]    # 曲率绝对值 c = κ > 0 (空间曲率 = -κ)
R_MIN = 0.0                               # 半径下界
R_MAX = 5.0                               # 半径上界
N_POINTS = 500                            # r 采样点数
ANNOT_R = 2.0                             # 比值标注所在的 r 位置
ANNOT_KAPPA_HI = 5.0                      # 标注分子 κ (必须 ∈ KAPPA_LIST)
ANNOT_KAPPA_LO = 0.1                      # 标注分母 κ (必须 ∈ KAPPA_LIST)
REPORT_R_POINTS = [1.0, 2.0, 3.0, 4.0, 5.0]   # JSON / stdout 报告的 r 采样点

OUTPUT_DIR = Path("/fs04/ar57/wenyu/GeneRec/taskA/_history/issue73_volume_growth")
PNG_PATH = OUTPUT_DIR / "figure6_volume_growth.png"
PDF_PATH = OUTPUT_DIR / "figure6_volume_growth.pdf"
JSON_PATH = OUTPUT_DIR / "volume_growth_numbers.json"

FIG_DPI = 150
FIG_SIZE = (18.0, 5.6)                    # 1×3 横向排列
CURVE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
LN10 = math.log(10.0)


def log10_volume(d: int, kappa: float, r: np.ndarray) -> np.ndarray:
    """log10 V_c(r) —— 直接在 log10 域解析计算, 避免 exp 溢出.

    V_c(r) ∝ exp((d - 1) · sqrt(kappa) · r)
    ⟹ log10 V_c(r) = (d - 1) · sqrt(kappa) · r / ln(10)
    """
    if d < 2:
        raise ValueError(f"维度 d 必须 >= 2 (体积增长指数含 d-1 因子), 实际收到 d={d}")
    if kappa <= 0.0:
        raise ValueError(f"曲率绝对值 kappa 必须 > 0 (双曲空间), 实际收到 kappa={kappa}")
    return (d - 1) * math.sqrt(kappa) * r / LN10


def log10_volume_ratio(d: int, kappa_hi: float, kappa_lo: float, r: float) -> float:
    """log10 [ V_{kappa_hi}(r) / V_{kappa_lo}(r) ] —— 同样在 log10 域计算.

    ratio = exp( (d - 1) · (sqrt(kappa_hi) - sqrt(kappa_lo)) · r )
    """
    if kappa_hi <= kappa_lo:
        raise ValueError(
            f"标注要求 kappa_hi > kappa_lo, 实际 kappa_hi={kappa_hi} kappa_lo={kappa_lo}"
        )
    return (d - 1) * (math.sqrt(kappa_hi) - math.sqrt(kappa_lo)) * r / LN10


def validate_constants() -> None:
    """常量自检 —— 不符合预期直接 raise (R2: 禁止 fallback / 默认值兜底)."""
    if len(D_LIST) != 3:
        raise ValueError(f"issue #73 要求 3 张子图 (d ∈ {{32,64,128}}), 实际 D_LIST={D_LIST}")
    if len(KAPPA_LIST) != 5:
        raise ValueError(f"issue #73 要求每子图 5 条曲线, 实际 KAPPA_LIST={KAPPA_LIST}")
    if len(CURVE_COLORS) < len(KAPPA_LIST):
        raise ValueError(
            f"颜色数 {len(CURVE_COLORS)} 少于曲线数 {len(KAPPA_LIST)}"
        )
    if ANNOT_KAPPA_HI not in KAPPA_LIST:
        raise ValueError(f"ANNOT_KAPPA_HI={ANNOT_KAPPA_HI} 不在 KAPPA_LIST={KAPPA_LIST} 中")
    if ANNOT_KAPPA_LO not in KAPPA_LIST:
        raise ValueError(f"ANNOT_KAPPA_LO={ANNOT_KAPPA_LO} 不在 KAPPA_LIST={KAPPA_LIST} 中")
    if not (R_MIN <= ANNOT_R <= R_MAX):
        raise ValueError(f"ANNOT_R={ANNOT_R} 超出 r 范围 [{R_MIN}, {R_MAX}]")
    for r_pt in REPORT_R_POINTS:
        if not (R_MIN <= r_pt <= R_MAX):
            raise ValueError(f"报告点 r={r_pt} 超出 r 范围 [{R_MIN}, {R_MAX}]")
    if N_POINTS < 2:
        raise ValueError(f"N_POINTS 必须 >= 2, 实际 {N_POINTS}")


def compute_numbers(r_grid: np.ndarray) -> dict:
    """计算全部数值结果 (曲线 + 报告点 + 标注比值)."""
    curves = {}          # d -> kappa -> log10 V 曲线 (np.ndarray)
    report = {}          # d -> kappa -> {r: log10 V}
    ratios = {}          # d -> log10 ratio @ ANNOT_R

    for d in D_LIST:
        curves[d] = {}
        report[d] = {}
        for kappa in KAPPA_LIST:
            curves[d][kappa] = log10_volume(d, kappa, r_grid)
            report[d][kappa] = {
                f"r={r_pt:.1f}": float(log10_volume(d, kappa, np.array([r_pt]))[0])
                for r_pt in REPORT_R_POINTS
            }
        ratios[d] = log10_volume_ratio(d, ANNOT_KAPPA_HI, ANNOT_KAPPA_LO, ANNOT_R)

    return {"curves": curves, "report": report, "ratios": ratios}


def make_figure(r_grid: np.ndarray, numbers: dict) -> plt.Figure:
    """绘制 Figure 6 —— 1×3 子图, 每子图 5 条 κ 曲线 + 比值标注."""
    curves = numbers["curves"]
    ratios = numbers["ratios"]

    fig, axes = plt.subplots(1, len(D_LIST), figsize=FIG_SIZE)
    if len(D_LIST) == 1:
        raise ValueError("D_LIST 长度为 1 时 axes 非数组, issue #73 要求 3 子图")

    for ax_idx, d in enumerate(D_LIST):
        ax = axes[ax_idx]
        y_max = 0.0
        for k_idx, kappa in enumerate(KAPPA_LIST):
            y = curves[d][kappa]
            y_max = max(y_max, float(y.max()))
            ax.plot(
                r_grid,
                y,
                color=CURVE_COLORS[k_idx],
                linewidth=2.2,
                label=rf"$\kappa$ = {kappa:g}",
            )

        # 标注 r=ANNOT_R 处的参考竖线 + 两端点
        ax.axvline(ANNOT_R, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
        y_hi = float(log10_volume(d, ANNOT_KAPPA_HI, np.array([ANNOT_R]))[0])
        y_lo = float(log10_volume(d, ANNOT_KAPPA_LO, np.array([ANNOT_R]))[0])
        ax.plot([ANNOT_R, ANNOT_R], [y_lo, y_hi], color="k", linewidth=0.0)
        ax.scatter(
            [ANNOT_R, ANNOT_R], [y_lo, y_hi],
            s=42, facecolors="white",
            edgecolors=[CURVE_COLORS[0], CURVE_COLORS[-1]],
            linewidths=1.8, zorder=5,
        )

        # 比值标注文字 (放在子图内左上区域, 留白避开曲线)
        log10_ratio = ratios[d]
        annot_text = (
            rf"@ $r$ = {ANNOT_R:g} :  $V_{{\kappa=5}} / V_{{\kappa=0.1}}$" "\n"
            rf"$\log_{{10}}$ ratio = {log10_ratio:.1f}" "\n"
            rf"(i.e. $10^{{{log10_ratio:.1f}}}\times$ larger)"
        )
        ax.text(
            0.035, 0.965, annot_text,
            transform=ax.transAxes,
            va="top", ha="left", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffbe6",
                      edgecolor="#c9a227", linewidth=1.1, alpha=0.95),
            zorder=6,
        )

        ax.set_title(rf"$d$ = {d}", fontsize=14, fontweight="bold")
        ax.set_xlabel(r"geodesic radius  $r$", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel(r"$\log_{10} V_c(r)$   (relative)", fontsize=12)
        ax.set_xlim(R_MIN, R_MAX)
        # 顶部留 28% 空间给标注框, 避免遮挡曲线
        ax.set_ylim(0.0, y_max * 1.38)
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.65)
        ax.legend(loc="lower right", fontsize=10.5, framealpha=0.92,
                  title=r"curvature $|\kappa|$", title_fontsize=10.5)

    fig.suptitle(
        "Figure 6: Hyperbolic volume growth  "
        r"$V_c(r) \propto \exp((d-1)\sqrt{c}\,r)$  "
        "— larger curvature expands space faster (geometry is 'tighter')",
        fontsize=14.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def dump_json(numbers: dict) -> None:
    """把全部数值结果写成 JSON (log10 域, 无溢出)."""
    payload = {
        "issue": "#73 (Issue F) 双曲体积增长可视化 volume growth curve",
        "formula_proportional": "V_c(r) ∝ exp((d-1)*sqrt(c)*r)",
        "formula_exact_reference": "V_c(r) = S_{d-1} * ∫_0^r (sinh(sqrt(c)*t)/sqrt(c))^(d-1) dt",
        "log10_transform": "log10 V_c(r) = (d-1)*sqrt(c)*r / ln(10)",
        "note_overflow": (
            "d=128, c=5, r=5 时指数 (d-1)*sqrt(c)*r ≈ 1419.9 (V ≈ 10^616), "
            "float64 会溢出, 故全程在 log10 域解析计算, 未调用 np.exp"
        ),
        "config": {
            "D_LIST": D_LIST,
            "KAPPA_LIST": KAPPA_LIST,
            "R_MIN": R_MIN,
            "R_MAX": R_MAX,
            "N_POINTS": N_POINTS,
            "ANNOT_R": ANNOT_R,
            "ANNOT_KAPPA_HI": ANNOT_KAPPA_HI,
            "ANNOT_KAPPA_LO": ANNOT_KAPPA_LO,
            "REPORT_R_POINTS": REPORT_R_POINTS,
        },
        "log10_volume": {
            str(d): {f"kappa={kappa:g}": numbers["report"][d][kappa] for kappa in KAPPA_LIST}
            for d in D_LIST
        },
        "log10_ratio_kappa5_vs_kappa0.1_at_r2": {
            str(d): float(numbers["ratios"][d]) for d in D_LIST
        },
        "takeaway": (
            "曲率越大 → 同 r 范围内体积增长越快 → 几何『更紧』; "
            "维度 d 越高放大效应越强 (指数含 d-1 因子)"
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_tables(numbers: dict) -> None:
    """stdout 打印关键数值表格 (中文表头)."""
    report = numbers["report"]
    ratios = numbers["ratios"]

    print("=" * 92)
    print("Issue #73 双曲体积增长 — log10 V_c(r) 数值表 (V_c(r) ∝ exp((d-1)·√c·r))")
    print("=" * 92)

    for d in D_LIST:
        print(f"\n【隐空间维度 d = {d}】  单元格 = log10 V_c(r)")
        header = f"{'曲率 |κ|':>10} |" + "".join(f"{f'r={r_pt:.0f}':>13}" for r_pt in REPORT_R_POINTS)
        print(header)
        print("-" * len(header))
        for kappa in KAPPA_LIST:
            row = f"{kappa:>10g} |"
            for r_pt in REPORT_R_POINTS:
                row += f"{report[d][kappa][f'r={r_pt:.1f}']:>13.2f}"
            print(row)

    print("\n" + "=" * 92)
    print(f"关键标注: r = {ANNOT_R:g} 处 κ={ANNOT_KAPPA_HI:g} 相对 κ={ANNOT_KAPPA_LO:g} 的体积比值")
    print("=" * 92)
    header = f"{'维度 d':>10} |{'log10 比值':>16} |{'倍数 (科学记数)':>26}"
    print(header)
    print("-" * len(header))
    for d in D_LIST:
        lr = ratios[d]
        print(f"{d:>10} |{lr:>16.2f} |{f'10^{lr:.2f} 倍':>24}")

    print("\n关键 takeaway: 曲率越大 → 同 r 范围内体积增长越快 → 几何『更紧』")
    print("               (指数含 d-1 因子, 故维度越高该放大效应越剧烈)")
    print("=" * 92)


def main() -> None:
    validate_constants()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    r_grid = np.linspace(R_MIN, R_MAX, N_POINTS)
    numbers = compute_numbers(r_grid)

    fig = make_figure(r_grid, numbers)
    fig.savefig(PNG_PATH, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    dump_json(numbers)

    # 产物存在性硬校验 (缺失即 raise, 不做任何回退)
    for path in (PNG_PATH, PDF_PATH, JSON_PATH):
        if not path.is_file():
            raise RuntimeError(f"产物写出失败, 文件不存在: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"产物为空文件 (0 字节): {path}")

    print_tables(numbers)
    print("\n产物:")
    for path in (PNG_PATH, PDF_PATH, JSON_PATH):
        print(f"  {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
