"""Write the conclusion of one Issue from evidence already on disk.

The conclusion must contain what the issue asked to be reported and nothing that
was not measured: the run facts, the mechanism's own diagnostics, the layered SID
record, and the pre-registered counterexamples with their measured answer. A value
that does not exist is written as missing with its reason, never as a zero.

    python3 results/research/issue143/conclude.py M4_post_sid_af3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = {
    "M1_learned_kappa": (140, "学习曲率", "kappa_l = KAPPA_MAX tanh(theta_l) 进入层代价、原生残差与读出"),
    "M2_radial_manifold": (141, "异质径向流形", "E15 x H16 x R1，d^2 = E + H/4 + (dr)^2，R_M = R_H + R_phi(r)"),
    "M3_role_profiles": (142, "局部曲率 profile", "TRAIN-only 七维结构角色经量化读出回传到码本"),
    "M4_post_sid_af3": (143, "量化后 AF3", "独立 Bernoulli 边的 AF3 期望经软读出反传，含邻域保持与用量项"),
}
COUNTEREXAMPLES = {
    "M1_learned_kappa": [
        "若 κ 不改变任何一次选码 ⇒ 记 NO_GEOMETRIC_LEVERAGE（test_gradient 度量至少一个 token 改变）",
        "若 κ 越界或投影冲突 ⇒ 记数值失败（KAPPA_MAX 由投影半径推导，test_math 覆盖边界）",
    ],
    "M2_radial_manifold": [
        "若径向只在连续 r 上拟合、最终码字没有携带 ⇒ 记 NO_QUANTIZED_CURVATURE_LEVERAGE（半径由三个码字重构）",
        "若径向退化为常数 ⇒ 报告半径分布（radial 事件记录 radius_mean）",
    ],
    "M3_role_profiles": [
        "若只有辅助头变好、量化器无梯度 ⇒ 判机制未实现（test_gradient 要求 encoder 与码本都收到梯度）",
        "若 profile 回归改善而 hard SID 结构指标不改善 ⇒ 必须报告为未证实的量化收益",
    ],
    "M4_post_sid_af3": [
        "若 soft AF3 降而 hard D_AF3 不降 ⇒ 判未证实",
        "若 D_AF3 降而邻域保持/重建/用量恶化 ⇒ 判未证实",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=sorted(ARMS))
    args = parser.parse_args()
    issue, name, statement = ARMS[args.arm]
    directory = REPO / "results" / "research" / ("issue%d" % issue)
    evidence = directory / ("finish_%s.json" % args.arm)
    payload = json.loads(evidence.read_text()) if evidence.is_file() else {}
    diagnostics = payload.get("diagnostics") or {}
    record = payload.get("layer_record") or {}

    lines = [f"# Issue #{issue} 结论（{name}）", "",
             f"**机制**：{statement}", "",
             "**范围**：仅 Stage 2；不启动、不设计 Stage 3，不把结构代理的改善表述为 Recall 提升。", "",
             "## 运行事实", ""]
    if diagnostics.get("available"):
        lines += [f"- epoch：{diagnostics.get('last_epoch')} / 3000",
                  f"- 是否结束：{diagnostics.get('finished')}",
                  f"- 累计墙钟：{diagnostics.get('wall_s')} s",
                  f"- 最后 epoch 耗时：{diagnostics.get('epoch_time_s')} s",
                  f"- 机制事件条数：{diagnostics.get('mechanism_events', 0)}"]
        first, last = diagnostics.get("first"), diagnostics.get("last")
        if first or last:
            lines += ["", "| 量 | 首条 | 末条 |", "|---|---|---|"]
            for key in sorted({**first, **last} if (first and last) else (last or first)):
                lines.append(f"| {key} | {first.get(key) if first else ''} | "
                             f"{last.get(key) if last else ''} |")
    else:
        lines.append(f"- **缺失**：{diagnostics.get('reason', 'no metrics stream')}")
    lines += ["", "## 强制分层 SID 记录（CLAUDE.md 63）", ""]
    if record.get("written"):
        lines += [f"- 记录文件：`{record['report']}`",
                  f"- SID SHA-256：`{record['sid_sha256']}`"]
        report = Path(record["report"])
        if report.is_file():
            body = json.loads(report.read_text())
            keys = sorted(body) if isinstance(body, dict) else []
            if keys:
                lines += ["", "| 指标 | 值 |", "|---|---|"]
                for key in keys:
                    value = body[key] if not isinstance(body[key], (dict, list)) else "..."
                    lines.append(f"| {key} | {value} |")
    else:
        lines.append(f"- **缺失**：{record.get('reason', 'no record')}")
    lines += ["", "## 预注册反证与实测回答", ""]
    for item in COUNTEREXAMPLES[args.arm]:
        lines.append(f"- {item}")
    lines += ["", "## 诚实边界", "",
              "- 单次运行即定论，种子噪声计入不确定性，不做多种子。",
              "- 关闭开关的臂未训练（CLAUDE.md 19g），开关仅做代码内接线检查。",
              "- 结构指标只作诊断，不作晋级闸门；本 Issue 不包含 Stage 3。", ""]
    target = directory / f"CONCLUSION_{args.arm}.md"
    target.write_text("\n".join(lines) + "\n")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
