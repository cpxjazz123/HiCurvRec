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
    "af3sign_capacity": (145, "AF3 符号容量分配", "L1=E32、L2/L3=H32（由 TRAIN 边 AF3 符号比例的 largest remainder 冻结），训练不含行为信号"),
    "factorized_joint_codebook": (147, "因子化 E×H 联合码本", "每层 16×16 分支码本对 → token=16*k_E+k_H，联合代价 + 基线 Sinkhorn"),
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
    "af3sign_capacity": [
        "若 AF3 符号比例极端导致某分量维数为 0 ⇒ 按确定性规则保留并报告",
        "若结构指标未改善 ⇒ 说明静态签名映射不受支持；D_AF3 改善但邻域/用量/碰撞恶化也不能称整体更好",
    ],
    "factorized_joint_codebook": [
        "若联合对明显稀疏或量化后结构未改善 ⇒ 如实记录为负结果",
        "若联合分配退化为分别 argmin 或绕过 Sinkhorn ⇒ 机制未实现",
    ],
}
#: The measured answer to each pre-registered counterexample, from the pinned
#: layer record and structural report. A number that was not recorded says so.
MEASURED = {
    "M1_learned_kappa": [
        "κ 轨迹 [-1,-1,-1] → [2.71e-7, -3.72e-6, 4.41e-6]，末步 κ 梯度 1.24e-4（非零）；"
        "test_gradient 4/4 通过（要求扰动改变至少一个 token）。结论不是「无作用」，而是"
        "**训练把曲率推到 0，几何退化为欧氏**。",
        "test_math 11/11 通过（κ→0± 极限、投影与定义域覆盖）；未观察到越界。",
        "hard SID：D_AF3 0.2880（基线 0.2887）、retention@16 0.0456（基线 0.0493）、"
        "collision 0.4618（基线 0.2745）、一级码 42/256（基线 247/256）。判 **NO_GAIN**："
        "几何自退化，且码本明显变差。",
    ],
    "M2_radial_manifold": [
        "机制事件只记录 `curvature_mean=-1.9997`（首末相同），**没有 radius_mean**；"
        "「半径是否由三个码字承载」无法从现有日志回答，记 NOT_RECORDED。",
        "`curvature_mean` 首末同为 -1.9997，未观测到非恒定曲率；结合 hard SID 的"
        "usage [15,64,254] 与 collision 0.8012，判 **NO_QUANTIZED_CURVATURE_LEVERAGE**。",
        "hard SID：D_AF3 0.2835（基线 0.2887）、retention@16 0.0374（基线 0.0493）、"
        "collision 0.8012（基线 0.2745）。",
    ],
    "M3_role_profiles": [
        "test_gradient 5/5 通过（encoder 与码本都收到梯度），耦合通路存在。",
        "机制事件只记录 `scale`（0.000462 → 2.8e-10），**没有 profile 预测误差**，记 NOT_RECORDED；"
        "hard SID：D_AF3 0.2841（基线 0.2887）略好，但 retention@16 0.0365（基线 0.0493）更差、"
        "collision 0.6900（基线 0.2745）、一级码 13/256。判 **未证实的量化收益**，且码本塌缩。",
    ],
    "M4_post_sid_af3": [
        "soft AF3 项 0.700 → 0.317（下降）；hard D_AF3 0.2781 < 基线 0.2887，hard 也随之下降，"
        "因此「soft 降而 hard 不降」这一条**不成立**。",
        "但 retention@16 0.0479 < 基线 0.0493、collision 0.3236 > 0.2745、一级码 156/256 < 247/256、"
        "重建 MSE 7.31e-5 > 6.71e-5：按第二条反证判 **未证实的结构收益**。",
        "test_gradient 7/7 通过（encoder/E-H 码本/逐层软分配均收到梯度）；test_entrypoint 1/1 通过"
        "（真实入口一 epoch 并导出）。",
        "首次完成运行的产物原被放在 `_aborted_620epochs/`（名称误导：内容是 3000 epoch 的完整导出），"
        "已原样复制到 `run_a_complete/` 并保留原目录；两次运行的 SID 分别为 "
        "`c6a0bfe3…`（首次）与 `6f71e6b2…`（交付），不做择优。",
    ],
    "af3sign_capacity": [
        "AF3 符号分布极极端：315391 条 TRAIN 边中 315390 负、1 零、0 正；largest remainder 给出 "
        "L2/L3 = H32 × E0 × S0，按确定性规则保留，球面分量为 0，因此没有球面代码路径。",
        "结构确实小幅改善：D_AF3 0.28763（基线 0.28866）、ret@16 0.05084（0.04922，首次超过基线）、"
        "Brier 0.24467（0.24827）、L3 碰撞 0.2704（0.2745）、最大三级桶 22（42）、Hard ER 6.296（6.468）。",
        "代价同时记录：一级码 177/256（基线 247）、L2 前缀 6701（7414）、重建 MSE 6.689e-5（6.665e-5）。"
        "判 **小幅结构改善但粗层容量与重建有代价**；效应量小，不宣称整体更好，也不宣称 Recall。",
        "test_math 6/6、test_gradient 3/3、test_ddp 1/1、test_entrypoint 1/1；碰撞闸门 True（0.2704 < 0.35）。",
    ],
    "factorized_joint_codebook": [
        "联合分配没有退化为分别 argmin：代价是 16×16 展开后的 256 个联合候选，Sinkhorn 仍按基线在最后一层执行，"
        "token 仍是 3 个、每层 256 个联合码（token=16*k_E+k_H）。",
        "联合对不稀疏：usage [220, 246, 256]，三级用满 256 个联合对；Hard ER 4.096（基线 6.468）、"
        "D_AF3 0.28593（0.28866）、Brier 0.24324（0.24827）、L2 前缀 5376（7414）改善。",
        "但 ret@16 0.04205（基线 0.04922）明显变差、L3 碰撞 0.3520（0.2745，闸门 False）、"
        "重建 MSE 7.48e-5（6.665e-5）变差：按预注册反证判 **未证实的整体结构改善**。",
        "test_math 5/5、test_gradient 3/3、test_ddp 1/1、test_entrypoint 1/1。",
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
    runs = payload.get("runs") or {}
    preserved = payload.get("preserved_run") or {}
    if runs.get("available") or preserved.get("available"):
        lines += ["", "## 重复运行与证据修复", ""]
        if runs.get("available"):
            audit = runs.get("audit") or {}
            delivered = audit.get("delivered") or {}
            lines.append("- 共享 metrics 流中的运行（按每次运行自己的起点分组）：")
            for start, body in sorted((audit.get("runs") or {}).items()):
                lines.append(f"  - `{start}`：{body['rows']} 行，epoch "
                             f"{body['first_epoch']}–{body['last_epoch']}，"
                             f"完整 1..{audit.get('epochs')} = {body['contiguous']}")
            lines.append(f"- 交付运行：`{delivered.get('run_start')}`（{delivered.get('rows')} 行，"
                         f"events={delivered.get('events')}）")
        if preserved.get("available"):
            lines += [f"- 首次完成运行的产物保存于 `{preserved['path']}`，SID SHA-256 "
                      f"`{preserved['sid_sha256']}`；与交付运行相同：{preserved['same_as_delivered']}",
                      "- 22:57 因误判进程被回收而重复启动了同一臂；重复运行不参与选优，"
                      "交付运行按“证据完整”这一与结果无关的规则选取，两次运行的差异如实并列。"]
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
                if "technically_valid_sid" in body:
                    lines += ["", "注：`technically_valid_sid` 是闸门（SID 可解析 + owner 的学习碰撞"
                              "阈值 ≤35%）的布尔结论，**不是**可解析性；为 False 时原因即上表"
                              " `metrics.l3_collision` 超阈，按诊断记录，不作本 Issue 的准入门槛。"]
    else:
        lines.append(f"- **缺失**：{record.get('reason', 'no record')}")
    structural = (REPO / "results" / "research" / "issue143"
                  / ("structure_TIGER_%s.json" % args.arm))
    if structural.is_file():
        body = json.loads(structural.read_text())
        lines += ["", "## 结构诊断（冻结支持、共享校准、Huber）", "",
                  f"- 报告：`{structural}`",
                  f"- SID SHA-256：`{body.get('sid', {}).get('sha256')}`",
                  f"- 支持：{body.get('support', {}).get('pairs')} 对（真边 "
                  f"{body.get('support', {}).get('edges')}）", "",
                  "| 指标 | 值 |", "|---|---|"]
        for key in ("distortion", "edge_half", "unobserved_half", "adjacency_brier",
                    "usage", "level2_prefixes", "level3_triples", "collision_full",
                    "collision_first", "max_triple_bucket", "max_level_code_usage"):
            lines.append(f"| {key} | {body.get(key)} |")
        retention = body.get("neighbour_retention@16") or {}
        lines.append(f"| neighbour_retention@16 | {retention.get('retention')} "
                     f"(k={retention.get('k')}, sample={retention.get('sample')}) |")
        reconstruction = body.get("reconstruction") or {}
        lines.append(f"| reconstruction_mse | {reconstruction.get('mse')} |")
        change = body.get("sid_change") or {}
        lines.append(f"| sid_changed_share | {change.get('changed_share')} "
                     f"(levels={change.get('levels')}) |")
        degree = body.get("degree_triangle") or {}
        if degree and not degree.get("computed", True):
            lines += ["", f"度/三角形误差**未计算**：{degree.get('reason')}"]
        control = REPO / "results" / "research" / "issue143" / "structure_TIGER_RQ-VAE.json"
        if control.is_file() and control != structural:
            base = json.loads(control.read_text())
            base_retention = base.get("neighbour_retention@16") or {}
            base_reconstruction = base.get("reconstruction") or {}
            lines += ["", "对照（冻结 TIGER，同一支持/同一校准/同一代码；诊断，不是无条件控制组）：", "",
                      "| 指标 | 冻结 TIGER | 本臂 |", "|---|---|---|",
                      f"| distortion | {base.get('distortion')} | {body.get('distortion')} |",
                      f"| adjacency_brier | {base.get('adjacency_brier')} | {body.get('adjacency_brier')} |",
                      f"| collision_full | {base.get('collision_full')} | {body.get('collision_full')} |",
                      f"| level2_prefixes | {base.get('level2_prefixes')} | {body.get('level2_prefixes')} |",
                      f"| usage | {base.get('usage')} | {body.get('usage')} |",
                      f"| neighbour_retention@16 | {base_retention.get('retention')} | {retention.get('retention')} |",
                      f"| reconstruction_mse | {base_reconstruction.get('mse')} | {reconstruction.get('mse')} |",
                      "", "注：`collision_full` 与分层记录的 `metrics.l3_collision` 是同一口径"
                      "（重复三级 SID 的商品占比），两处数值应一致；不一致即为计算缺陷，不得并列解释。"
                      "训练日志里的 `collision=` 是 `1 − 不同三元组/商品数`，会低估"
                      "（每个重复桶里有一个商品被当成不碰撞），不得用来替代上述冻结口径。"]
    lines += ["", "## 预注册反证与实测回答", ""]
    for item in COUNTEREXAMPLES[args.arm]:
        lines.append(f"- 预注册：{item}")
    lines += ["", "**实测回答（来自分层记录与结构报告的实际数值）**", ""]
    for item in MEASURED.get(args.arm, ["实测回答待结构报告生成后补写"]):
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
