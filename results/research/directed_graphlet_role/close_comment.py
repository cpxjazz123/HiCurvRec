"""Emit the issue #221 closing comment from the artefacts on disk.

Numbers come from ABLATION.md's source (ablation.json), so the comment cannot
drift from the runs; the narrative is fixed text.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    rows = json.loads((HERE / "ablation.json").read_text())
    table = ["| arm | L1 | L2 | L3 | collision | subtype F1 | orbit F1 | test_R@10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append("| {arm} | {L1} | {L2} | {L3} | {coll} | {sub} | {orb} | {rec} |".format(
            arm=row["arm"], L1=fmt(row.get("L1")), L2=fmt(row.get("L2")), L3=fmt(row.get("L3")),
            coll=fmt(row.get("collision")), sub=fmt(row.get("subtype_macro_f1")),
            orb=fmt(row.get("orbit_macro_f1")), rec=fmt(row.get("test_recall@10"))))
    text = f"""## 结项：有向结构进入 RQ-VAE 的完整对照已完成，判定为负结果并停止

按 #221 §3–§6 要求的链路逐项交付，全部证据在 `results/research/directed_graphlet_role/`
（`MANIFEST.json` 逐文件 sha256，`CGV.md` 是 CGV6 的可执行证据索引）。

### 1. 消融表（Stage2 冻结配方 3000 epoch、seed 42；Stage3 冻结协议）

{chr(10).join(table)}

读法：

- **U（无向标签）与 S（打乱方向）与 D1 现在只差标签模式**：同一棵树、同一 `ARM_VARIANT=D1`（头读 z）、同一预算。两臂最初以 `D3` 建树（梯度进码本），epoch 49 就撞码本健康门（L1 最大桶 0.988 / 三级碰撞 0.995），与 D2/D3/D4 的塌缩同型；改成 D1 后重跑，才成为「只差标签」的对照。
- **D0 没有 Stage3**：它的三级碰撞 0.1623 过不了 §89 的 0.10 入场门槛，`stage3_start` 会拒绝；冻结基线（0.1693）的 0.0555 来自门槛生效之前的历史 run，只能作参照并如此标注。
- 参照臂在跑通之前还修了三个会伪造结论的缺陷：无向词表缺 `allowed` 掩码、无向标签取的是整池前 N 行而不是本批、打乱模式把局部 orbit id 当全局 id 喂进 376 类 CE（直接导致 aux 非有限）。每个都有回归测试。

### 2. 三条已定稿的结论

1. **给定邻接，结构可识别**：subtype macro-F1 0.712、无条件 orbit 0.610；方向打乱后 0.169/0.146，邻接清零 0.030/0.012。
2. **不给邻接，结构不可预测**：z/q/SID/DGDV/度数/Dir-GNN 全部落在 0.493–0.514（SE 0.008）；行为边的存在性可读（分层负样本 AUC 0.69–0.73，度数基线 0.535），但**方向不可读**（0.495–0.507，训练集方向损失停在 ln 2）。
3. **结构监督进量化器没有兑现**：D1 的 SID 更细（碰撞 0.0253）但 SID→orbit 探针在任何深度都低于多数类基线，Stage3 0.0444 < 冻结基线 0.0555；量化本身吃掉约 1/3 的关系结构（z 0.0167 → q 0.0112）。

按 §6 的判断标准（「辅助 Loss 只到 head/Encoder、hard SID/Recall 不改善，明确记录失败并停止，而不是引入新的 AF3/几何模块掩盖负结果」），本 issue 以此负结果结项，不追加掩盖性模块。

### 3. 未执行的部分（明确记录，不是遗漏）

- **802 维 DGDV 输入臂不启动**：离线闸门在修掉反向边泄漏后为 0（0.493–0.514），按 §19g 的一次性预算不应投入。
- **多卡 DDP 证据是框架守卫**（两 rank gloo + 静态拒绝），不是 4-rank 有向臂：按 §19d0b，单卡主机上的有向臂走单进程分支、全局 batch 仍为 4096。

复现命令见 `REPORT_DIRECTED_GRAPHLET_ROLE.md` 各节末尾与 `CGV.md`。
"""
    (HERE / "issue_close_comment.md").write_text(text)
    print(text[:400])


if __name__ == "__main__":
    main()
