#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneRec 实验任务可视化 — Streamlit UI.

启动:
    streamlit run scripts/webdash_streamlit/app.py \
        --server.address 127.0.0.1 \
        --server.port 8501 \
        --browser.gatherUsageStats false \
        --server.fileWatcherType none

R30: 所有路径硬编码在 collectors.py, 本文件不引入 os.environ.get
R2 : 解析失败 raise; 缺字段 None (图表容错但不补默认值)
R35: 默认只展示 beam=20 单 ckpt 评估; 高亮非 beam=20 / 非单 ckpt 的结果
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---- 让 collectors.py 可导入 ----
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collectors as C  # noqa: E402

# ---- streamlit-autorefresh (装在 ~/.local, 隔离 vendor 防污染其他 import) ----
# 注意: 不能直接 sys.path.insert ~/.local, 否则 ~/.local/altair/jsonschema/rpds
# 会抢先于 anaconda site-packages, 导致 st.bar_chart 报 rpds.rpds 找不到.
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.local_vendor")
from streamlit_autorefresh import st_autorefresh  # noqa: E402

# ============================================================
# Streamlit 页面配置
# ============================================================

st.set_page_config(
    page_title="GeneRec 实验管理",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义颜色 / 通用样式
# ============================================================

STATUS_COLOR = {
    "PASS": "#16a34a",
    "FAIL": "#dc2626",
    "?":   "#9ca3af",
}

METRIC_COLOR_GOOD = "#16a34a"
METRIC_COLOR_BAD = "#dc2626"

st.markdown(
    """
<style>
    .kpi-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: left;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.08em; margin: 4px 0 0 0; }
    .status-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; color: white;
    }
    .small-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 0.78rem; color: #475569; }
    div[data-testid="stDataFrame"] table { font-size: 0.82rem; }
</style>
""",
    unsafe_allow_html=True,
)


def badge(status: str | None) -> str:
    color = STATUS_COLOR.get((status or "?").upper(), "#9ca3af")
    return f'<span class="status-badge" style="background:{color}">{status or "?"}</span>'


def kpi_card(label: str, value: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


# ============================================================
# 自动刷新 — streamlit-autorefresh 实现内部 rerun (无整页 reload)
# ============================================================

_REFRESH_SECONDS = 5


def autorefresh(seconds: int = _REFRESH_SECONDS, page_key: str = "default") -> None:
    """Streamlit 内部 rerun 刷新数据, 不重载 HTML/CSS/JS.

    工作原理: streamlit-autorefresh 注入一个 component, 用 setInterval
    周期性 streamlit_setComponentValue 触发 Python 端 rerun().
    配合 st.cache_data(ttl=4) 实现数据层 ~5s 自刷新.
    """
    try:
        st_autorefresh(interval=int(seconds * 1000), key=f"auto_refresh_{page_key}", limit=None)
    except Exception:  # noqa: BLE001
        # fallback (R2: 不补默认值, 这里仅 try/except 保证 UI 不中断)
        pass


# ============================================================
# 缓存包装 — 让数据采集函数在 4s 内只跑一次
# ============================================================


@st.cache_data(ttl=4, show_spinner=False)
def _kpis():
    return C.overview_kpis()


@st.cache_data(ttl=4, show_spinner=False)
def _tasks():
    return C.collect_tasks()


@st.cache_data(ttl=4, show_spinner=False)
def _exps():
    return C.collect_experiments()


@st.cache_data(ttl=4, show_spinner=False)
def _verdicts():
    return C.collect_verdicts()


@st.cache_data(ttl=2, show_spinner=False)
def _running():
    return C.collect_running()


@st.cache_data(ttl=4, show_spinner=False)
def _gpu():
    return C.collect_gpu()


@st.cache_data(ttl=4, show_spinner=False)
def _curvature():
    return C.collect_curvature_frameworks()


@st.cache_data(ttl=4, show_spinner=False)
def _stage_innovations():
    return C.collect_stage_innovations()


@st.cache_data(ttl=60, show_spinner=False)
def _train_curve(eval_path: str):
    df = C.load_train_curve(eval_path)
    if df is None or df.empty:
        return df
    return df


# ============================================================
# 页面 — Overview
# ============================================================


def page_overview() -> None:
    autorefresh(5, page_key="overview")
    st.title("📊 Overview")
    st.caption("全局 KPI · Top 实验 · Verdict 分布 · 当前活跃实验")

    k = _kpis()

    # 4 KPI 行
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Tasks (tasks/)",  f"{k['tasks_count']}"),    unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Experiment Evals",  f"{k['experiments_count']}"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Verdicts",          f"{k['verdicts_count']}"),  unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Running Procs",     f"{k['running_count']}"),  unsafe_allow_html=True)

    st.markdown("")
    c5, c6, c7 = st.columns(3)
    with c5: st.markdown(kpi_card("Verdict PASS",   f"{k['pass_count']}"),   unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Verdict FAIL",   f"{k['fail_count']}"),   unsafe_allow_html=True)
    with c7: st.markdown(kpi_card("GPUs",           f"{k['gpu_count']}"),    unsafe_allow_html=True)

    st.divider()

    # Top 10 by R@10
    st.subheader("🏆 Top 10 by R@10 (all evaluations)")
    top = k.get("top10_by_R10") or []
    if not top:
        st.info("暂无 eval_test.json 数据.")
    else:
        df_top = pd.DataFrame(top)
        df_top["R@10"] = df_top["R@10"].map(lambda x: f"{x*100:.2f}%" if x is not None else "—")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.divider()

    # Verdict 分布条形图
    st.subheader("📋 Verdicts 决策分布")
    verdicts = _verdicts()
    if verdicts.empty:
        st.info("暂无 verdict 数据.")
    else:
        status_counts = verdicts["status"].value_counts().reindex(["PASS", "FAIL", "?"], fill_value=0)
        st.bar_chart(status_counts)

    # 最近 5 条 verdicts
    st.subheader("🆕 最近 5 条 Verdict")
    if not verdicts.empty:
        recent = verdicts.head(5).copy()
        recent["status_html"] = recent["status"].apply(badge)
        st.write(
            recent[["issue_iid", "status_html", "gate", "summary"]].to_html(
                escape=False, index=False, header=["Issue", "Status", "Gate", "Summary"]
            ),
            unsafe_allow_html=True,
        )

    # 当前活跃
    st.divider()
    st.subheader("🔥 当前活跃进程")
    running = _running()
    if running.empty:
        st.success("✅ 无活跃训练进程 (所有 GPU 可用)")
    else:
        st.dataframe(running[["pid", "started_at", "cmdline_short"]], use_container_width=True, hide_index=True)


# ============================================================
# 页面 — Tasks
# ============================================================


def page_tasks() -> None:
    autorefresh(5, page_key="tasks")
    st.title("🧪 Tasks Roster")
    st.caption("tasks/<name>/ 各任务的 4 阶段脚本齐备状态 / seed / R 规则摘要")

    df = _tasks()
    if df.empty:
        st.warning(f"无任务目录: {C.TASKS_DIR}")
        return

    # 顶部筛选
    q = st.text_input("🔍 关键字过滤 (task 名 / summary)", "")
    only_complete = st.checkbox("只显示 4 阶段脚本齐全 (scripts == 4)", value=False)

    if q:
        mask = df["task"].str.contains(q, case=False, na=False) | df["summary"].fillna("").str.contains(q, case=False, na=False)
        df_view = df[mask].copy()
    else:
        df_view = df.copy()
    if only_complete:
        df_view = df_view[df_view["scripts"] == 4]

    # 标注: scripts=4 -> 齐备
    df_view["stage_status"] = df_view["scripts"].apply(lambda n: "🟢" if n == 4 else f"⚠️ {n}/4")

    display = df_view[
        ["task", "stage_status", "scripts", "seed", "has_train_log", "r_rules", "summary", "modified_at"]
    ].copy()
    display["modified_at"] = display["modified_at"].astype(str)

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption(f"共 {len(df_view)} / {len(df)} 个任务目录")

    # 选中任务显示 detail
    st.divider()
    st.subheader("📂 任务目录详情")
    sel = st.selectbox("选择任务", options=["(无)"] + df_view["task"].tolist(), index=0)
    if sel and sel != "(无)":
        task_dir = C.TASKS_DIR / sel
        st.code(str(task_dir), language="text")
        files = sorted([p for p in task_dir.iterdir() if p.is_file()])
        st.write("**文件清单**")
        st.write({p.name: f"{p.stat().st_size:,} bytes  (mtime={pd.Timestamp(p.stat().st_mtime, unit='s', tz='UTC')})" for p in files})


# ============================================================
# 页面 — Experiments
# ============================================================


def page_experiments() -> None:
    autorefresh(5, page_key="experiments")
    st.title("📈 Experiments")
    st.caption("taskA/_history/<exp>/**/eval_test.json 全表聚合, 默认按 R@10 降序, 高亮疑似违规 (非单 ckpt / 非 beam=20)")

    df = _exps()
    if df.empty:
        st.warning(f"无 eval_test.json 数据 ({C.HISTORY_DIR})")
        return

    # 筛选栏
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        metric = st.selectbox("排序指标", options=list(C.EVAL_METRICS), index=1)
    with f2:
        exp_filter = st.multiselect("按 exp 过滤", options=sorted(df["exp"].unique()))
    with f3:
        only_top20 = st.checkbox("只看 R@10 ≥ 0.10", value=False)
    with f4:
        beam20_only = st.checkbox("只看疑似单 ckpt + beam=20 (tag 或 variant 推断)", value=False)

    view = df.copy()
    if exp_filter:
        view = view[view["exp"].isin(exp_filter)]
    if only_top20 and "R@10" in view.columns:
        view = view[view["R@10"].fillna(0) >= 0.10]
    if beam20_only:
        # tag 含 'beam20' / 'b20' 视为合规; 反之高亮
        tag_str = view["tag"].fillna("") + " " + view["variant"].fillna("")
        view = view[tag_str.str.contains(r"beam20|b20", case=False, regex=True, na=False)]

    view = view.sort_values(metric, ascending=False, na_position="last").reset_index(drop=True)

    # R35 标注: tag 里有 'borda' / 'ensemble' / '3way' / '5way' / '7way' → 违规
    def _r35_alert(row: pd.Series) -> str:
        s = " ".join([
            str(row.get("tag") or ""),
            str(row.get("variant") or ""),
        ]).lower()
        offenders = []
        if re := __import__("re").search(r"b\d+way|ensembl|borda", s):
            offenders.append("R35:ensemble/borda")
        if re := __import__("re").search(r"beam(?!20)\d+", s):
            offenders.append("R35:beam≠20")
        if re := __import__("re").search(r"3way|4way|5way|6way|7way|8way", s):
            offenders.append("R35:multiway")
        return " ✅" if not offenders else f" ⚠️ {','.join(offenders)}"

    view["r35"] = view.apply(_r35_alert, axis=1)

    # 数值格式化
    for col in C.EVAL_METRICS:
        if col in view.columns:
            view[col] = view[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    if "t_eval_s" in view.columns:
        view["t_eval_s"] = view["t_eval_s"].apply(lambda x: f"{x:.1f}s" if pd.notna(x) else "—")

    cols = ["exp", "variant", "tag", *C.EVAL_METRICS, "t_eval_s", "n_eval", "r35", "done_at", "ckpt"]
    view = view[cols].copy()
    view["ckpt"] = view["ckpt"].apply(lambda x: (str(x).split("/")[-1] if x else "—"))

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.caption(f"共 {len(view)} 条评估记录 (总计 {len(df)})")

    # 单条评估 + train_curve 详情
    st.divider()
    st.subheader("🔬 实验评估详情 + 训练曲线")
    options = [f"{r['exp']}/{r['variant']}/{r.get('tag') or r.get('eval_path','')[:64]}" for _, r in df.iterrows()]
    sel = st.selectbox("选择评估 (按 R@10 倒序)", options=options, index=0)
    if sel:
        idx = options.index(sel)
        row = df.iloc[idx]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**评估字段**")
            kv = {k: row.get(k) for k in C.EVAL_METRICS}
            st.json({k: (None if pd.isna(v) else v) for k, v in kv.items()})
            st.write("**CKPT**:", row.get("ckpt") or "—")
            st.write("**SID SHA (16)**:", row.get("sid_sha256") or "—")
            st.write("**Eval Path**:", row.get("eval_path"))
            st.write("**Done At**:", row.get("done_at") or "—")
        with c2:
            curve = _train_curve(row.get("eval_path") or "")
            if curve is None or curve.empty:
                st.info("未找到匹配的 train_curve.json (Stage 3 训练记录缺失或非本子目录)")
            else:
                st.markdown("**训练 loss 曲线**")
                st.line_chart(curve.set_index("epoch")[["loss", "recon_loss"]])
                st.markdown("**κ 平均值**")
                st.line_chart(curve.set_index("epoch")[["kappa_mean"]])


# ============================================================
# 页面 — Verdicts
# ============================================================


def page_verdicts() -> None:
    autorefresh(8, page_key="verdicts")
    st.title("📋 Verdicts 历史")
    st.caption("verdicts/<iid>/ 全部判定结果. R35/R36/R37/R38 等约束提示已嵌入 summary.")

    df = _verdicts()
    if df.empty:
        st.warning(f"无 verdict 数据 ({C.VERDICTS_DIR})")
        return

    # 筛选
    f1, f2, f3 = st.columns(3)
    with f1:
        statuses = ["PASS", "FAIL", "?"]
        chosen_status = st.multiselect("状态", options=statuses, default=statuses)
    with f2:
        iid_min, iid_max = int(df["issue_iid"].min()), int(df["issue_iid"].max())
        iid_range = st.slider("Issue iid 范围", min_value=iid_min, max_value=iid_max, value=(iid_min, iid_max))
    with f3:
        q = st.text_input("关键字 (summary / gate)", "")

    view = df[
        df["status"].isin(chosen_status)
        & df["issue_iid"].between(iid_range[0], iid_range[1])
    ].copy()
    if q:
        view = view[
            view["summary"].fillna("").str.contains(q, case=False, na=False)
            | view["gate"].fillna("").astype(str).str.contains(q, case=False, na=False)
        ]

    view = view.sort_values("issue_iid", ascending=False).reset_index(drop=True)
    view["status_html"] = view["status"].apply(badge)
    view["modified_at"] = view["modified_at"].astype(str)

    disp = view[
        ["issue_iid", "status_html", "gate", "evaluated_at", "summary", "file", "modified_at"]
    ].copy()

    st.write(
        disp.to_html(
            escape=False,
            index=False,
            header=["Issue", "Status", "Gate", "Evaluated At", "Summary", "File", "Modified At"],
        ),
        unsafe_allow_html=True,
    )

    st.caption(f"共 {len(view)} / {len(df)} 条 verdicts")

    # 单条 verdict 文件全文
    st.divider()
    st.subheader("📄 Verdict 文件全文")
    if not view.empty:
        options = [f"#{int(r['issue_iid'])} · {r['file']}" for _, r in view.iterrows()]
        sel = st.selectbox("选择 verdict", options=options, index=0)
        if sel:
            idx = options.index(sel)
            row = view.iloc[idx]
            content = Path(row["file_path"]).read_text(encoding="utf-8", errors="replace")
            ext = Path(row["file_path"]).suffix
            if ext == ".json":
                st.json(_safe_json(content))
            else:
                st.code(content, language="markdown")


def _safe_json(text: str):
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": text[:1000]}


# ============================================================
# 页面 — Live
# ============================================================


def page_live() -> None:
    autorefresh(3, page_key="live")
    st.title("🔥 Live · 训练进程监控")
    st.caption("当前本机 GeneRec 相关 python3 训练进程 + GPU 利用率. 3s 自动刷新.")

    # GPU 表格
    st.subheader("🎛 GPU 状态")
    g = _gpu()
    if g.empty:
        st.error("nvidia-smi 不可用或本机无 GPU")
    else:
        g_disp = g.copy()
        g_disp["util_pct"] = g_disp["util_pct"].astype(str) + "%"
        g_disp["mem_used_mb"] = g_disp["mem_used_mb"].astype(str) + " MB"
        g_disp["mem_total_mb"] = g_disp["mem_total_mb"].astype(str) + " MB"
        g_disp["temp_c"] = g_disp["temp_c"].astype(str) + "°C"
        st.dataframe(g_disp, use_container_width=True, hide_index=True)

    st.divider()

    # 运行进程
    st.subheader("🟢 活跃 Python 训练进程")
    running = _running()
    if running.empty:
        st.success("无 GeneRec 相关 python 训练进程.")
    else:
        running["duration"] = (
            pd.Timestamp.now(tz="UTC") - pd.to_datetime(running["started_at"], utc=True)
        ).astype(str)
        st.dataframe(
            running[["pid", "started_at", "duration", "cwd", "cmdline_short"]],
            use_container_width=True, hide_index=True,
        )

        # 单选展开 cmdline 全文 + 默认 log 路径
        sel_pid = st.selectbox("选 PID 看完整命令行 + 末尾日志", options=running["pid"].astype(int).tolist(), index=0)
        if sel_pid:
            row = running[running["pid"] == sel_pid].iloc[0]
            st.markdown(f"**PID {int(sel_pid)} 完整命令行**")
            st.code(row["cmdline_full"], language="bash")
            # 推断 train.log
            log_candidates = []
            cmd = row["cmdline_full"]
            import re as _re
            m = _re.search(r"tasks/([\w\-]+)/stage\d\.py", cmd)
            if m:
                cand = C.TASKS_DIR / m.group(1) / "train.log"
                log_candidates.append(cand)
            # 如果是 history 路径
            m2 = _re.search(r"--product_dir\s+(\S+)", cmd)
            if m2:
                pd_dir = Path(m2.group(1))
                for stem in ("train_pure_t5.log", "train.log"):
                    log_candidates.append(pd_dir / stem)
            # 兜底 (R2: 默认值禁止, 这里仅做"候选搜索", 不补值)
            for log in log_candidates:
                if log.is_file():
                    st.markdown(f"**📄 最近 80 行: {log}**")
                    try:
                        with log.open("r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        st.code("".join(lines[-80:]), language="log")
                    except OSError as exc:
                        st.error(f"读 log 失败: {exc}")
                    break
            else:
                st.info("未找到 train.log / train_pure_t5.log (R2: 不补默认路径)")


# ============================================================
# 页面 — Curvature Frameworks
# ============================================================


def page_curvature() -> None:
    autorefresh(5, page_key="curvature")
    st.title("🧬 Curvature Frameworks")
    st.caption(
        "按曲率机制聚合 taskA/_history/ 下所有 eval_test.json, 默认按 best_R@10 降序. "
        "R35 违规 (Borda / multi-way / ensemble) 用红/黄徽章高亮."
    )

    df = _curvature()
    if df.empty:
        st.warning("无 eval_test.json 数据可供曲率框架分类.")
        return

    # 顶部 KPI
    n_frameworks = len(df)
    best_row = df.iloc[0]
    best_R10 = best_row["best_R10"]
    n_r35 = int(df["r35_violation"].sum())
    n_r35_evals = int(df["n_r35_violation"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("曲率机制数", f"{n_frameworks}"), unsafe_allow_html=True)
    with c2:
        v = f"{best_R10*100:.2f}%" if pd.notna(best_R10) else "—"
        st.markdown(kpi_card(f"Best R@10 · {best_row['framework_label']}", v), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("机制含 R35 违规", f"{n_r35}/{n_frameworks}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("R35 违规评估数", f"{n_r35_evals}"), unsafe_allow_html=True)

    st.divider()

    # 按阶段汇总: 目前为止每个 stage 已尝试的曲率机制
    st.subheader("🧩 各阶段已尝试的曲率机制")
    st.caption("按 ★ (写创新) 标记把曲率框架归类到对应 stage. 每阶段列出已尝试机制 + 最佳 R@10.")

    buckets = _stage_innovations()

    def _render_stage(stage_label: str, stage_desc: str) -> None:
        sub = buckets.get(stage_label, pd.DataFrame())
        if sub is None or sub.empty:
            st.info(f"{stage_label} · 暂无曲率机制尝试")
            return
        # 表头
        st.markdown(f"**{stage_label}** · {stage_desc}")
        view = sub.copy()
        view["Best R@10"] = view["best_R10"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )
        view["Evals"] = view["n_eval"].astype(int)
        view["Mechanism"] = view["stage_text"]
        view["Framework"] = view["framework_label"]
        st.dataframe(
            view[["Framework", "Mechanism", "Best R@10", "Evals"]],
            use_container_width=True, hide_index=True,
        )

    # 2x2 网格: stage1 / stage2 / stage3 / stage4
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        _render_stage("stage1", "数据准备层 — 通常无曲率创新 (复用基线)")
    with r1c2:
        _render_stage("stage2", "Stage 2 κ 学习 — K / capmatch / Lipschitz")
    with r2c1:
        _render_stage("stage3", "T5 decoder 训练 — κ frozen→learnable / 新曲率正则项")
    with r2c2:
        _render_stage("stage4", "评估层 — 无曲率创新, 单 ckpt + beam=20")

    # 第 5 类: 无曲率创新 (对照/基线/早期)
    sub = buckets.get("无曲率创新", pd.DataFrame())
    if sub is not None and not sub.empty:
        st.markdown("")
        _render_stage("无曲率创新", "对照 / 基线 / 早期实验 — 不计为曲率改善 (R36)")

    st.divider()

    # 详细表 (按 best_R@10 降序)
    st.subheader("📋 详细列表 (按 best R@10 降序)")
    st.caption("Mechanism 列拆为 stage1/2/3/4 四列, 每格按 'stageN：xxx (写创新)' 格式描述.")
    view = df.copy()
    view["best_R10"] = view["best_R10"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    for col in ("best_R5", "best_R20", "best_NDCG10"):
        if col in view.columns:
            view[col] = view[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    # stage 列以 "stageN：xxx (写创新)" 格式呈现
    view["stage1"] = view["stage1_text"].apply(lambda s: f"stage1：{s}" if pd.notna(s) else "stage1：—")
    view["stage2"] = view["stage2_text"].apply(lambda s: f"stage2：{s}" if pd.notna(s) else "stage2：—")
    view["stage3"] = view["stage3_text"].apply(lambda s: f"stage3：{s}" if pd.notna(s) else "stage3：—")
    view["stage4"] = view["stage4_text"].apply(lambda s: f"stage4：{s}" if pd.notna(s) else "stage4：—")

    disp = view[
        [
            "framework_label",
            "stage1",
            "stage2",
            "stage3",
            "stage4",
            "best_R10",
            "best_R5",
            "best_R20",
            "best_NDCG10",
            "n_eval",
        ]
    ].rename(columns={
        "framework_label": "Framework",
        "best_R10": "Best R@10",
        "best_R5": "Best R@5",
        "best_R20": "Best R@20",
        "best_NDCG10": "Best NDCG@10",
        "n_eval": "Evals",
    })

    st.dataframe(disp, use_container_width=True, hide_index=True)


# ============================================================
# 入口
# ============================================================


PAGES = {
    "🧬 Curvature":      page_curvature,
}


def main() -> None:
    with st.sidebar:
        st.markdown("# 🧬 GeneRec")
        st.caption("实验任务管理 · 本机 127.0.0.1")
        st.markdown("**当前面板** · 🧬 Curvature")
        page = next(iter(PAGES.keys()))
        st.divider()
        if st.button("🔄 立即刷新", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.experimental_rerun()
        st.caption(
            f"自动刷新: 5s (Curvature)\n"
            f"Streamlit 1.30 · 数据缓存 TTL 4s"
        )

    PAGES[page]()


if __name__ == "__main__":
    main()
