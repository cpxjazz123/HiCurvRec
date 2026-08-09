#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneRec 任务管理 Web Dashboard.

设计原则 (R2 禁 fallback):
  - 所有数据源路径硬编码进模块常量, 不读 env
  - 解析失败必须 raise, 缺失合法字段返回 None
  - 单 ckpt + beam=20 评估是 R35 硬约束, UI 提示
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, abort

# ---------- 硬编码路径 (R30 强化) ----------
REPO = Path("/fs04/ar57/wenyu/GeneRec")
TASKS_DIR = REPO / "tasks"
VERDICTS_DIR = REPO / "verdicts"
LOOP_MD = REPO / "loop.md"
CLAUDE_MD = REPO / "CLAUDE.md"

# ---------- Flask ----------
app = Flask(__name__, template_folder="templates", static_folder="static")


# ============================================================
# 工具函数
# ============================================================
def _read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    """读文件, 限制大小避免 OOM."""
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes] + "\n\n... [truncated]"
    return path.read_text(encoding="utf-8", errors="replace")


def _list_immediate_dirs(parent: Path) -> list[Path]:
    """列直接子目录, 跳过隐藏 / __pycache__."""
    if not parent.is_dir():
        raise FileNotFoundError(f"missing dir: {parent}")
    return sorted(
        [p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"],
        key=lambda p: p.name,
    )


def _parse_md_status(text: str) -> str:
    """从 verdict markdown 解析状态 (GO / NO-GO / ...). 找不到返回 'UNKNOWN'."""
    m = re.search(r"状态[:：]\s*\*?\*?([A-Z\-/ ]{2,30})", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\*\*(GO|NO-GO|PASS|FAIL|✅|⛔|🟡|🟢|🔴)\*\*", text)
    if m:
        return m.group(1)
    return "UNKNOWN"


def _parse_r10(text: str) -> dict[str, float | None]:
    """从文本提取 test_R@10 / valid_R@10 等数字."""
    out: dict[str, float | None] = {"test_r10": None, "valid_r10": None}
    for label in ("test_R@10", "test R@10", "test_R10"):
        m = re.search(rf"{re.escape(label)}\s*[=:≈]\s*\*?\*?(\d+\.\d+)", text)
        if m:
            out["test_r10"] = float(m.group(1))
            break
    for label in ("valid_R@10", "valid R@10", "valid_R10"):
        m = re.search(rf"{re.escape(label)}\s*[=:≈]\s*\*?\*?(\d+\.\d+)", text)
        if m:
            out["valid_r10"] = float(m.group(1))
            break
    return out


def _parse_loop_active(text: str) -> list[dict[str, Any]]:
    """从 loop.md 解析 '当前迭代' 段下的版本表 / 目标条目.

    支持两种格式:
    1. 表格: | 版本 | 改动 | test_R@10 | valid_R@10 |
    2. bullet: - **v18 NEW HIGH** (test_R@10=0.1011, 描述...)
    """
    items: list[dict[str, Any]] = []
    m = re.search(r"## 当前迭代[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
    if not m:
        return items
    section = m.group(1)
    # 1) 表格
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| 版本"):
            continue
        cells = [c.strip().replace("**", "").replace("`", "") for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        version = cells[0]
        if not version or version == "版本":
            continue
        items.append({
            "version": version,
            "change": cells[1] if len(cells) > 1 else "",
            "test_r10": cells[2] if len(cells) > 2 else None,
            "valid_r10": cells[3] if len(cells) > 3 else None,
        })
    # 2) bullet (兜底, 表格解析为空时)
    if not items:
        for line in section.splitlines():
            line = line.strip()
            if not line.startswith("- **") and not line.startswith("* **"):
                continue
            m2 = re.search(r"\*\*([^*]+)\*\*\s*\(([^)]*)\)", line)
            if not m2:
                continue
            version = m2.group(1).strip()
            desc = m2.group(2).strip()
            r10 = _parse_r10(desc)
            items.append({
                "version": version,
                "change": desc,
                "test_r10": str(r10["test_r10"]) if r10["test_r10"] is not None else None,
                "valid_r10": str(r10["valid_r10"]) if r10["valid_r10"] is not None else None,
            })
    return items


# ============================================================
# 任务解析
# ============================================================
def _parse_task(task_dir: Path) -> dict[str, Any]:
    """解析单个 task 目录."""
    name = task_dir.name
    stages: dict[str, dict[str, Any]] = {}
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        # 允许 stage4_beam20.py 这种变体命名
        candidates = [task_dir / f"{stage}.py"] + sorted(task_dir.glob(f"{stage}*.py"))
        py_files = [c for c in candidates if c.is_file()]
        if py_files:
            py = py_files[0]
            try:
                txt = _read_text(py, max_bytes=50_000)
            except FileNotFoundError:
                txt = ""
            stages[stage] = {
                "exists": True,
                "file": py.name,
                "size": py.stat().st_size,
                "mtime": py.stat().st_mtime,
                "head": "\n".join(txt.splitlines()[:25]),
                "line_count": len(txt.splitlines()),
            }
        else:
            stages[stage] = {"exists": False}

    # 关联 verdict: 同时扫数字 iid 目录 + 命名目录 (如 v18_branch_curvature_beam20)
    related_verdicts: list[dict[str, Any]] = []
    for vd in _list_immediate_dirs(VERDICTS_DIR):
        for f in vd.iterdir():
            if not f.is_file() or f.suffix not in (".md", ".json", ".txt"):
                continue
            try:
                content_head = _read_text(f, max_bytes=8000)
            except FileNotFoundError:
                continue
            if name in f.name or name in content_head or name in vd.name:
                related_verdicts.append({
                    "iid": int(vd.name) if vd.name.isdigit() else None,
                    "verdict_dir": vd.name,
                    "file": f.name,
                    "status": _parse_md_status(content_head),
                    "r10": _parse_r10(content_head),
                })

    return {
        "name": name,
        "path": str(task_dir),
        "stages": stages,
        "stage_count": sum(1 for s in stages.values() if s["exists"]),
        "related_verdicts": related_verdicts,
        "mtime": max(
            (s["mtime"] for s in stages.values() if s.get("mtime")),
            default=task_dir.stat().st_mtime,
        ),
    }


# ============================================================
# Verdict 解析
# ============================================================
def _parse_verdict_dir(vd: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for f in sorted(vd.iterdir()):
        if not f.is_file():
            continue
        try:
            content = _read_text(f, max_bytes=200_000)
        except FileNotFoundError:
            continue
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "mtime": f.stat().st_mtime,
            "status": _parse_md_status(content),
            "r10": _parse_r10(content),
            "content": content,
        })
    statuses = [fl["status"] for fl in files if fl["status"] != "UNKNOWN"]
    agg_status = max(set(statuses), key=statuses.count) if statuses else "UNKNOWN"
    return {
        "iid": int(vd.name) if vd.name.isdigit() else None,
        "path": str(vd),
        "file_count": len(files),
        "agg_status": agg_status,
        "files": files,
    }


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    """Dashboard 总览."""
    tasks = [_parse_task(p) for p in _list_immediate_dirs(TASKS_DIR)]
    verdicts = [_parse_verdict_dir(p) for p in _list_immediate_dirs(VERDICTS_DIR) if p.name.isdigit()]

    loop_text = _read_text(LOOP_MD)
    active = _parse_loop_active(loop_text)

    # 统计
    status_counts: dict[str, int] = {}
    for v in verdicts:
        status_counts[v["agg_status"]] = status_counts.get(v["agg_status"], 0) + 1

    stage_coverage: dict[str, int] = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0}
    for t in tasks:
        for s in stage_coverage:
            if t["stages"].get(s, {}).get("exists"):
                stage_coverage[s] += 1

    # 最新 verdict
    recent_verdicts = sorted(
        verdicts, key=lambda v: max((f["mtime"] for f in v["files"]), default=0), reverse=True
    )[:5]

    return jsonify({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_total": len(tasks),
        "verdict_total": len(verdicts),
        "status_counts": status_counts,
        "stage_coverage": stage_coverage,
        "active_iterations": active,
        "recent_verdicts": [
            {
                "iid": v["iid"],
                "agg_status": v["agg_status"],
                "file_count": v["file_count"],
                "top_file": v["files"][0]["name"] if v["files"] else None,
                "r10": v["files"][0]["r10"] if v["files"] else None,
                "mtime": max((f["mtime"] for f in v["files"]), default=0),
            }
            for v in recent_verdicts
        ],
    })


@app.route("/api/tasks")
def api_tasks():
    tasks = [_parse_task(p) for p in _list_immediate_dirs(TASKS_DIR)]
    # 排序: 按 mtime desc
    tasks.sort(key=lambda t: t["mtime"], reverse=True)
    # 转 mtime 为可读
    for t in tasks:
        t["mtime_iso"] = datetime.fromtimestamp(t["mtime"]).isoformat(timespec="seconds")
        t.pop("mtime", None)
    return jsonify({"tasks": tasks})


@app.route("/api/tasks/<name>")
def api_task_detail(name: str):
    target = TASKS_DIR / name
    if not target.is_dir():
        abort(404, description=f"task not found: {name}")
    data = _parse_task(target)
    data["mtime_iso"] = datetime.fromtimestamp(data["mtime"]).isoformat(timespec="seconds")
    data.pop("mtime", None)
    return jsonify(data)


@app.route("/api/tasks/<name>/script/<stage>")
def api_task_script(name: str, stage: str):
    """读单个 stage 脚本全文."""
    if stage not in {"stage1", "stage2", "stage3", "stage4"}:
        abort(400, description="invalid stage")
    target = TASKS_DIR / name
    if not target.is_dir():
        abort(404)
    candidates = [target / f"{stage}.py"] + sorted(target.glob(f"{stage}*.py"))
    py_files = [c for c in candidates if c.is_file()]
    if not py_files:
        abort(404, description=f"no script for {stage}")
    return jsonify({
        "task": name,
        "stage": stage,
        "file": py_files[0].name,
        "content": _read_text(py_files[0], max_bytes=500_000),
    })


@app.route("/api/verdicts")
def api_verdicts():
    # 包含数字 iid 目录 + 命名目录 (如 v18_branch_curvature_beam20)
    verdicts = [
        _parse_verdict_dir(p) for p in _list_immediate_dirs(VERDICTS_DIR)
        if p.name.isdigit() or p.name.startswith("v")
    ]
    # 排序: 数字 iid desc, 命名目录按 mtime desc
    numeric_vs = [v for v in verdicts if v["iid"] is not None]
    named_vs = [v for v in verdicts if v["iid"] is None]
    numeric_vs.sort(key=lambda v: v["iid"], reverse=True)
    named_vs.sort(key=lambda v: max((f["mtime"] for f in v["files"]), default=0), reverse=True)
    verdicts = numeric_vs + named_vs
    # 精简返回 (不返回 content)
    slim = []
    for v in verdicts:
        slim.append({
            "iid": v["iid"],
            "path": v["path"],
            "agg_status": v["agg_status"],
            "file_count": v["file_count"],
            "files": [
                {"name": f["name"], "size": f["size"], "mtime": f["mtime"],
                 "status": f["status"], "r10": f["r10"]}
                for f in v["files"]
            ],
        })
    return jsonify({"verdicts": slim})


@app.route("/api/verdicts/<iid>")
def api_verdict_detail(iid: str):
    if not iid.isdigit():
        abort(400)
    target = VERDICTS_DIR / iid
    if not target.is_dir():
        abort(404)
    data = _parse_verdict_dir(target)
    return jsonify(data)


@app.route("/api/verdicts/<iid>/file/<path:filename>")
def api_verdict_file(iid: str, filename: str):
    if not iid.isdigit():
        abort(400)
    target = VERDICTS_DIR / iid / filename
    # 防穿越
    if VERDICTS_DIR / iid not in target.resolve().parents:
        abort(400)
    if not target.is_file():
        abort(404)
    return jsonify({"content": _read_text(target, max_bytes=500_000)})


@app.route("/api/loop")
def api_loop():
    text = _read_text(LOOP_MD)
    return jsonify({
        "content": text,
        "active": _parse_loop_active(text),
        "mtime": LOOP_MD.stat().st_mtime,
        "mtime_iso": datetime.fromtimestamp(LOOP_MD.stat().st_mtime).isoformat(timespec="seconds"),
    })


@app.route("/api/claudemd")
def api_claudemd():
    return jsonify({"content": _read_text(CLAUDE_MD, max_bytes=200_000)})


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "tasks_dir_exists": TASKS_DIR.is_dir(),
        "verdicts_dir_exists": VERDICTS_DIR.is_dir(),
        "loop_md_exists": LOOP_MD.is_file(),
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"[webdash] http://{args.host}:{args.port}  (repo: {REPO})")
    app.run(host=args.host, port=args.port, debug=False)
