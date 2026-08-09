#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneRec 实验任务可视化 — 数据采集层 (纯函数, 无 UI 依赖).

设计原则:
  - R30: 所有路径硬编码, 不读 env / os.environ.get
  - R2: 解析失败 (损坏 JSON) → raise, 缺失合法字段 → 返回 None
  - 禁 fallback: 缺字段不补默认值, 缺文件直接抛 FileNotFoundError
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# ============================================================
# 硬编码路径 (R30)
# ============================================================

REPO = Path("/fs04/ar57/wenyu/GeneRec")
TASKS_DIR = REPO / "tasks"
HISTORY_DIR = REPO / "taskA" / "_history"
VERDICTS_DIR = REPO / "verdicts"
LOOP_MD = REPO / "loop.md"
CLAUDE_MD = REPO / "CLAUDE.md"

STAGE_FILES: tuple[str, ...] = ("stage1.py", "stage2.py", "stage3.py", "stage4.py")

# 评估关键指标
EVAL_METRICS: tuple[str, ...] = ("R@5", "R@10", "R@20", "NDCG@5", "NDCG@10", "NDCG@20")

# ============================================================
# 工具
# ============================================================


def _read_json(path: Path) -> dict[str, Any]:
    """读 JSON 文件. 损坏 → raise."""
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_mtime(path: Path) -> datetime | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except FileNotFoundError:
        return None


# ============================================================
# 1) Tasks Roster — 扫描 tasks/<name>/ 4 阶段脚本存在性
# ============================================================


def _parse_task_description(task_dir: Path) -> dict[str, Any]:
    """从 stage3.py docstring 提取基线/seed/超参 hints."""
    out: dict[str, Any] = {
        "summary": None,
        "baseline": None,
        "seed": None,
        "r_constraints": [],
    }
    candidates = [task_dir / "stage3.py", task_dir / "stage2.py", task_dir / "stage1.py"]
    for c in candidates:
        if not c.is_file():
            continue
        try:
            txt = c.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r'"""([\s\S]*?)"""', txt)
        if m:
            doc = m.group(1).strip().split("\n")[0]
            if doc and not out["summary"]:
                out["summary"] = doc[:200]
        for hit in re.findall(r"seed\s*=\s*(\d+)", txt):
            out["seed"] = int(hit)
            break
        for rule in re.findall(r"R\d+", txt):
            if rule not in out["r_constraints"]:
                out["r_constraints"].append(rule)
        if out["summary"]:
            break
    return out


def collect_tasks() -> pd.DataFrame:
    """扫描 tasks/<name>/, 返回任务元信息表."""
    if not TASKS_DIR.is_dir():
        raise FileNotFoundError(f"missing dir: {TASKS_DIR}")
    rows: list[dict[str, Any]] = []
    for p in sorted(TASKS_DIR.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name == "__pycache__":
            continue
        present: dict[str, bool] = {f: (p / f).is_file() for f in STAGE_FILES}
        meta = _parse_task_description(p)
        log_path = p / "train.log"
        rows.append(
            {
                "task": p.name,
                "has_stage1": present["stage1.py"],
                "has_stage2": present["stage2.py"],
                "has_stage3": present["stage3.py"],
                "has_stage4": present["stage4.py"],
                "scripts": sum(present.values()),
                "has_train_log": log_path.is_file(),
                "train_log_size": log_path.stat().st_size if log_path.is_file() else 0,
                "summary": meta["summary"],
                "seed": meta["seed"],
                "r_rules": ", ".join(meta["r_constraints"][:5]) if meta["r_constraints"] else None,
                "task_dir": str(p),
                "modified_at": _safe_mtime(p),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 2) Experiments — 扫描 taskA/_history/<name>/*/eval_test.json
# ============================================================


def _gather_eval_jsons() -> Iterable[tuple[Path, Path]]:
    """yield (eval_test.json, eval_subdir_or_root).

    覆盖两种布局:
      A) HISTORY/<exp>/eval_test.json           (单次评估)
      B) HISTORY/<exp>/<sub>/eval_test.json     (多次评估, 如 v18/beq_top_best, v15/borda5way/b50 等)
    """
    if not HISTORY_DIR.is_dir():
        raise FileNotFoundError(f"missing dir: {HISTORY_DIR}")
    for exp_dir in sorted(HISTORY_DIR.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        # 类型 A: 根级 eval_test.json
        root_eval = exp_dir / "eval_test.json"
        if root_eval.is_file():
            yield (root_eval, exp_dir)
        # 类型 B: 第一层子目录里有 eval_test.json
        try:
            for child in sorted(exp_dir.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                child_eval = child / "eval_test.json"
                if child_eval.is_file():
                    yield (child_eval, child)
                # 类型 B2: 第二层子目录 (如 stage4_test_on_best/ep50/eval_test.json)
                try:
                    for grand in sorted(child.iterdir()):
                        if not grand.is_dir():
                            continue
                        grand_eval = grand / "eval_test.json"
                        if grand_eval.is_file():
                            yield (grand_eval, grand)
                except OSError:
                    continue
        except OSError:
            continue


def collect_experiments() -> pd.DataFrame:
    """聚合所有 eval_test.json, 返回实验评估表."""
    rows: list[dict[str, Any]] = []
    for eval_path, owner in _gather_eval_jsons():
        try:
            data = _read_json(eval_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            # 缺失 → 跳过; 损坏 → raise (R2)
            if isinstance(exc, FileNotFoundError):
                continue
            raise
        # 由 owner 路径推断 exp / variant / sub_variant label
        try:
            rel = eval_path.relative_to(HISTORY_DIR)
        except ValueError:
            rel = Path(eval_path.name)
        parts = rel.parts
        exp = parts[0] if len(parts) >= 1 else "(unknown)"
        variant = parts[1] if len(parts) >= 2 else "_root"
        sub_variant = parts[2] if len(parts) >= 3 else "_root"

        rows.append(
            {
                "exp": exp,
                "variant": variant,
                "sub_variant": sub_variant,
                "tag": data.get("tag"),
                "done_at": data.get("done_at"),
                "ckpt": data.get("ckpt"),
                "sid_sha256": (data.get("sid_sha256") or "")[:16],
                "n_eval": data.get("n_eval"),
                "t_eval_s": data.get("t_eval_s"),
                **{m: data.get(m) for m in EVAL_METRICS},
                "eval_path": str(eval_path),
                "owner": str(owner),
                "modified_at": _safe_mtime(eval_path),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "R@10" in df.columns:
        df = df.sort_values("R@10", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ============================================================
# 3) Verdicts — verdicts/<iid>/*.{json,md}
# ============================================================


_VERDICT_KEYS_FAIL = re.compile(r"FAIL|NO[-_]?GO|NO‑GO", re.IGNORECASE)
_VERDICT_KEYS_PASS = re.compile(r"\bPASS\b|GO", re.IGNORECASE)


def _summarize_verdict(text: str) -> str:
    """提首行非空内容."""
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:240]
    return ""


def _classify_verdict(verdict_line: str) -> str:
    if not verdict_line:
        return "?"
    if _VERDICT_KEYS_FAIL.search(verdict_line):
        return "FAIL"
    if _VERDICT_KEYS_PASS.search(verdict_line):
        return "PASS"
    return "?"


def collect_verdicts() -> pd.DataFrame:
    """聚合 verdicts/<iid>/*.json 或 *.md."""
    if not VERDICTS_DIR.is_dir():
        raise FileNotFoundError(f"missing dir: {VERDICTS_DIR}")
    rows: list[dict[str, Any]] = []
    for vdir in sorted(VERDICTS_DIR.iterdir(), key=lambda p: p.name):
        if not vdir.is_dir() or vdir.name.startswith(".") or vdir.name == "_misc":
            continue
        try:
            iid = int(vdir.name)
        except ValueError:
            continue
        files = sorted([p for p in vdir.iterdir() if p.is_file() and p.suffix in (".json", ".md")])
        if not files:
            continue
        # 优先级: 含 'verdict' 字样的优先, 否则最新 mtime
        chosen = next((p for p in files if "verdict" in p.stem.lower()), None) or max(files, key=lambda p: p.stat().st_mtime)
        try:
            text = _read_text(chosen)
        except FileNotFoundError:
            continue
        summary = _summarize_verdict(text)
        # 数据提取
        issue: Any = None
        gate: Any = None
        verdict_line: Any = None
        evaluated_at: Any = None
        r_line: Any = None
        title: Any = None
        if chosen.suffix == ".json":
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    issue = data.get("issue")
                    gate = data.get("gate")
                    verdict_line = data.get("verdict") or data.get("final_verdict")
                    evaluated_at = data.get("evaluated_at")
                    title = data.get("title") or data.get("step")
                    r_line = data.get("r_constraints") if isinstance(data.get("r_constraints"), str) else None
            except json.JSONDecodeError:
                pass
        rows.append(
            {
                "issue_iid": iid,
                "file": chosen.name,
                "issue": issue,
                "gate": gate,
                "title": title,
                "verdict": verdict_line,
                "status": _classify_verdict(str(verdict_line)) if verdict_line else _classify_verdict(summary),
                "evaluated_at": evaluated_at,
                "summary": summary,
                "file_path": str(chosen),
                "modified_at": _safe_mtime(chosen),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("issue_iid", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# 4) Running — 当前活跃 Python 训练进程 (本项目相关)
# ============================================================


_GENE_REC_CMDLINE_HINTS = (
    "stage3_train_pure_t5",
    "stage4_eval_pure_t5",
    "gen_codebook",
    "tasks/",
    "stage1.py",
    "stage2.py",
    "stage3.py",
    "stage4.py",
    "torch.distributed.run",
    "train_HG-Rec",
    "train_hrqvae",
    "stage3/stage3_train",
    "stage4/stage4_eval",
)


def _is_genrec_proc(cmdline: str) -> bool:
    return any(hint in cmdline for hint in _GENE_REC_CMDLINE_HINTS)


def _read_proc_stat(pid: int) -> dict[str, Any] | None:
    """读 /proc/<pid>/stat + cmdline + cwd, 返回 None 当进程已退出."""
    try:
        stat = Path(f"/proc/{pid}").joinpath("stat").read_text(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    parts = stat.rsplit(")", 1)[-1].split()  # 跳过含空格 comm
    # 字段 22 starttime (clock ticks since boot)
    try:
        start_ticks = int(parts[19])  # 22 - 3 offset
    except (IndexError, ValueError):
        start_ticks = None
    try:
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        cmdline = cmdline_path.read_bytes().decode("utf-8", errors="replace").replace("\x00", " ").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        cwd = None
    return {
        "start_ticks": start_ticks,
        "cmdline": cmdline,
        "cwd": cwd,
    }


_BOOTTIME_CACHE: int | None = None


def _bottime() -> int:
    global _BOOTTIME_CACHE
    if _BOOTTIME_CACHE is None:
        try:
            _BOOTTIME_CACHE = int(Path("/proc/stat").read_text().split("btime ", 1)[1].split()[0])
        except Exception:
            _BOOTTIME_CACHE = int(datetime.now(tz=timezone.utc).timestamp())
    return _BOOTTIME_CACHE


def collect_running() -> pd.DataFrame:
    """枚举当前本机 GeneRec 相关 python3 训练进程."""
    rows: list[dict[str, Any]] = []
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        raise FileNotFoundError("/proc not available")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        info = _read_proc_stat(pid)
        if info is None:
            continue
        if not _is_genrec_proc(info["cmdline"]):
            continue
        if info["start_ticks"] is None:
            started_at = None
        else:
            started_at = datetime.fromtimestamp(_bottime() + info["start_ticks"] / 100.0, tz=timezone.utc)
        rows.append(
            {
                "pid": pid,
                "started_at": started_at,
                "cmdline_short": info["cmdline"][:160] + ("..." if len(info["cmdline"]) > 160 else ""),
                "cwd": info["cwd"],
                "cmdline_full": info["cmdline"],
            }
        )
    return pd.DataFrame(rows).sort_values("pid", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["pid", "started_at", "cmdline_short", "cwd", "cmdline_full"]
    )


# ============================================================
# 5) GPU — nvidia-smi
# ============================================================


def collect_gpu() -> pd.DataFrame:
    """调用 nvidia-smi 拿 GPU util/mem. 失败 → raise (R2)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("nvidia-smi not in PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi returned {exc.returncode}: {exc.stderr}") from exc

    rows: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 6:
            continue
        rows.append(
            {
                "gpu_index": int(cells[0]),
                "name": cells[1],
                "util_pct": int(cells[2]) if cells[2].isdigit() else None,
                "mem_used_mb": int(cells[3]) if cells[3].isdigit() else None,
                "mem_total_mb": int(cells[4]) if cells[4].isdigit() else None,
                "temp_c": int(cells[5]) if cells[5].isdigit() else None,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 6) Train curve helper — 取一条 train_curve.json
# ============================================================


def load_train_curve(eval_path: str) -> pd.DataFrame | None:
    """根据 eval_test.json 路径推断同一 _history 子目录, 返回 train_curve.json DataFrame."""
    p = Path(eval_path)
    # 解析: .../_history/<exp>/<variant>/eval_test.json → 同级找 train_curve.json
    sibling = p.parent / "train_curve.json"
    if not sibling.is_file():
        # 向上找
        cur = p.parent
        for _ in range(3):
            cur = cur.parent
            cand = cur / "train_curve.json"
            if cand.is_file():
                sibling = cand
                break
    if not sibling.is_file():
        return None
    try:
        arr = json.loads(sibling.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    rows: list[dict[str, Any]] = []
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        kappas = entry.get("kappas") or []
        rows.append(
            {
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry.get("loss"),
                "recon_loss": entry.get("recon_loss"),
                "rq_loss": entry.get("rq_loss"),
                "kappa_mean": float(sum(kappas) / len(kappas)) if kappas else None,
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 7) Curvature Framework Classifier (R36 视角)
# ============================================================

import re as _re


def classify_curvature(row: pd.Series | dict) -> dict[str, Any]:
    """把一条 eval_test.json 记录分类到 (曲率机制 key, 显示名, 机制说明, R35 违规).

    R35 违规识别: tag/exp/variant 含 borda/b<N>way/ensemble 字样 → 单 ckpt + beam=20 硬约束破坏.
    R36 曲率机制识别 (按优先级): v85p/q SID → branch_curvature/kmeans → geo_residual → capmatch → HAB ΔD → HAB residual → pureT5 → baseline.
    """
    if isinstance(row, dict):
        exp = row.get("exp", "") or ""
        tag = row.get("tag", "") or ""
        variant = row.get("variant", "") or ""
        ckpt = row.get("ckpt", "") or ""
    else:
        exp = str(row.get("exp", "") or "")
        tag = str(row.get("tag", "") or "")
        variant = str(row.get("variant", "") or "")
        ckpt = str(row.get("ckpt", "") or "")

    text = f"{exp} {tag} {variant}".lower()

    # ----- R35 violation detection -----
    # borda / multi-way / ensemble → 单 ckpt + beam=20 硬约束破坏
    # 排除 "beam20" / "beam30" (合法合规 beam size 不算违规)
    text_lower = text.lower()
    borda_hit = "borda" in text_lower or "bord_" in text_lower
    multiway_hit = bool(_re.search(r"\b\d+way\b", text)) or bool(_re.search(r"b\d+way", text_lower))
    ensemble_hit = "ensembl" in text_lower
    r35_violation = borda_hit or multiway_hit or ensemble_hit

    # ----- Curvature framework classification (priority order) -----
    # 注意: framework_label / mechanism 仅允许纯文字描述, 不出现 v\d+ / Issue #\d+ /
    #       K=[..] / λ=[..] / Stage \d / 任务 #\d 等数字字母混合的版本号与配置标识
    if _re.search(r"v85p", text) or "/v85p" in ckpt or "v85p" in ckpt:
        return {
            "framework_key": "v85p_SID",
            "framework_label": "Lipschitz SID + learnable κ",
            "mechanism": "Lipschitz 约束 + learnable κ + shared decode",
            "r35_violation": r35_violation,
        }
    if _re.search(r"v85q", text) or "v85q" in ckpt:
        return {
            "framework_key": "v85q_SID",
            "framework_label": "Lipschitz SID 变体 (略改 κ 初值)",
            "mechanism": "Lipschitz SID 变体 (略改 κ 初值/正则)",
            "r35_violation": r35_violation,
        }
    if "branch_curvature" in text:
        return {
            "framework_key": "v18_branch_quantile",
            "framework_label": "分桶曲率 (quantile 桶策略)",
            "mechanism": "分桶曲率 (quantile 桶策略)",
            "r35_violation": r35_violation,
        }
    if "branch_kmeans" in text:
        return {
            "framework_key": "v21_branch_kmeans",
            "framework_label": "分桶曲率 (kmeans 桶策略)",
            "mechanism": "分桶曲率 (kmeans 桶策略)",
            "r35_violation": r35_violation,
        }
    if "geo_residual" in text:
        return {
            "framework_key": "v20_geo_residual",
            "framework_label": "几何残差 MLP (per-layer κ 注入)",
            "mechanism": "per-layer 几何残差 MLP (hidden state 注入 κ_e)",
            "r35_violation": r35_violation,
        }
    if "hab_delta" in text or "ΔD" in tag or "ΔD" in exp or "delta_d" in text:
        return {
            "framework_key": "hab_delta_D",
            "framework_label": "HAB ΔD 调制 (距离差调制注意力)",
            "mechanism": "HAB ΔD 调制 (距离差调制注意力)",
            "r35_violation": r35_violation,
        }
    if _re.search(r"v15_capmatch|capmatch_1000ep|hyp_v2_capmatch", text) or "v15" in text and "capmatch" in text:
        return {
            "framework_key": "v15_capmatch",
            "framework_label": "Learnable κ + capmatch 容量匹配",
            "mechanism": "learnable κ + capmatch 容量匹配",
            "r35_violation": r35_violation,
        }
    if "/v74" in ckpt or "_v74" in ckpt or "hab_lambda" in text or "hyperbolic_attn" in text:
        return {
            "framework_key": "hab_residual_v74",
            "framework_label": "HAB residual (负 α 强约束)",
            "mechanism": "HAB residual 负 α 强约束 + weight decay + dropout 正则",
            "r35_violation": r35_violation,
        }
    if "puret5" in text or "pure_t5" in text:
        return {
            "framework_key": "pureT5_no_curv",
            "framework_label": "纯 T5 对照 (无曲率注入)",
            "mechanism": "纯 T5 训练, 移除所有曲率注入 (对照组 — 隔离曲率贡献)",
            "r35_violation": r35_violation,
        }
    if "baseline" in text or "#84" in tag:
        return {
            "framework_key": "baseline_hgrec",
            "framework_label": "HG-Rec 基线 (任务基线对照)",
            "mechanism": "HG-Rec 默认架构 — 任务基线对照",
            "r35_violation": r35_violation,
        }
    return {
        "framework_key": "other_legacy",
        "framework_label": "其他 / 早期实验",
        "mechanism": "无法明确归类的早期实验 (调参/调试版本, 不计为曲率改善)",
        "r35_violation": r35_violation,
    }


# ============================================================
# 每曲率框架的 4 阶段任务显示 ("stage1：xxx (写创新)" 格式)
# ============================================================
# 格式: (stage1, stage2, stage3, stage4)
#   - ★ (写创新) 标记本框架的曲率创新所在阶段
#   - 其他阶段标注 (复用基线) 或对应基线行为
#   - 未覆盖框架 → 4 个阶段统一 xxx (写创新)
#   - 注意: 不含数字字母混合的版本号 / 配置标识 (保持纯文字描述)


_STAGE_BREAKDOWN: dict[str, tuple[str, str, str, str]] = {
    "v85p_SID": (
        "数据准备 (复用基线)",
        "Lipschitz 约束 + learnable κ + shared decode ★ (写创新)",
        "T5 decoder 训练 (复用基线, κ frozen)",
        "单 ckpt + beam=20 评估",
    ),
    "v85q_SID": (
        "数据准备 (复用基线)",
        "Lipschitz SID 变体 (略改 κ 初值/正则) ★ (写创新)",
        "T5 decoder 训练 (复用基线)",
        "单 ckpt + beam=20 评估",
    ),
    "v18_branch_quantile": (
        "数据准备 (复用基线)",
        "κ (复用基线)",
        "分桶曲率 (quantile 桶策略) ★ (写创新)",
        "单 ckpt + beam=20 评估",
    ),
    "v21_branch_kmeans": (
        "数据准备 (复用基线)",
        "κ (复用基线)",
        "分桶曲率 (kmeans 桶策略) ★ (写创新)",
        "单 ckpt + beam=20 评估",
    ),
    "v20_geo_residual": (
        "数据准备 (复用基线)",
        "κ (复用基线)",
        "per-layer 几何残差 MLP (hidden state 注入 κ_e) ★ (写创新)",
        "单 ckpt + beam=20 评估",
    ),
    "hab_delta_D": (
        "数据准备 (复用基线)",
        "κ (复用基线)",
        "HAB ΔD 调制 (距离差调制注意力) ★ (写创新)",
        "单 ckpt + beam=20 评估",
    ),
    "v15_capmatch": (
        "数据准备 (复用基线)",
        "learnable κ + capmatch 容量匹配 ★ (写创新)",
        "T5 decoder 训练 (复用基线)",
        "单 ckpt + beam=20 评估",
    ),
    "hab_residual_v74": (
        "数据准备 (复用基线)",
        "κ (复用基线)",
        "HAB residual (负 α 强约束 + WD + dropout 正则) ★ (写创新)",
        "单 ckpt + beam=20 评估",
    ),
    "pureT5_no_curv": (
        "数据准备",
        "不做曲率学习 (无 κ)",
        "纯 T5 训练 (移除所有曲率注入)",
        "单 ckpt + beam=20 评估",
    ),
    "baseline_hgrec": (
        "数据准备 (基线)",
        "基线 κ",
        "基线 T5 decoder",
        "基线评估",
    ),
}

_STAGE_BREAKDOWN_DEFAULT = (
    "xxx (写创新)",
    "xxx (写创新)",
    "xxx (写创新)",
    "xxx (写创新)",
)


def _stage_breakdown(framework_key: str) -> tuple[str, str, str, str]:
    """按 framework_key 取 4 阶段任务描述. 未命中 → 全部 xxx (写创新)."""
    return _STAGE_BREAKDOWN.get(framework_key, _STAGE_BREAKDOWN_DEFAULT)


def collect_curvature_frameworks() -> pd.DataFrame:
    """按曲率机制聚合所有 eval_test.json, 每个机制返回一行 (best_R@10 + 详细 metrics).

    列: framework_key, framework_label, mechanism, r35_violation,
        best_R10, best_R5, best_R20, best_NDCG10, n_eval,
        best_tag, best_exp, best_variant, best_ckpt, best_done_at,
        stage1_text, stage2_text, stage3_text, stage4_text
    """
    df = _experiments_df() if hasattr(__import__("__main__"), "__name__") else collect_experiments()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "framework_key", "framework_label", "mechanism", "r35_violation",
                "best_R10", "best_R5", "best_R20", "best_NDCG10", "n_eval",
                "best_tag", "best_exp", "best_variant", "best_ckpt", "best_done_at",
                "stage1_text", "stage2_text", "stage3_text", "stage4_text",
            ]
        )

    # 给每行加 curvature classification
    classifications = df.apply(classify_curvature, axis=1, result_type="expand")
    df = pd.concat([df, classifications], axis=1)

    # 按 framework_key 聚合 (不用 r35_violation 做 key, 避免相同机制被拆两行)
    grouped = (
        df.groupby(["framework_key", "framework_label", "mechanism"], dropna=False)
        .apply(_agg_best_R10)
        .reset_index()
    )
    # 计算每个机制中有多少条 R35 违规
    r35_counts = (
        df.groupby(["framework_key"], dropna=False)["r35_violation"]
        .sum()
        .astype(int)
        .rename("n_r35_violation")
        .reset_index()
    )
    grouped = grouped.merge(r35_counts, on="framework_key", how="left")
    grouped["n_r35_violation"] = grouped["n_r35_violation"].fillna(0).astype(int)
    # r35_violation 字段 (聚合层) = 是否有任何 R35 违规
    grouped["r35_violation"] = grouped["n_r35_violation"] > 0

    # 附 4 阶段任务显示
    breakdowns = grouped["framework_key"].apply(_stage_breakdown)
    grouped["stage1_text"] = [b[0] for b in breakdowns]
    grouped["stage2_text"] = [b[1] for b in breakdowns]
    grouped["stage3_text"] = [b[2] for b in breakdowns]
    grouped["stage4_text"] = [b[3] for b in breakdowns]

    grouped = grouped.sort_values("best_R10", ascending=False, na_position="last").reset_index(drop=True)
    return grouped


def collect_stage_innovations() -> dict[str, pd.DataFrame]:
    """按 ★ (写创新) 标记把曲率框架归类到 (stage1, stage2, stage3, stage4, 无曲率创新) 5 个桶.

    返回 dict[stage_label, DataFrame], 每行: framework_label / best_R10 / n_eval / stage_text.
    按 best_R@10 降序.
    """
    df = collect_curvature_frameworks()
    if df.empty:
        empty = pd.DataFrame(columns=["framework_label", "best_R10", "n_eval", "stage_text"])
        return {f"stage{i}": empty.copy() for i in range(1, 5)} | {"无曲率创新": empty.copy()}

    stage_cols = {1: "stage1_text", 2: "stage2_text", 3: "stage3_text", 4: "stage4_text"}
    buckets: dict[str, pd.DataFrame] = {f"stage{i}": [] for i in range(1, 5)}  # type: ignore
    buckets["无曲率创新"] = []  # type: ignore

    for _, r in df.iterrows():
        placed = False
        for stage_n, col in stage_cols.items():
            txt = r.get(col) or ""
            if "★" in txt:
                buckets[f"stage{stage_n}"].append({
                    "framework_label": r["framework_label"],
                    "best_R10": r["best_R10"],
                    "n_eval": int(r["n_eval"]),
                    "stage_text": txt,
                })
                placed = True
                break
        if not placed:
            buckets["无曲率创新"].append({
                "framework_label": r["framework_label"],
                "best_R10": r["best_R10"],
                "n_eval": int(r["n_eval"]),
                "stage_text": "无 ★ 标记的曲率创新 (对照 / 基线 / 早期实验)",
            })

    out: dict[str, pd.DataFrame] = {}
    for k, rows in buckets.items():
        sub = pd.DataFrame(rows)
        if sub.empty:
            out[k] = sub
        else:
            out[k] = sub.sort_values("best_R10", ascending=False, na_position="last").reset_index(drop=True)
    return out


def _experiments_df():
    return collect_experiments()


def _agg_best_R10(group: pd.DataFrame) -> pd.Series:
    """取组内 R@10 最高的那一行, 返回聚合指标."""
    s = group.dropna(subset=["R@10"]).sort_values("R@10", ascending=False)
    if s.empty:
        return pd.Series(
            {
                "n_eval": len(group),
                "best_R10": None,
                "best_R5": None,
                "best_R20": None,
                "best_NDCG10": None,
                "best_tag": None,
                "best_exp": None,
                "best_variant": None,
                "best_ckpt": None,
                "best_done_at": None,
            }
        )
    top = s.iloc[0]
    return pd.Series(
        {
            "n_eval": len(group),
            "best_R10": top.get("R@10"),
            "best_R5": top.get("R@5"),
            "best_R20": top.get("R@20"),
            "best_NDCG10": top.get("NDCG@10"),
            "best_tag": top.get("tag"),
            "best_exp": top.get("exp"),
            "best_variant": top.get("variant"),
            "best_ckpt": (top.get("ckpt") or "").split("/")[-1] if top.get("ckpt") else None,
            "best_done_at": top.get("done_at"),
        }
    )


# ============================================================
# 7) Overview KPI 拼接
# ============================================================


def overview_kpis() -> dict[str, Any]:
    tasks = collect_tasks()
    exps = collect_experiments()
    vrds = collect_verdicts()
    running = collect_running()
    gpu = collect_gpu()
    top10 = []
    if not exps.empty and "R@10" in exps.columns:
        top10 = (
            exps[["exp", "variant", "R@10", "tag", "done_at"]]
            .dropna(subset=["R@10"])
            .head(10)
            .to_dict("records")
        )
    return {
        "tasks_count": int(len(tasks)),
        "experiments_count": int(len(exps)),
        "verdicts_count": int(len(vrds)),
        "running_count": int(len(running)),
        "gpu_count": int(len(gpu)),
        "pass_count": int((vrds["status"].astype(str) == "PASS").sum()) if not vrds.empty else 0,
        "fail_count": int((vrds["status"].astype(str) == "FAIL").sum()) if not vrds.empty else 0,
        "top10_by_R10": top10,
    }
