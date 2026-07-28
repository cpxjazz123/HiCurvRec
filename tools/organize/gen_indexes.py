"""gen_indexes.py — Phase 8: 生成 descriptions/, verdicts/, products/, TASKS_INDEX.md 索引。"""
import json
import pathlib

REPO = pathlib.Path("/home/wlia0047/ar57/wenyu/GeneRec")
TA = REPO / "GRID/task_artifacts"  # legacy path; only for results/loop/docs/slurm below

# After 2026-07-16 reorganization, 4 categories live at REPO root.
DESCRIPTIONS = REPO / "descriptions"
VERDICTS = REPO / "verdicts"
PRODUCTS = REPO / "products"
SCRIPTS = REPO / "scripts"


def scan(root):
    if not root.exists():
        return {}
    out = {}
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix == '.md' and p.name != 'README.md':
            out[p.stem] = p
    return out


def main():
    keep = json.loads((REPO / "tools/organize/keep_set.json").read_text())

    desc = scan(DESCRIPTIONS)
    verd = scan(VERDICTS)
    prod = PRODUCTS
    prod_ids = set()  # numeric NEW ids
    if prod.exists():
        for d in prod.iterdir():
            if d.is_dir():
                prod_ids.add(d.name.replace('task', ''))

    n_desc = len(keep)
    n_verd = len(verd)
    n_prod = len(prod_ids)
    n_full = sum(1 for k in keep
                 if k['new_id'] in prod_ids
                 and any(k['filename'].replace('.md', '') in v for v in verd))
    print(f"keep={n_desc} descriptions={n_desc} verdicts={n_verd} products={n_prod} full3={n_full}")

    # descriptions/README.md
    lines = [
        "# Task Descriptions",
        "",
        f"**{n_desc} tasks** with active products (verdicts and/or execution artifacts).",
        "",
        "| new_id | filename | has_products | has_verdict |",
        "|---:|---|:---:|:---:|",
    ]
    for k in keep:
        nid = k['new_id']
        has_prod = nid in prod_ids
        verd_prefix = f"task{nid}_"
        has_verd = any(v.startswith(verd_prefix) for v in verd.keys())
        lines.append(f"| {nid} | `{k['filename']}` | {'✓' if has_prod else '✗'} | {'✓' if has_verd else '✗'} |")
    lines += ["", "---", "", "Templates:", "- `task.md` — generic task template"]
    (DESCRIPTIONS / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  ✓ descriptions/README.md ({n_desc} entries)")

    # verdicts/README.md
    lines = [
        "# Task Verdicts",
        "",
        f"**{n_verd} verdict files** from completed tasks.",
        "",
        "| file | task |",
        "|---|---|",
    ]
    for v in sorted(verd.keys()):
        m = v.split('_', 1)
        nid = m[0].replace('task', '') if m else '?'
        lines.append(f"| `{v}.md` | {nid} |")
    (VERDICTS / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  ✓ verdicts/README.md ({n_verd} entries)")

    # products/README.md
    lines = [
        "# Task Products",
        "",
        f"**{n_prod} tasks** with execution artifacts (ckpts / sid_tensors / eval / train / inference / paper).",
        "",
        "Layout per task:",
        "```",
        "products/task<N>/",
        "├── ckpts/         # model checkpoints (.ckpt)",
        "├── sid_tensors/   # SID tensors (.pt)",
        "├── eval/          # evaluation JSON files",
        "├── train/         # symlinks to GRID/logs/train/runs/...",
        "├── inference/     # symlinks to GRID/logs/inference/runs/...",
        "└── paper/         # paper section drafts (.md / .tex)",
        "```",
        "",
        "| task | ckpts | sid_tensors | eval | train | inference | paper |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in sorted((p for p in prod.iterdir() if p.is_dir() and p.name.startswith('task')), key=lambda p: int(p.name.replace('task', ''))):
        nid = d.name.replace('task', '')
        row = [nid]
        for sub in ['ckpts', 'sid_tensors', 'eval', 'train', 'inference', 'paper']:
            sub_p = d / sub
            if sub_p.exists():
                if sub_p.is_symlink():
                    n = 'sym'
                else:
                    n = sum(1 for _ in sub_p.iterdir())
                row.append(str(n) if n != 'sym' else 'lnk')
            else:
                row.append('-')
        lines.append("| " + " | ".join(row) + " |")
    # Add legacy/orphan sections
    legacy = prod / '_legacy_result'
    if legacy.exists():
        lines.append("")
        lines.append(f"## Legacy: `_legacy_result/` ({sum(1 for _ in legacy.iterdir())} entries)")
        lines.append("")
        lines.append("Moved from `GRID/result/` (Phase 3 cleanup). Contains 102 idea/diag/task directories and 10 .md/.json files.")
    orphans = prod / '_orphans_from_root_result'
    if orphans.exists():
        lines.append("")
        lines.append(f"## Orphans: `_orphans_from_root_result/` ({sum(1 for _ in orphans.iterdir())} entries)")
        lines.append("")
        lines.append("3 unique items from root `result/` (not in GRID/result/):")
        for o in sorted(orphans.iterdir()):
            lines.append(f"- `{o.name}`")
    (PRODUCTS / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  ✓ products/README.md ({n_prod} entries)")

    # TASKS_INDEX.md (top-level)
    lines = [
        "# GRID Tasks Index",
        "",
        "## Three artifact categories",
        "",
        "- **descriptions/**: task definitions (markdown). 28 entries.",
        "- **verdicts/**: task result verdicts (markdown). 6 entries.",
        "- **products/**: execution artifacts (ckpt/pt/json/log/paper). 26 tasks.",
        "",
        "## Coverage matrix",
        "",
        "| new_id | description | verdict | product |",
        "|---:|---|:---:|:---:|",
    ]
    for k in keep:
        nid = k['new_id']
        has_desc = '✓'
        verd_prefix = f"task{nid}_"
        has_verd = '✓' if any(v.startswith(verd_prefix) for v in verd.keys()) else '✗'
        has_prod = '✓' if nid in prod_ids else '✗'
        lines.append(f"| {nid} | `{k['filename']}` | {has_verd} | {has_prod} |")
    lines += [
        "",
        "## Deleted tasks (32)",
        "",
        "Tasks with **description only, no execution**:",
        "",
        "{1,2,3,4,5,6,7,8,9,10,18,19,22,23,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55}",
        "",
        "Their `task_definitions/task<N>_*.md` were removed. See `tools/organize/delete_set.json`.",
        "",
        "## Out-of-scope (kept as-is)",
        "",
        "- `GRID/task_artifacts/results/exp388*` — exp388 series is outside the 60-task renumbering mapping.",
        "- `GRID/task_artifacts/scripts/` — script files synced (Phase 7).",
        "- `GRID/logs/{train,inference}/runs/` — original run directories, symlinked from products/.",
        "",
        "## Renumbering artifacts",
        "",
        "- `tools/renumber/mapping.json` — OLD→NEW id mapping (60 entries).",
        "- `tools/organize/keep_set.json` — 28 kept tasks (with products).",
        "- `tools/organize/delete_set.json` — 32 deleted tasks (description only).",
        "- `tools/organize/*.py` — migration scripts (audit trail).",
    ]
    (REPO / "TASKS_INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"  ✓ TASKS_INDEX.md (at root)")


if __name__ == "__main__":
    main()
