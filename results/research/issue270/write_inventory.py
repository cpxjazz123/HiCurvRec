"""Render INVENTORY.md from the frozen inventory/cohort files."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"


def main() -> None:
    inv = json.loads((OUT / "RUN_INVENTORY.json").read_text())
    coh = json.loads((OUT / "COHORT.json").read_text())
    rows = coh["rows"]
    usable = [r for r in rows if r["usable_geometry"]]
    fam = Counter(r["family"] for r in usable)
    proto = Counter()
    for r in usable:
        p = Path(r["stage3_final"]).parent / "predictions"
        proto["single" if len(list(p.glob("rank*.jsonl"))) == 1 else "multi"] += 1
    lines = ["# Issue #270 run inventory", "",
             "Generated from `RUN_INVENTORY.json` and `COHORT.json`; every number below is a",
             "re-read of an archived artefact, no imputation.", "",
             "## Coverage", "",
             f"- Stage3 runs with `test_final.json`: **{len(rows)}**.",
             f"- R2 ledger arms with a four-piece entry: **{inv['ledger_arms']}**.",
             f"- Usable geometry cohort (exact SID + checkpoint + 57,439 predictions): **{len(usable)}**.",
             f"  - record-verified immutable-run bindings: **{sum(r['record_verified'] for r in usable)}**;",
             f"  - legacy arm-directory bindings (no SID/checkpoint hash in the Stage3 record): **{sum(not r['record_verified'] for r in usable)}**.",
             f"- Duplicate SID groups: **{len(coh['duplicate_sids'])}**.",
             f"- Protocol proxy (rank-file count): single-card **{proto['single']}**, four-card **{proto['multi']}**.",
             "",
             "Families:"]
    for k in sorted(fam):
        lines.append(f"- `{k}`: {fam[k]}")
    lines += ["", "## Frozen TIGER baseline and reruns", "",
              "| run | R@10 |", "|---|---|"]
    tigers = sorted(Path(REPO / "results/stage3_T5Train/TIGER_RQ-VAE").glob("**/test_final.json"),
                    key=lambda p: p.parent.name)
    for p in tigers:
        d = json.loads(p.read_text())
        lines.append(f"| `{p.parent.name}` | {d.get('test_recall@10'):.12f} |")
    vals = [json.loads(p.read_text()).get("test_recall@10") for p in tigers]
    lines += ["", f"Range: **{min(vals):.12f} – {max(vals):.12f}**. The canonical latest run is",
              "`Nov-23-2026_10-57-48` and is used as the pairing baseline; the spread is reported",
              "so no single-run difference is interpreted as significant on its own.", ""]
    lines += ["## Usable cohort", "",
              "| arm | family | records | binding | SID sha256 | checkpoint sha256 | R@10 | predictions |",
              "|---|---|---|---|---|---|---|---|"]
    for r in usable:
        lines.append("| `%s` | %s | %s | %s | `%s` | `%s` | %.12f | %d |" % (
            r["arm"], r["family"], "verified" if r["record_verified"] else "legacy",
            r["binding"], (r["sid_sha256"] or "-")[:16], (r["checkpoint_sha256"] or "-")[:16],
            r["recall@10"], r["prediction_rows"]))
    excluded = [r for r in rows if not r["usable_geometry"] and r["arm"] in {
        "TIGER_role_geom_rq", "TIGER_role_geom_global_rq", "TIGER_role_geom_shuffled_rq",
        "TIGER_mixed_rq_F0"}]
    lines += ["", "## Known gaps", "",
              "- #259–#264 geometry arms (`role_geom*`, `mixed_rq_F0`) exist with SIDs and checkpoints",
              "  but their Stage3 prediction shards are absent locally, so they cannot enter the",
              "  recommendation linkage; they are listed in `RUN_INVENTORY.json` and excluded here.",
              "- Legacy bindings lack a recorded SID/checkpoint hash in the Stage3 record; the",
              "  files are still read exactly and their hashes are recorded in `COHORT.json`.",
              "- No validation predictions exist (Stage3 `NO_EVAL=True`); all recommendation",
              "  linkage is explanatory error analysis on the frozen test set."]
    (OUT / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    print("INVENTORY.md written", len(lines), "lines")


if __name__ == "__main__":
    main()
