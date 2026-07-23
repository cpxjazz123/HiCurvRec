#!/usr/bin/env python3
"""
Task #87 — paper Table 2 baseline 复现最终汇总 (vs paper ETEGRec Table 2)

输入: 5 baseline 各自最新 R@10/NDCG@10 (从 task78/#80/#81/#82/#83 推理输出抽取)
输出: paper_table2_baseline_comparison.csv + .md 表格 (R@10 paper vs 复现, Δ)

任务来源: paper "A Systematic Reproducibility Study of ETEGRec" arxiv 2024.
Table 2 baseline R@10 (Amazon Musical Instruments 5-core):
  #1 SAsRec:        0.0596
  #2 S³Rec:         0.0538
  #3 FMLP-Rec:      0.0657
  #4 DuoRec:        0.0588
  #5 LightGCN:      0.0454
  #6 LETTER:        0.0581
  #7 FDSA:          0.0557
  #8 S³Rec (sep):   same as #2
  #9 TIGER (T5):    0.0574
  #10 CoST:         0.0570
  #11 P5-CID:       0.0507
  #12 P5-SID:       0.0438

复现规划:
- task78 TIGER (T5): paper #9
- task80 FDSA: paper #7
- task81 S3Rec: paper #2 (RecBole 现成)
- task82 P5-CID: paper #11
- task83 P5-SID: paper #12
"""
import os, sys, re, json, glob
from pathlib import Path
from datetime import datetime

GENE_REC = Path("/home/wlia0047/ar57/wenyu/GeneRec")
LOGS = GENE_REC / "logs"
RESULTS = GENE_REC / "results"
RESULTS.mkdir(exist_ok=True)


def parse_letter_test_output(log_path: Path):
    """LETTER/test.py 输出格式: recall@1=, recall@5=, recall@10=, ndcg@5=, ndcg@10="""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    metrics = {}
    for m in ['recall@1', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
        match = re.search(rf'{m}=([\d.]+)', text, re.IGNORECASE)
        if match:
            metrics[m] = float(match.group(1))
    return metrics if metrics else None


def parse_recbole_test_output(log_path: Path):
    """RecBole test 输出: recall@10 = X, ndcg@10 = Y"""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    metrics = {}
    for m in ['recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
        match = re.search(rf"{m}\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
        if match:
            metrics[m] = float(match.group(1))
    return metrics if metrics else None


def parse_p5_test_output(log_path: Path):
    """LLM-RecSys-ID main.py 输出格式待定: 通常是 NDCG@10 = X"""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    metrics = {}
    # P5 LLM-RecSys-ID 输出可能是 test_metric lines
    for m in ['recall@1', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
        for line in text.split('\n'):
            match = re.search(rf'(?:{m}|hit).*?([\d.]{{3,5}})', line, re.IGNORECASE)
            if match:
                # 取最高的, 因为 P5 输出全部 metrics
                val = float(match.group(1))
                if val < 1.0:  # R@10 应 < 1.0
                    metrics[m] = max(metrics.get(m, 0), val)
    return metrics if metrics else None


PAPER_TABLE = {
    "TIGER (T5)":     0.0574,
    "FDSA":           0.0557,
    "S3Rec":          0.0538,
    "P5-CID":         0.0507,
    "P5-SID":         0.0438,
}

TASK_CONFIG = {
    "TIGER (T5)": {
        "log": LOGS / "task84_tiger_t5_inference_*.log",
        "parser": parse_letter_test_output,
        "task_id": 84,
    },
    "FDSA": {
        "log": LOGS / "task85_fdsa_inference_*.log",
        "parser": parse_recbole_test_output,
        "task_id": 85,
    },
    "S3Rec": {
        "log": LOGS / "task85_s3rec_inference_*.log",
        "parser": parse_recbole_test_output,
        "task_id": 85,
    },
    "P5-CID": {
        "log": LOGS / "task86_p5_cid_inference_*.log",
        "parser": parse_p5_test_output,
        "task_id": 86,
    },
    "P5-SID": {
        "log": LOGS / "task86_p5_sid_inference_*.log",
        "parser": parse_p5_test_output,
        "task_id": 86,
    },
}


def main():
    """主函数: 收集所有 baseline 复现结果, 写 csv + md."""
    rows = []
    for baseline, config in TASK_CONFIG.items():
        # 找最新 log
        log_paths = sorted(glob.glob(str(config["log"])))
        if not log_paths:
            metrics = None
            log_used = "(无)"
        else:
            log_path = Path(log_paths[-1])
            metrics = config["parser"](log_path)
            log_used = str(log_path)
        
        paper_r10 = PAPER_TABLE[baseline]
        our_r10 = metrics.get('recall@10') if metrics else None
        delta = (our_r10 - paper_r10) if our_r10 is not None else None
        delta_pct = (delta / paper_r10 * 100) if (delta is not None and paper_r10) else None
        
        rows.append({
            "baseline": baseline,
            "task_id": config["task_id"],
            "paper_R@10": paper_r10,
            "our_R@10": our_r10,
            "delta": delta,
            "delta_pct": delta_pct,
            "our_NDCG@10": metrics.get('ndcg@10') if metrics else None,
            "log": log_used,
        })
    
    # 写 csv
    csv_path = RESULTS / "paper_table2_baseline_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write("baseline,task_id,paper_R@10,our_R@10,delta,delta_pct,our_NDCG@10,log\n")
        for r in rows:
            f.write(
                f"{r['baseline']},{r['task_id']},{r['paper_R@10']},"
                f"{r['our_R@10'] if r['our_R@10'] is not None else ''},"
                f"{r['delta'] if r['delta'] is not None else ''},"
                f"{r['delta_pct']:.1f}%" if r['delta_pct'] is not None else f",,"
                f",{r['our_NDCG@10'] if r['our_NDCG@10'] is not None else ''},"
                f"{r['log']}\n"
            )
    
    # 写 md
    md_path = RESULTS / "paper_table2_baseline_comparison.md"
    with open(md_path, 'w') as f:
        f.write("# Paper Table 2 Baseline 复现汇总 (Musical Instruments)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Baseline | Paper R@10 | 复现 R@10 | Δ | Δ% | NDCG@10 | Task |\n")
        f.write("|----------|-----------|---------|---|---|---------|------|\n")
        for r in rows:
            r10_str = f"{r['our_R@10']:.4f}" if r['our_R@10'] is not None else "⏳ 训练中"
            d_str = f"{r['delta']:+.4f}" if r['delta'] is not None else "—"
            dp_str = f"{r['delta_pct']:+.1f}%" if r['delta_pct'] is not None else "—"
            n10_str = f"{r['our_NDCG@10']:.4f}" if r['our_NDCG@10'] is not None else "—"
            f.write(f"| {r['baseline']} | {r['paper_R@10']:.4f} | {r10_str} | {d_str} | {dp_str} | {n10_str} | #{r['task_id']} |\n")
        
        # 总结
        completed = [r for r in rows if r['our_R@10'] is not None]
        if len(completed) > 0:
            f.write(f"\n## Summary\n\n")
            f.write(f"- 复现完成: {len(completed)} / {len(rows)}\n")
            won = [r for r in completed if r['delta'] and r['delta'] > 0]
            lost = [r for r in completed if r['delta'] and r['delta'] < 0]
            f.write(f"- 超越 paper: {len(won)} ({', '.join(r['baseline'] for r in won) if won else 'None'})\n")
            f.write(f"- 低于 paper: {len(lost)} ({', '.join(r['baseline'] for r in lost) if lost else 'None'})\n")
    
    # stdout 报告
    print(f"[OK] 写入: {csv_path}")
    print(f"[OK] 写入: {md_path}")
    print()
    print("===== Paper Table 2 Baseline 复现 vs paper =====")
    print(f"{'Baseline':<12} {'Paper R@10':>10} {'Our R@10':>10} {'Δ':>8} {'Δ%':>8} {'Task':>5}")
    for r in rows:
        r10_str = f"{r['our_R@10']:.4f}" if r['our_R@10'] is not None else "⏳ 训练中"
        d_str = f"{r['delta']:+.4f}" if r['delta'] is not None else "—"
        dp_str = f"{r['delta_pct']:+.1f}%" if r['delta_pct'] is not None else "—"
        print(f"{r['baseline']:<12} {r['paper_R@10']:>10.4f} {r10_str:>10} {d_str:>8} {dp_str:>8} #{r['task_id']:>4}")


if __name__ == "__main__":
    main()
