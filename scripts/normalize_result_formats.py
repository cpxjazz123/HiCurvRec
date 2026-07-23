#!/usr/bin/env python3
"""规范化 result/ 目录下所有 md 和 json 文件格式。

MD: 统一 4 节结构 (设计/现象/结论/建议) + metadata 头
JSON: 统一加 task/method/date/status 顶层字段

用法:
    python3 normalize_result_formats.py --dry-run   # 预览所有改动
    python3 normalize_result_formats.py              # 实际执行
    python3 normalize_result_formats.py --md-only    # 只处理 md
    python3 normalize_result_formats.py --json-only  # 只处理 json
"""

import os
import re
import sys
import json
import argparse
import glob
from datetime import datetime

ROOT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result'

# 不处理的 md 文件（顶层 paper draft / 综合报告）
EXCLUDED_MD = {
    'idea3_paper_draft.md',
    'idea3_paper_draft_verification.md',
    'idea3_4axis_closing_report.md',
    'idea3_closing_report.md',
    'final_5algo_comparison.md',
    'task3_comparison_result.md',
    'task3_group_a_result.md',
    'task3_group_b_result.md',
    'task3_group_c_result.md',
    'task4_19_20_summary.md',
    'paper_tightening/claim_rewrites_and_limitations.md',
    'p4_3seed_v2/status.md',
    'p4_3seed/seed43_collapse_diagnosis.md',
    'task54_multi_seed/replay_plan.md',
    'ideaB_causal/STEP_VERIFICATION_REPORT.md',
    'task4_unified_rerun/RERUN_REPORT.md',
}

# 4 节标题
SECTIONS = ['设计', '现象', '结论', '建议']

# 章节映射规则
SECTION_PATTERNS = {
    '设计': ['设计', '背景', '实验设置', '方法', '目的', '动机', '测什么', '核心目标', '定义'],
    '现象': ['结果', '数据', '现象', '实验现象', 'per-layer', 'per-algorithm', 'per-config',
             'per-seed', 'per-c', '表现', '实测', 'r@', 'mse', 'statistics',
             'regression', 'r10', 'recall', 'ndcg', 'mrr'],
    '结论': ['结论', '判读', '关键发现', '核心结论', '关键结论', '判定', '解读',
             'consolidated findings', '主发现', '总结', 'causal'],
    '建议': ['后续', '建议', '局限', '展望', 'follow-up', '产物', 'discussion'],
}


# ========== JSON 规范化 ==========

def infer_task_name(file_path):
    """从文件所在目录名推断 task 名。

    例: ./taskA6_hrq_topology_vs_perf/xxx.json -> taskA6
        ./idea2_dnc/xxx.json -> idea2_dnc
        ./taskA11/xxx.json -> taskA11
    """
    rel = os.path.relpath(file_path, ROOT)
    dir_name = os.path.dirname(rel).split(os.sep)[0]

    # 优先匹配 taskAxx 或 taskxxx
    m = re.match(r'^(taskA?\d+)', dir_name)
    if m:
        return m.group(1)
    # idea{N}
    m = re.match(r'^(idea[ABCD]?\d*_?[a-z_]*)', dir_name)
    if m:
        return m.group(1)
    # diag{N}
    m = re.match(r'^(diag\d+_?[a-z_]*)', dir_name)
    if m:
        return m.group(1)
    # 顶层文件
    if not os.path.dirname(rel):
        return 'top_level'
    return dir_name


def infer_method_name(file_path):
    """从文件名推断 method 名。

    例: topology_vs_perf.json -> topology_vs_perf
        abc.json -> abc
    """
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def normalize_json(file_path, dry_run=False):
    """规范化 json 文件：加 task/method/date/status 顶层字段。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return 'skipped_json_error'

    # 已是 list 或 str 等非 dict 类型 → 包裹
    if not isinstance(data, dict):
        if dry_run:
            return f'wrapped:{type(data).__name__}'
        wrapped = {
            'task': infer_task_name(file_path),
            'method': infer_method_name(file_path),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'status': 'completed',
            'data': data,
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(wrapped, f, indent=2, ensure_ascii=False)
        return 'wrapped'

    # 已有 task + method 字段 → 跳过
    if 'task' in data and 'method' in data:
        # 补 date/status（缺失时）
        if 'date' not in data or 'status' not in data:
            data['date'] = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            data['status'] = data.get('status', 'completed')
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return 'partial_updated'
        return 'skipped'

    # 已有完整 schema 但缺 data → 跳过
    if {'task', 'method', 'date', 'status'}.issubset(set(data.keys())):
        return 'skipped'

    # 包裹到 data 字段
    if dry_run:
        return 'wrapped'

    wrapped = {
        'task': infer_task_name(file_path),
        'method': infer_method_name(file_path),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'status': 'completed',
        'data': data,
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(wrapped, f, indent=2, ensure_ascii=False)
    return 'normalized'


# ========== MD 规范化 ==========

def extract_metadata(content):
    """提取已有 metadata blockquote。"""
    m = re.search(r'(> \*\*[^*]+\*\*[^\n]*\n(?:> [^\n]*\n?)+)', content)
    if m:
        return m.group(1).rstrip() + '\n\n'
    return ''


def extract_h1(content):
    """提取 H1 标题。"""
    m = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_date(content):
    """从 metadata 或内容里提取日期。"""
    # 已有元数据
    m = re.search(r'\*\*日期\*\*[：:]\s*(\d{4}-\d{2}-\d{2})', content)
    if m:
        return m.group(1)
    # 文件 mtime
    return datetime.now().strftime('%Y-%m-%d')


def extract_status(content):
    """从 metadata 或内容里提取状态。"""
    m = re.search(r'\*\*状态\*\*[：:]\s*([^\n]+)', content)
    if m:
        return m.group(1).strip()
    return '✅ 完成'


def extract_executor(content):
    """从 metadata 或内容里提取执行人。"""
    m = re.search(r'\*\*执行人\*\*[：:]\s*([^\n]+)', content)
    if m:
        return m.group(1).strip()
    return 'Claude'


def infer_md_task_name(file_path):
    """从 md 文件路径推断任务名（同 json 逻辑）。"""
    return infer_task_name(file_path)


def split_sections(content):
    """按 H2/H3 切分现有 md 内容到 4 节。

    智能识别"顶层节"：取出现次数最多的 # 层级作为顶层节边界。
    例：原文全是 ### (无 ##)，则所有 ### 都是顶层节，按关键词映射到 4 大节。
        原文有 ##, ### 混合，则 ## 是顶层节，### 是嵌套节。
    """
    # 移除 metadata blockquote
    content_no_meta = re.sub(r'(> \*\*[^*]+\*\*[^\n]*\n(?:> [^\n]*\n?)+)', '', content)

    # 移除 H1 标题（重新生成）
    h1_match = re.search(r'^#\s+.+$', content_no_meta, re.MULTILINE)
    if h1_match:
        content_no_meta = re.sub(r'^#\s+.+\n', '', content_no_meta, count=1)

    # 找所有 H2/H3 标题
    section_pattern = re.compile(r'^(#{2,3})\s+(.+?)$', re.MULTILINE)
    matches = list(section_pattern.finditer(content_no_meta))

    if not matches:
        # 没有 H2/H3 → 全部内容归入"现象"
        return {
            '设计': '',
            '现象': content_no_meta.strip(),
            '结论': '',
            '建议': '',
        }

    # 优先使用 ## 作为顶层节（如果存在），否则用 ###
    h2_matches = [m for m in matches if len(m.group(1)) == 2]
    h3_matches = [m for m in matches if len(m.group(1)) == 3]
    top_matches = h2_matches if h2_matches else h3_matches

    # 提取顶层节及其内容（包含可能嵌套的更深层级）
    top_sections = []
    for i, m in enumerate(top_matches):
        title = m.group(2).strip()
        start = m.end()
        end = top_matches[i + 1].start() if i + 1 < len(top_matches) else len(content_no_meta)
        body = content_no_meta[start:end].strip()

        # 清理 body：移除与顶层标题重复的嵌套 ### 标题（如 ### 设计 在 ## 设计 下）
        if title.strip() in SECTIONS:
            sub_pattern = re.compile(r'^###\s+' + re.escape(title.strip()) + r'\s*$\n?', re.MULTILINE)
            body = sub_pattern.sub('', body).strip()

        top_sections.append((title, body))

    # 映射到 4 节
    mapped = {'设计': '', '现象': '', '结论': '', '建议': ''}
    unmatched = []

    for orig_title, body in top_sections:
        # 检查哪个 section 匹配
        target = None
        for target_section, keywords in SECTION_PATTERNS.items():
            if any(kw.lower() in orig_title.lower() for kw in keywords):
                target = target_section
                break
        if target:
            if mapped[target]:
                mapped[target] += '\n\n'
            # 如果标题就是 4 节关键词本身（如 "### 设计"），就不重复加 ### 标题
            if orig_title.strip() in SECTIONS or orig_title.strip() in ['设计', '现象', '结论', '建议']:
                mapped[target] += body
            else:
                mapped[target] += f'### {orig_title}\n\n{body}'
        else:
            unmatched.append((orig_title, body))

    # 未匹配的章节 → 归入"现象"
    if unmatched:
        if mapped['现象']:
            mapped['现象'] += '\n\n'
        for orig_title, body in unmatched:
            if orig_title.strip() in SECTIONS or orig_title.strip() in ['设计', '现象', '结论', '建议']:
                mapped['现象'] += body
            else:
                mapped['现象'] += f'### {orig_title}\n\n{body}'

    return mapped


def build_md(file_path, content):
    """组装规范化后的 md 内容。"""
    title = extract_h1(content)
    if not title:
        # 从文件名推断
        base = os.path.splitext(os.path.basename(file_path))[0]
        title = base.replace('_', ' ').title()

    date = extract_date(content)
    status = extract_status(content)
    executor = extract_executor(content)
    task = infer_md_task_name(file_path)

    sections = split_sections(content)

    parts = [
        '---',
        '',
        f'> **任务**：{task}',
        f'> **日期**：{date}',
        f'> **状态**：{status}',
        f'> **执行人**：{executor}',
        '',
        '---',
        '',
        f'# {title}',
        '',
    ]

    for section in SECTIONS:
        parts.append(f'## {section}')
        parts.append('')
        if sections[section]:
            parts.append(sections[section])
            parts.append('')

    return '\n'.join(parts)


def normalize_md(file_path, dry_run=False):
    """规范化 md 文件。"""
    rel = os.path.relpath(file_path, ROOT)
    if rel in EXCLUDED_MD:
        return 'excluded'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return 'skipped_encoding'

    new_content = build_md(file_path, content)

    if dry_run:
        return 'would_update'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return 'normalized'


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只打印将要改动的文件')
    parser.add_argument('--md-only', action='store_true', help='只处理 md')
    parser.add_argument('--json-only', action='store_true', help='只处理 json')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    args = parser.parse_args()

    md_files = []
    json_files = []

    if not args.json_only:
        md_files = glob.glob(f'{ROOT}/**/*.md', recursive=True)
        # 排除顶层 paper draft
        md_files = [f for f in md_files if os.path.relpath(f, ROOT) not in EXCLUDED_MD]

    if not args.md_only:
        json_files = glob.glob(f'{ROOT}/**/*.json', recursive=True)
        # 排除 Lightning 自动生成的 metadata 文件（不规范化）
        json_files = [
            f for f in json_files
            if 'metadata/restart_metadata.json' not in f
            and 'multirun_metadata' not in f
        ]

    print(f'==== Normalize result/ formats ====')
    print(f'MD files: {len(md_files)}')
    print(f'JSON files: {len(json_files)}')
    print(f'Dry-run: {args.dry_run}')
    print()

    stats = {}

    # Process MD
    if md_files:
        print(f'--- Processing {len(md_files)} md files ---')
        for f in md_files:
            result = normalize_md(f, dry_run=args.dry_run)
            stats[result] = stats.get(result, 0) + 1
            if args.verbose or args.dry_run:
                rel = os.path.relpath(f, ROOT)
                print(f'  [{result:20s}] {rel}')
        print()

    # Process JSON
    if json_files:
        print(f'--- Processing {len(json_files)} json files ---')
        for f in json_files:
            result = normalize_json(f, dry_run=args.dry_run)
            stats[result] = stats.get(result, 0) + 1
            if args.verbose or args.dry_run:
                rel = os.path.relpath(f, ROOT)
                print(f'  [{result:20s}] {rel}')
        print()

    print('==== Stats ====')
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()