"""Task #358 — KG Phase 2A LLM-based entity+relation extraction.

R10/R11.5 自主决策:
- 抽样范围: 50 verdict (task300+ 近期活跃, schema 兼容性好, 控制成本)
- LLM 客户端: PersoanlQuery/llm_client.py MiniMaxAnthropicClient (Rule 8/9)
- 抽取 schema: kg/schema.json (11 entity types + 23 relationship types)
- Prompt 模式: zero-shot + schema 强约束 + JSON output
- 错误处理: empty response 自动 retry, JSON 解析失败落盘 _FAILED
- 增量合并: 复用 kg/build_kg.py (已支持多 extracted 文件 merge)

输出:
- kg/extracted/verdict_task<N>_llm.json  (per-verdict 抽取, LLM 产出)
- kg/extracted/verdict_task<N>_llm_FAILED.json  (LLM 失败, 标记原因)
- kg/kg_graph.json + kg/kg_statistics.json + kg/kg_visualization.png (重 build)
- logs/task358_extract.jsonl (每 verdict 调用 + cost log)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path('/home/wlia0047/ar57/wenyu/GeneRec')
KG_DIR = REPO / 'kg'
EXTRACTED_DIR = KG_DIR / 'extracted'
SCHEMA_PATH = KG_DIR / 'schema.json'
LOG_DIR = REPO / 'logs'
LOG_PATH = LOG_DIR / 'task358_extract.jsonl'

# MiniMax LLM client (Rule 8/9: must use llm_client.py MiniMaxAnthropicClient)
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
# Add .local to PYTHONPATH so anthropic SDK (installed via pip.conf target=~/.local) is importable
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/.local')
from llm_client import MiniMaxAnthropicClient


def load_schema_summary():
    """Read kg/schema.json, build compact entity/relation schema summary for prompt."""
    with open(SCHEMA_PATH) as fp:
        schema = json.load(fp)
    entity_lines = []
    for etype, edef in schema['entity_types'].items():
        required = edef.get('required_fields', [])
        entity_lines.append(f"  - {etype} (required: {', '.join(required)})")
    relation_lines = []
    for rtype, rdef in schema['relationship_types'].items():
        relation_lines.append(f"  - {rtype}")
    entity_text = '\n'.join(entity_lines)
    relation_text = '\n'.join(relation_lines)
    return entity_text, relation_text


PROMPT_TEMPLATE = """你是 GeneRec 知识图谱构建助手. 任务: 从以下 verdict 文件抽取 entities + relationships, 严格按 schema 输出 JSON.

## Entity Types (11)
{entities}

## Relationship Types (23, 选适用的)
{relations}

## Verdict 文件内容

```
{verdict_content}
```

## 输出格式 (严格 JSON, 无 markdown)

```json
{{
  "entities": {{
    "Verdict": [{{"id": "verdict_task<N>_shortslug", "task_id": <N>, "title": "...", "date": "YYYY-MM-DD", "decision": "GO|NO-GO|PASS|FAIL|NEUTRAL", "verdict_status": "...", "r10_metric": 0.0, "baseline_r10": 0.1020, "delta_pct": 0.0, "rationale_summary": "..."}}],
    "Issue": [{{"id": "issue_<NN>", "number": <NN>, "title": "...", "state": "OPEN|CLOSED", "created_at": "YYYY-MM-DD"}}],
    "Task": [{{"id": "task_<N>", "task_id": <N>, "title": "...", "gate": -1, "stage": 1, "status": "PASS|FAIL"}}],
    "Method": [{{"id": "method_<shortslug>", "name": "...", "paper_ref": "arXiv:..."}}],
    "Hyperparameter": [{{"id": "hyperparam_<shortslug>", "name": "...", "value": "...", "scope": "L0|L1|L2|global"}}],
    "Metric": [{{"id": "metric_<shortslug>", "name": "Recall@10", "k": 10, "higher_is_better": true, "anchor_value": 0.1020}}],
    "RootCause": [{{"id": "rootcause_<shortslug>", "name": "...", "description": "..."}}],
    "Decision": [{{"id": "decision_<shortslug>", "decision": "GO|NO-GO|NEUTRAL", "verdict_ref": "verdict_task<N>_shortslug"}}],
    "Stage": [{{"id": "stage_<N>", "stage_number": <N>, "name": "..."}}],
    "Gate": [{{"id": "gate_<N>", "gate_number": <N>, "name": "..."}}],
    "Hypothesis": [{{"id": "hypothesis_<shortslug>", "statement": "..."}}]
  }},
  "relationships": [
    {{"from": "verdict_xxx", "to": "issue_yyy", "type": "Verdict--evaluates-->Issue"}},
    {{"from": "task_xxx", "to": "method_yyy", "type": "Task--implements-->Method"}}
  ]
}}
```

## 重要规则

1. **Verdict 实体必须有且仅有一个**: `id="verdict_task<N>_shortslug"`, task_id 取自文件名
2. **Issue 实体**: 仅当 verdict 显式提到 GitHub issue 编号 (如 #43, #63) 时抽取
3. **Task 实体**: 当前 verdict 直接评估的 task
4. **Method 实体**: verdict 提到的算法/技术 (HypPreEncoder, κ-Stereographic, RQ-VAE, T5 等)
5. **Hyperparameter 实体**: 出现的具体数值 (c=0.74, K=128, num_emb_list=[64,128,256] 等)
6. **Metric 实体**: R@K, NDCG@K, 4-digit unique, utilization 等
7. **RootCause 实体**: NO-GO 根因 (mode collapse, Sinkhorn gap, κ→Euclidean collapse 等)
8. **Decision 实体**: GO/NO-GO 决策点
9. **Stage 实体**: Stage 1/2/3/4 (若 verdict 涉及)
10. **Gate 实体**: Gate -1/0/1/2/3/4 (若 verdict 涉及)
11. **Hypothesis 实体**: 可测试的假设
12. **Relationships**: 仅输出 schema 列出的 23 种关系类型, 严格按 `from-->to` 顺序
13. **无 markdown**: 仅输出 JSON, 不要 ```json ``` 包装
14. **空字段省略**: 没有的 entity type 可省略, 不写空数组
15. **id 命名**: `<type>_<shortslug>`, shortslug 用 lowercase + underscore

输出 JSON (不要 markdown 包装):
"""


def parse_llm_json(text):
    """Extract JSON from LLM response (may have prose around)."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block from markdown
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def extract_one_verdict(client, verdict_path, entities_text, relations_text, max_retries=2):
    """Call LLM to extract entities + relationships from one verdict file."""
    content = verdict_path.read_text(encoding='utf-8', errors='replace')
    # Truncate to ~6000 chars to stay within token budget
    if len(content) > 6000:
        content = content[:6000] + "\n\n... (truncated)"
    prompt = PROMPT_TEMPLATE.format(
        entities=entities_text,
        relations=relations_text,
        verdict_content=content,
    )
    for attempt in range(max_retries):
        try:
            thinking, text = client.call_with_thinking(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.3,
                max_retries=3,
            )
            if not text:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None, "empty_response_after_retries", thinking
            parsed = parse_llm_json(text)
            if parsed is None:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None, f"json_parse_failed: text[:300]={text[:300]!r}", thinking
            return parsed, None, thinking
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None, f"exception: {type(e).__name__}: {e}", ""
    return None, "max_retries_exceeded", ""


def collect_sample_verdicts(sample_size=50, min_task_id=300):
    """Collect verdict files to extract (task300+ recent + prior Phase 1 samples)."""
    verdicts_dir = REPO / 'verdicts'
    all_verdicts = sorted(verdicts_dir.glob('task*_*.md'))
    # Prioritize task300+ (recent, schema-compatible)
    recent = [v for v in all_verdicts
              if (m := re.match(r'task(\d+)_', v.stem))
              and int(m.group(1)) >= min_task_id]
    # Sort by task_id descending, take latest first
    recent.sort(key=lambda v: int(re.match(r'task(\d+)_', v.stem).group(1)), reverse=True)
    sample = recent[:sample_size]
    # Always include Phase 1 manually-extracted verdicts (task334 + task350) for consistency check
    return sample


def log_call(verdict_path, status, error=None, thinking_len=0, text_len=0):
    """Append one call record to logs/task358_extract.jsonl."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'verdict': verdict_path.name,
        'status': status,
        'error': error,
        'thinking_len': thinking_len,
        'text_len': text_len,
    }
    with open(LOG_PATH, 'a') as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='KG Phase 2A LLM-based extraction')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Number of verdict files to extract (default 50)')
    parser.add_argument('--min-task-id', type=int, default=300,
                        help='Minimum task_id (default 300 for recent verdicts)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without calling LLM')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of verdicts to extract (overrides sample-size)')
    args = parser.parse_args()

    print("=" * 70)
    print("Task #358 — KG Phase 2A LLM-based entity+relation extraction")
    print("=" * 70)

    print("\n[1/5] Loading schema...")
    entities_text, relations_text = load_schema_summary()
    print(f"  Entity types: {entities_text.count(chr(10)) - 1}")
    print(f"  Relation types: {relations_text.count(chr(10)) - 1}")

    print("\n[2/5] Collecting sample verdicts...")
    samples = collect_sample_verdicts(args.sample_size, args.min_task_id)
    if args.limit:
        samples = samples[:args.limit]
    print(f"  Sample size: {len(samples)}")
    for v in samples[:5]:
        print(f"    - {v.name}")
    if len(samples) > 5:
        print(f"    ... ({len(samples) - 5} more)")

    if args.dry_run:
        print("\n[DRY-RUN] Skipping LLM calls")
        return 0

    print("\n[3/5] Initializing MiniMaxAnthropicClient...")
    client = MiniMaxAnthropicClient(model='M2.5')
    print("  Client ready")

    print("\n[4/5] Extracting entities+relationships...")
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0
    skip_count = 0
    for i, verdict_path in enumerate(samples, 1):
        out_path = EXTRACTED_DIR / f"{verdict_path.stem}_llm.json"
        if out_path.exists():
            skip_count += 1
            print(f"  [{i}/{len(samples)}] SKIP (cached): {verdict_path.name}")
            continue
        print(f"  [{i}/{len(samples)}] EXTRACT: {verdict_path.name} ...", end=' ', flush=True)
        result, error, thinking = extract_one_verdict(
            client, verdict_path, entities_text, relations_text)
        if result is not None:
            # Add metadata
            result['_meta'] = {
                'source_verdict': verdict_path.name,
                'extraction_method': 'llm_minimax_m2.5',
                'extracted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            with open(out_path, 'w') as fp:
                json.dump(result, fp, indent=2, ensure_ascii=False)
            n_ent = sum(len(v) for v in result.get('entities', {}).values())
            n_rel = len(result.get('relationships', []))
            print(f"OK ({n_ent} entities, {n_rel} relations)")
            log_call(verdict_path, 'success', thinking_len=len(thinking),
                     text_len=n_ent * 100 + n_rel * 200)
            success_count += 1
        else:
            fail_path = EXTRACTED_DIR / f"{verdict_path.stem}_llm_FAILED.json"
            with open(fail_path, 'w') as fp:
                json.dump({
                    'source_verdict': verdict_path.name,
                    'error': error,
                    'thinking_preview': thinking[:500] if thinking else '',
                    'attempted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                }, fp, indent=2, ensure_ascii=False)
            print(f"FAIL ({error[:80]})")
            log_call(verdict_path, 'fail', error=error)
            fail_count += 1
        time.sleep(0.5)  # Rate-limit politeness

    print(f"\n  Summary: {success_count} success, {fail_count} fail, {skip_count} cached")

    print("\n[5/5] Done. Run kg/build_kg.py to rebuild NetworkX graph.")
    print(f"  Outputs in {EXTRACTED_DIR}/")
    print(f"  Cost log: {LOG_PATH}")
    return 0


if __name__ == '__main__':
    sys.exit(main())