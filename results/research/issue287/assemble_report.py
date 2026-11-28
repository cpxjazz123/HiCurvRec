"""Issue #287: fill the round placeholders in `REPORT.md` from the evaluation records.

The head and the tail were merged once; the placeholders (`{{R4P}}`, `{{R4}}`, ...) are filled in
place, so the script is idempotent: re-running it only replaces what is still unfilled.
"""
from __future__ import annotations

import json
from pathlib import Path

P = Path('results/research/issue287')
ROUNDS = {'round1': 'R1', 'round2': 'R2', 'round3': 'R3',
          'round4': 'R4', 'round5': 'R5', 'round6': 'R6', 'round7': 'R7'}
THRESHOLD = (0.08097285816257246, 0.008752316503966327, 0.1511367795989751)
REF = {'R2': (0.0783613920855168, 0.008752316503966327, 0.1511367795989751),
       'R3': (0.08097285816257246, 0.00901661164455637, 0.1543498596819457)}


def fmt(value: float) -> str:
    return repr(float(value))


def run_block(tag: str, rec: dict) -> str:
    train = rec.get('training', {})
    passes = [rec.get('passes_P'), rec.get('passes_N'), rec.get('passes_C')]
    verdict = ('**all three thresholds pass.**' if all(passes) else
               f"**{sum(1 for p in passes if p)} of 3 thresholds pass.**")
    lines = [f"3000 epochs, {train.get('total_wall_time_s')} s wall time, "
             f"{train.get('epoch_time_s_mean')} s/epoch mean (single card), final training collision "
             f"{train.get('final_collision')}. The checkpoint-bound export "
             f"(`item_sids.json`, sha `{rec.get('sid_sha256', '')[:16] if rec.get('sid_sha256') else ''}…`) "
             f"evaluated with the frozen evaluator:", '',
             '| metric | ' + tag.replace('round', 'round ') + ' | round 1 | R2 | R3 | threshold | verdict |',
             '|---|---|---|---|---|---|---|']
    round1 = json.loads((P / 'results' / 'round1.json').read_text())
    metrics = (('P', 'passes_P', '> 0.08097285816257246'),
               ('N', 'passes_N', '< 0.008752316503966327'),
               ('C', 'passes_C', '< 0.1511367795989751'))
    for name, flag, threshold in metrics:
        index = 'PNC'.index(name)
        lines.append(f"| {name} | **{fmt(rec[name])}** | {fmt(round1[name])} | "
                     f"{fmt(REF['R2'][index])} | {fmt(REF['R3'][index])} | {threshold} | "
                     f"**{'PASS' if rec.get(flag) else 'FAIL'}** |")
    diagnostics = rec.get('diagnostics', {})
    lines += ['', f"Diagnostics: Hard ER {rec.get('hard_er', 0.0):.4f}, L2 prefixes "
                  f"{diagnostics.get('distinct_l2_prefixes')}, L3 refinement "
                  f"{diagnostics.get('l3_refinement_gain_bits', 0.0):.4f} bits.", '',
              f"**Round verdict: {verdict}**", '']
    return '\n'.join(lines)


def main() -> None:
    report = P / 'REPORT.md'
    text = report.read_text()
    rounds = {}
    for tag in ROUNDS:
        path = P / 'results' / f'{tag}.json'
        if path.is_file():
            rounds[tag] = json.loads(path.read_text())
    for tag, short in ROUNDS.items():
        rec = rounds.get(tag)
        if rec is None:
            continue
        text = text.replace('{{' + short + 'P}}', fmt(rec['P']))
        text = text.replace('{{' + short + 'N}}', fmt(rec['N']))
        text = text.replace('{{' + short + 'C}}', fmt(rec['C']))
        text = text.replace('{{' + short + 'P_PASS}}', 'PASS' if rec.get('passes_P') else 'FAIL')
        text = text.replace('{{' + short + 'N_PASS}}', 'PASS' if rec.get('passes_N') else 'FAIL')
        text = text.replace('{{' + short + 'C_PASS}}', 'PASS' if rec.get('passes_C') else 'FAIL')
        text = text.replace('{{' + short + 'ALL}}', 'yes' if rec.get('passes_all') else 'no')
        if tag in ('round4', 'round5', 'round6', 'round7'):
            text = text.replace('{{' + short + '}}', run_block(tag, rec))
    report.write_text(text)
    print('report filled', {k: (v['P'], v['N'], v['C'], v.get('passes_all')) for k, v in rounds.items()})
    print('remaining placeholders:', text.count('{{'))


if __name__ == '__main__':
    main()
