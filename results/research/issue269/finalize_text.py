"""Fill the verdict/comment with the measured Stage3 numbers (no new computation)."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
rec=json.loads((ROOT/'results/research/issue269/analysis/STAGE3_RECALL.json').read_text())
cur=rec['current']; base=rec['baselines']
d10=rec['delta_vs_parent']; d10t=rec['delta_vs_tiger']
replay=json.loads((ROOT/'results/research/issue269/analysis/REPLAY.json').read_text())
status=replay.get('status')
failed=replay.get('failed') or []
verdict_passed = bool(cur['test_recall@10']>0.065)
q4=(f"`test_R@10` = **{cur['test_recall@10']:.12f}**, `R@5` = {cur['test_recall@5']:.12f}, "
    f"`NDCG@10` = {cur['test_ndcg@10']:.12f}, n_eval = {cur['n_eval']}. "
    f"Frozen TIGER `0.05520639286895663` (delta {d10t:+.6f}); #265 parent "
    f"`0.052594926791900976` (delta {d10:+.6f}); target `>0.065` "
    f"{'MET' if verdict_passed else 'NOT MET'}. Independent replay status `{status}`"
    f"{', failed checks: '+str(failed) if failed else ''}.")
v=ROOT/'results/research/issue269/VERDICT.md'; t=v.read_text()
t=t.replace("Stage3 result: **PENDING**.", "Stage3 result: `test_R@10` **%.12f** (see §4)." % cur['test_recall@10'])
t=t.replace("`test_R@10` = **PENDING** (Stage3 single-card run `TIGER_calibrated_kappa_rq`, 150 epochs,\nseed 42, beam 20). Comparison points: frozen TIGER `0.05520639286895663`, #265 parent\n`0.052594926791900976`, target `> 0.065`.",
            q4+"\n\nSingle-card comparison caveat: the run used the single-card branch (global train batch\n4096 against the four-card 16384), so it is not numerically comparable to four-card arms;\nit is the only layout this host exposes. Existing test access risk: the frozen Stage3 test was\nalready read by earlier arms; this arm is one more read and was not tuned on it.")
t=t.replace("| C12 | One Stage3 150 epoch; freeze candidate, final test once; independent Recall/NDCG; compare TIGER/parent/0.065; disclose single-card and test-reuse risk | Stage3 run + `replay.py` | PENDING |",
            "| C12 | One Stage3 150 epoch; freeze candidate, final test once; independent Recall/NDCG; compare TIGER/parent/0.065; disclose single-card and test-reuse risk | `analysis/STAGE3_RECALL.json`, `analysis/REPLAY.json` | PASS |")
t=t.replace("| C13 | Answer the four questions | this file | PENDING (Q4) |","| C13 | Answer the four questions | this file | PASS |")
p=ROOT/'results/research/issue269/COMMENT_003.md'; c=p.read_text()
c=c.replace("| test R@10 | PENDING |", "| test R@10 | `%.12f` |" % cur['test_recall@10'])
c=c.replace("| test R@5 | PENDING |", "| test R@5 | `%.12f` |" % cur['test_recall@5'])
c=c.replace("| test NDCG@10 | PENDING |", "| test NDCG@10 | `%.12f` |" % cur['test_ndcg@10'])
c=c.replace("Independent replay (`tools/iteration/replay.py`): PENDING.",
            "Independent replay (`tools/iteration/replay.py`): `%s`%s." % (status, "" if not failed else ", failed: "+str(failed)))
c=c.replace("**Verdict:** PENDING.", "**Verdict:** %s" % ("target MET" if verdict_passed else "target NOT MET; the result is recorded as a negative recommendation outcome, not a tuning signal"))
v.write_text(t); p.write_text(c)
print(json.dumps({'r10':cur['test_recall@10'],'delta_parent':d10,'delta_tiger':d10t,'target':verdict_passed,'replay':status,'failed':failed},indent=1))
