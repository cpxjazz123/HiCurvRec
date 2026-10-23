# The #109 postmortem (issue #110, section 0)

What the previous campaign actually did, measured rather than remembered, and what is now
encoded as a check. The checks live in `tools/harness2_genrec/forensics_109.py` and are run
against the archives on demand.

## The eight lessons

1. **False progress.** Rounds 48–119 carried 72 portfolio files with no implementation file,
   and 73 rounds went to adopting arms that had already been measured. A re-adoption is not a
   new candidate, a novel proposal, a training or a successful round, and the number of
   rounds is never a measure of research progress.
2. **Repeated candidate identities.** 117 proposals over 93 distinct names, with 17 names
   recurring. The registry now compares the normalised equation, the operator classes and the
   implementation's abstract syntax, is keyed by the proposal hash and the run id, and an
   absent or edited registry **blocks** rather than skipping the comparison.
3. **A lost CGV contract.** **None** of the 117 proposals carried a caller-supplied
   seven-group contract, and the launch substituted a generic template. A template is not a
   contract: a verifier that falls back to one verifies the template instead of the
   mechanism. The contract is now required, a missing group is blocking, and the launch raises
   rather than substituting.
4. **Repeated work.** The screen and the cold run each recreated the tree, asked for the
   implementation again and ran another probe. The implementation is implemented and probed
   once, and the cold run launches from the sealed tree.
5. **Repeated Stage-3.** Runs wrote their rank shards and then died before merging them,
   and were relaunched. Complete shards are a recoverable final artifact: the metrics are
   recovered from them, with a convention derived from the archived finals and asserted to
   reproduce them to twelve decimals.
6. **Wrong scientific surrogates.** A high ER did not transfer to Recall or NDCG. Both ER
   shares are reported, the D2 carries its support flag — on the frozen behaviour pairs only
   about one per cent of the pairs share their level-1 code, so the D2's positive conditional
   set is a few hundred pairs and every D2 so far is flagged low-support — and the history gain
   is exploratory: its retrospective rank correlation with the Recall is about +0.46 over
   thirteen arms against the ER's +0.17, which is a hypothesis and not a certified oracle.
7. **An invalid scientific end.** The success decision was made in three places, and one of
   them read the Stage-3 metrics without the frozen evaluation. There is now one predicate,
   `is_verified_triple_win`, and every phase and every report asks it.
8. **Euclidean framing.** **None** of the 117 proposals used a hyperbolic operator, and 88
   were Euclidean under a geometry title. Running inside a paper about hyperbolic geometry does
   not make a mechanism hyperbolic, which is why the H2 gate traces the operator in the
   actual loss.

## The account that was corrected

The campaign's account was that the frozen ER is bought with code concentration, and that
the concentration is what the recommender loses. The first half holds. The second half does
not: the arm with the **best** balance of the cohort — ER 16.6019, a level-0 Gini of 0.193
against the control's 0.216, a largest cell of 253 against 237, 66.5 prefix branches against
65.9, better on all three — lost the **most** downstream quality of any arm, −0.0108 of
Recall@10, three times the next worst. Over the eleven arms with both measurements, the
Gini's rank correlation with the downstream loss is −0.07, which is no correlation, while the
tail's is −0.52 and the prefix tree's +0.44.

So the requirement is not "do not concentrate the codes" in general. It is: do not let the
largest level-0 cell grow and do not let the prefix tree narrow — and the loss is probably
not a property of the export's shape at all, but of the mechanism's effect on the training,
which the shape cannot show. That is why the campaign now carries the conditional L2
discrimination and the history-conditioned gain, and why the next portfolios are aimed at the
gain rather than at the Gini.
