## Round 3 start: codebook-side gradient budget

Issue body sha256 `8814eae7356ac36ba26c0812b51a1d4327b499171327eaa31be3b70bbeca84f7`. The
acceptance list is unchanged: P1 full common-ruler J <= 0.338336 with the 1-hop/2-hop tolerance;
P2 local A.1 MAE <= 1.893506, Spearman > 0.015851, sign > 0.432453, non-degenerate
(var_ratio >= 0.5, slope >= 0.1); P3 L2 >= 10,918, rank >= 8, prefix/behaviour >= TIGER; P4
recon <= 8.589e-5; P5 mu active. Stage3 remains forbidden by the issue.

**Root cause under test.** The local curvature term and the global J term both update the codebook
(the assignment). All tested arms applied them fully to the codebook; round 2's 10% local codebook
share collapsed the assignment (collision 0.983), while round 1's ~0.05% did not; the 0.5–2% share
was never tested. Competing explanations: the encoder-side fit (null), the scale shrink (P2
non-degeneracy), and the kappa_t base itself (the #265 Euclidean L2).

**Change (single primary).** Codebook-side gradient budget `beta = 0.01` on the local+constraint
terms; the encoder, decoder and reconstruction keep full gradients; the local+global terms still
reach the codebook at beta > 0. No objective, architecture, selection, Sinkhorn, floor or
reconstruction change (the issue's "梯度预算及冲突处理" allowance). The round-1 immutable source
(`stage2_4dffd8b4.../src`) is the base; the round-2 tree is archived per §1.0a.

**Tests.** `tests/test_codebook_budget.py` (known answers for beta=0/0.01/1, the inactive
constraint, the missing gradient, the independent composition) and
`tests/test_codebook_budget_mutations.py` (M1–M6, including the no-budget red baseline). 12 passed;
one reference test failed on a test-side graph-retention bug, was fixed and re-run (1 passed). Logs:
`results/research/issue272/checks/round3_budget_tests*.log`; `MUTATIONS.md` in the tree.

**Plan.** Smoke -> 3000 epochs -> independent local evaluator + common-ruler J + exported SID +
reconstruction + mu -> the P1–P5 PASS table. If the P3 rank/L2 improve but the P2 ordering stays
below the TIGER, the next round targets the encoder-side local loss (R3) or the base (R4) with the
measured direction; if the P3 stays at the round-1 level, R1 is excluded. The iteration continues
under the owner 2026-11-27 contract; `X_tested` is not an `X_allowed` proof.
