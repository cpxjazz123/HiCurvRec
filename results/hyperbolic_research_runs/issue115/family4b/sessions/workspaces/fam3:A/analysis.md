# Frozen evidence: what a new family must move

## 1. Joint ER and Stage3 movement

The measured obstruction identifies the relevant structural quantity: **the information carried by the three-token SID, especially whether the later token(s) distinguish items within their prefix cells**. Merely changing the distance used to choose a codeword changes assignments, but has not changed this structure. On the frozen host, metric changes moved assignments by 19.35% at level 0 and 3.5–6.9% per level in the level-wise probe, yet left the single-code (L1,L2) cell fraction essentially unchanged: 0.9208 (Euclidean) versus 0.9263 (Möbius). In other words, a candidate seeking joint movement must alter the *useful partition/information structure* rather than just the metric or ER neighborhood geometry.

The frozen artifact summary reinforces where the available structure is. There are 24,587 items; control level-0 uses 245/256 codes (largest cell 237; median cell 94, 99th percentile 203), level-1 uses 256 (largest 261), and level-2 uses 256 (largest 293). Of 16,147 prefix cells, 4,624 (28.64%) admit at least two codes, while 11,523 (71.36%) have one code; mean codes per prefix cell is 1.5227. The frozen ER input has shapes 52,855 anchors, 52,855 positives, and 52,855×10 negatives; latents are float32 with shape 24,587×768 (mean row norm 0.559254). These are measured opportunity/denominator facts, not a prediction of candidate success.

The numerical gates remain joint and strict: Hard-L1 ER > 16.38953942232631, Stage3 Recall@10 > 0.06406796775709883, and NDCG@10 > 0.03530651701940677. For scale, the failed cold level-0 product-geodesic arm had ER 1.0 versus archived control ER 16.334505666275888 and collapsed level 0 to one code for all 24,587 items; it received no Stage3 evaluation. Thus ER alone cannot establish joint success.

## 2. What the six closed families left untouched

The common untouched object, as measured, is **the useful SID/prefix-cell information structure that Stage3 can consume**. The six closures differ in their immediate failure:

1. **Product-geodesic level-0 assignment scored by codeword radius:** reduced the decision to a radius comparison and collapsed assignment to one code; it did not create a discriminative catalogue partition. The measured failed arm had one distinct level-0 code and ER 1.0.
2. **Horospheric/Busemann level sets:** the measured level gap did not separate the host's label; the target-relevant partition remained untouched.
3. **Codeword-frame transported residual:** ill-posed as proposed; its well-posed form is an identity, so it changes neither residual information nor assignments.
4. **Role-assigned token layout:** measured VESTIGIAL—the declared geometry was not in the forward computation. Therefore the effective representation/score and the information available to ranking were untouched.
5. **Gyro-composition/Möbius associator score:** rejected at definition level twice for type errors; no valid operator was applied, hence no measured change to the SID or ranking.
6. **Unsigned tangent-space wedge area:** parked after 14 adversarial rounds without unanimous approval. It is not refuted, but it is not a candidate; no accepted measurement establishes a change to the useful SID structure. LOTV (6b) is separately withdrawn: it is tie-break-only and no assignment-stream evidence established its preregistered ≥5% per-level tie rate. Do not treat either as a demonstrated mechanism.

Across the metric probes, the quantitative obstruction is explicit: Euclidean within-cell residual spread / level-3 nearest-codeword spacing = 0.7895, while Möbius displacement raises it to 1.4251, yet single-code (L1,L2) cells remain 0.9208 versus 0.9263. The measured conclusion is that changing query metric alters *which* codeword wins, not the trained codebook's capacity or the residual information structure. No closed family has demonstrated a change to what useful factor the three tokens encode.

## 3. Falsifiable explanation for “better ER, worse downstream”

**Hypothesis:** an ER gain can arise from a coarser or more tail-concentrated level-0 partition that improves the frozen hard-L1 positive/negative separation, while simultaneously erasing distinctions used by the Stage3 ranker. This is causal only if the ER-improving arm's changed assignments disproportionately merge items that Stage3 needs to rank separately.

**Discriminating measurement (pre/post, on identical frozen items and evaluation):** report (a) level-0 largest-cell share and cell-size quantiles, (b) number/share of multi-code prefix cells and per-prefix later-token entropy, and (c) Stage3 Recall@10/NDCG@10 stratified by whether each query's relevant item is in a newly merged level-0 cell versus an unchanged cell. The hypothesis predicts increased concentration/reduced within-prefix SID information, with the downstream loss concentrated in newly merged cells; if ER improves without those structural changes, or the ranking loss is not concentrated there, this explanation is falsified. The frozen control gives the baseline for that test: level-0 largest cell 237/24,587 = 0.9639%, median cell 94, 99th percentile 203; 4,624/16,147 prefix cells (28.64%) have ≥2 codes.

This is a testable explanation, not a proposed mechanism or a claim that the available measurements already prove the causal link.
