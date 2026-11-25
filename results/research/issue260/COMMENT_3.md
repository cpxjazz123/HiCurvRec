### #260 R06 frozen: topology token residual on the L2 assignment target

**Diagnosis** (`DIAGNOSIS.md`, `diagnosis/r05_probe.json`): the R05 cost channel is assignment-inert.
With the same checkpoint/encoder/codebooks, behavior off changes 0.11% of L2 codes, curvature off
1.69%, gate uniform 0.50%. Role mixtures stayed near-uniform (entropy 1.508–1.584 vs 1.585), the L3
gate collapsed (entropy 0.176; 99.59% one expert), and tokens add 0.026 role-classification accuracy
over degree only. Per-level quantisation MSE is 1.91e-5 / 5.39e-6 / 2.32e-6.

**Single mechanism** (`CONTRACT.md`): add the masked GNN token mean through an identity-initialized
32×32 projection to the **level-2 assignment target**:
`C_2(i,c)=Σ_e m_{r(i),e} d_e(r_i^2 + P t_i, c)`.
L1/L3 costs and the Sinkhorn/floor assignment rule are unchanged. Behavior signal: frozen local GNN
tokens; curve operation: existing three-expert mixed-curvature cost; same path. Switches exist in code
and will be tested, not trained as ablation arms.

**Local check before training** (`diagnosis/token_residual_probe.json`): identity projection changes
99.94% of L2 codes, L2 prefixes 9609→4082, triples 23228→7448, L1 unchanged; L2 positive behavior rate
0.00360→0.01071, L3 0.000416→0.00878. Pre-registered Stage2 signature: L2 prefixes ≤6000; L2 positive
rate ≥0.006; L3 positive ≥0.002; L1 unchanged. If this fails, no Stage3 is launched and no test is read.

**Six answers**: (1) inert assignment channel; (2) probe measurements above; (3) token mean directly
shifts the target before the nearest-codeword decision; (4) R03–R05 changed cost weights, R06 changes
the target; (5) validation-side signature then one final test Recall@10 >0.065; (6) if negative, learn
whether topology-target shift is insufficient and diagnose reconstruction/label quality separately.

Arm: `TIGER_topology_residual_l2_rq` (R05 archived to `_archive/..._r05`; only one active arm).
Formal protocol unchanged: 3000 epochs/4 cards/seed42/collision<0.40; Stage3 150/beam20; one test.
