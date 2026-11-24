## Stage3 MCP refused the corrected SID; no gate waiver

Actual call: `stage3_start` on `results/stage2_RQ-VAE/_runs/issue255_formal_r1/TIGER_prefix_curvature_gate_rq/result/item_sids.json`, with no optional override.

```json
{"started":false,"collision":0.9814129417985114,"collision_gate_waived":false,"refused":"SID quality gate: three-level collision 0.981413 is not below 0.40: 98.1% of the items share an SID with another item"}
```

Exact refusal and SID/checkpoint hashes: `results/research/issue255/STAGE3_REFUSAL.json`. Continuing to a frozen Stage3 final test now requires violating the frozen collision rule; the existing authorization does not grant a waiver or another formal training run. Recall, paired-user changes and a complete four-piece R2 archive therefore cannot be claimed.

All reachable work continues first: repair/verify the historical catalogue extractor, full mechanism evaluation and independent reductions, vector diagnostics, final evidence-based verdict and scoped push. The issue is not being closed merely because a tool failed; the only eventual blocked closure is the verified frozen-protocol conflict, after the reachable work is delivered.
