# Failed controlled attempt (recorded, not hidden)

Attempt: archived R04 tree, its own `validation_entry.py` one-epoch four-rank diagnostic via `launcher.launch`.
Result: launcher exit 1 after 304.97 s, 0 rank reports.

```
23-[rank3]:   File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/validation_entry.py", line 493, in main
24-[rank3]:     training.main(diagnostic_epochs=1)
25-[rank3]:   File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/train_rqvae.py", line 458, in main
26-[rank3]:     DIAGNOSTIC_OBSERVER("first_step_before_backward", model=raw_module,
27-[rank3]:   File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/validation_entry.py", line 123, in __call__
28-[rank3]:     raise RuntimeError('gate weights do not cover the three experts')
29:[rank3]: RuntimeError: gate weights do not cover the three experts
--
33-[rank2]:   File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/validation_entry.py", line 493, in main
34-[rank2]:     training.main(diagnostic_epochs=1)
35-[rank2]:   File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/_archive/TIGER_role_conditioned_l3_rq/train_rqvae.py", line 458, in main
36-[rank2]:     DIAGNOSTIC_OBSERVER("first_step_before_backward", model=raw_module,
```

Cause: that observer asserts `model.last_weights` shaped `(N,3)`; the mechanism stores the per-level gate tensor `(N,2,3)`. The admission path uses `check_entry`, so the assertion was never exercised; the failure is a latent tree inconsistency, not a pipeline fault.

The canonical admission gate (`checks.run_all`) and a real four-rank training entry then ran green (see `controlled_entry.json`, `controlled_four_card.json`).
