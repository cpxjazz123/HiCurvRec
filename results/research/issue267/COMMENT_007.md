## Pre-Stage3 score prediction frozen

- Frozen formula and ranges: `SCORE_CONTRACT.md` (sha256 in `SCORE_PREDICTION.json`).
- Stage2 endpoint: R `8.386929403059185e-05`, G_gate `0.10466435496425182`, G_L3 `0.0005665696095017952`; relative alpha `0.3075788718799832`.
- Normalized components: N_R `0.07529986029441926`, N_gate `1.1096520270125871`, N_L3 `2.089176670111412`; **S = 1.2921335084032926**.
- This is **above all ten historical scores** (max `0.7411182932`, arm 102); no clipping. Predicted interval: above the full historical range.
- Identity: fixed batch `251dced0…`, sample centre 15918/28 rows, parameters unchanged, optimizer steps 0; export was validated and rewritten only from the checkpoint through the existing `export_check` interface (238 initial token differences; manifest records them).
- Commit: `49d311d54f029dfc88db43fd5d581957c082160a`.
- Next: one frozen Stage3 150-epoch run; S is diagnostic only and will not decide whether to launch it.
