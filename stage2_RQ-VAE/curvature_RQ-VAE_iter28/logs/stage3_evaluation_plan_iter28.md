# stage3_evaluation_plan_iter28

## Scope

The completed iter28 Stage2 produced 4-token TIGER-compatible SIDs
(`sids_for_hgrec.npy`, shape `(24587, 4)`) and a JSON
(`item_sids.json`).  Stage3 must consume these with the unchanged
HG-Rec protocol and produce a comparable `test_final.json`.

## Launch command (zero CLI args; no env overrides)

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter28
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  scripts/run_stage3_iter27.py \
  > logs/_stage3_run.log 2>&1 &
```

The script `scripts/run_stage3_iter27.py` imports
`stage3_T5Train/train_HG-Rec.py`, sets `CODE_PATH`,
`RQVAE_VARIANT`, `LOG_PATH`, `SAVE_PATH`, and the NCCL env, then
forks `torchrun --standalone --nproc_per_node=4 --master_port=50201`.
The actual Stage3 training logs land in
`results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/_stage3_launcher.log`.

## Inputs

| Input | Path |
|---|---|
| 4-token SID JSON | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/item_sids.json` |
| 4-token SID numpy | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/dataset/Instruments/sids_for_hgrec.npy` |
| Stage3 code commit | `bdcbbf9` (iter28 launch HEAD) |

## Expected outputs

| Output | Path |
|---|---|
| Run log | `results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/_stage3_launcher.log` |
| Best checkpoint | `results/stage3_T5Train/curvature_RQ-VAE_iter28/ckpt/Amazon_2023_Instruments/.../HG_Rec_best.pth` |
| Training metrics | `results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/.../training_metrics.jsonl` |
| Test final | `results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/.../test_final.json` |

## Expected test_final.json content

```json
{
  "best_checkpoint": "results/stage3_T5Train/curvature_RQ-VAE_iter28/ckpt/.../HG_Rec_best.pth",
  "n_eval": 57439,
  "test_recall@5": float,
  "test_recall@10": float,
  "test_ndcg@5": float,
  "test_ndcg@10": float
}
```

## Stage3 protocol (inherited from iter18)

- seed = 42
- n_epochs = 150 (with `early_stop = 20`)
- batch_size = 3072
- learning_rate = 1e-4
- weight_decay = 1e-4
- mixed_precision = bf16
- beam = 20
- n_eval = 57439

## Stop conditions

- Final evaluation `test_recall@10 > 0.065` ⇒ promote (adoption target).
- Final evaluation `test_recall@10 ∈ (0.0599, 0.065]` ⇒ `ACTIVE_NEUTRAL` (near-parity with iter18).
- Final evaluation `test_recall@10 < 0.0599 - 0.003 = 0.0569` ⇒ `ACTIVE_NEGATIVE` (regression by ≥ iter18 noise band).
- Crash / NaN / contract violation ⇒ `ITERATION_ABORTED_INFEASIBLE` (per SKILL.md §2.9).

## Comparators (protocol-compatible)

| Iter | R@10 | n_eval |
|---|---:|---:|
| iter11 | 0.05976775 | 57439 |
| **iter18** (canonical baseline) | **0.05988962** | 57439 |
| iter25 | 0.05882763 | 57439 |
| iter26 | 0.057017 | 57439 |