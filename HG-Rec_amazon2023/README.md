# HG-Rec
The code of [ICML2026] Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook Strategy

## RecBole3.0 vanilla TIGER alignment

`train_HG-Rec.py` is the RecBole-baseline-compatible T5 runner.  It uses
zero-based item ids, maps SID row `i` to item `i`, shifts every raw SID value
by `+1` in one shared namespace, keeps one sample per train prefix (including
the empty-history prefix), and right-pads `[user, history SID..., EOS]`.

The default fast A100 protocol is `batch_size=10240`, `infer_size=64`,
`n_user_tokens=1`, `beam=20`, `top-k=[5,10]`, BF16, cuDNN fast paths,
`torch.compile`, and validation after every epoch. The best checkpoint is
selected by validation `NDCG@10`, with early-stopping patience 20, followed by
one final test. It uses AdamW (`lr=0.003`, `weight_decay=0.05`) for up to 150
epochs. Use
`--no_compile`, `--no_fast`, and `--no_bf16` to disable the corresponding
throughput settings.

The default run validates every epoch, saves `HG_Rec_best.pth` by the highest
validation `NDCG@10`, applies `--early_stop` to validation stagnation, and runs
the test split once after training. Use `--no_eval` for the faster loss-only
training mode.

Example for the current A100:

```bash
python train_HG-Rec.py
```

The command above uses all of the fast defaults listed below; explicit CLI
values can still override any individual setting.

For beam 20, use the conservative `--infer_size 64` for validation: the
autoregressive beam search becomes slower at very large inference batches,
even when the GPU has enough memory.  The training batch can still be chosen
independently, for example `--bf16 --batch_size 10240 --infer_size 64` on the
80 GiB A100.

For strict parity, export the already prepared RecBole frames first (this uses
RecBole's own raw parser, remapping, split and history builder):

```bash
PYTHONPATH=../RecBole3.0/src python export_recbole_tiger_data.py
```

If `Instruments.inter.json` is known to be the same remapped sequence source,
the local non-destructive fallback is:

```bash
python data/process_Amazon2023.py
```

Verify the resulting files:

```bash
python verify_recbole_alignment.py
```

Then run `train_HG-Rec.py`.  The historical
`train_HG_Rec_amazon2023_ddp.py` entry point forwards to the same protocol.
