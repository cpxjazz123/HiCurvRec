Correction to commit c61e1bcb

The commit message quoted the top-1/top-2 gap's maximum as 0.01304. The measured
value in results/research/issue150/marginal_scale_probe.json is **0.5500**. The JSON
was and is correct; only the summary sentence was wrong. The other numbers stand:
p5 9.26e-5, median 1.265e-3, p50 1.267e-3, p95 6.55e-3, share below the leverage
82.8%, tokens changed between the pi and uniform plans 25.1%, histograms 234 codes /
1.27% / 0.951 against 242 / 1.07% / 0.969.
