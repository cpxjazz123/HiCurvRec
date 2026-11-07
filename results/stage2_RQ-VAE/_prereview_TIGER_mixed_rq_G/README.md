# Pre-review artifacts of TIGER_mixed_rq_G

These are the artifacts of the run that was made **before** the review commit
`3ee9374c`. The review changed this arm's mechanism path (52 files, +7498/-1238
across the trained arms, including `train_rqvae.py` and every mechanism module)
and fixed defects on it: the repeated squared cost, the tangent-to-ball mapping,
the incremental degree dependencies, the rank-zero codebook initialisation and
the alpha rollback.

The numbers these files hold therefore describe a mechanism that is no longer in
the repository, and they are kept only so that nothing is lost and the re-run
can be compared with them. The live results of the arm live in
`results/stage2_RQ-VAE/TIGER_mixed_rq_G/` and come from the reviewed source.

The re-run is a repair of the source, not a second experiment: same seed, same
frozen budget, same one run. The comparison is not a selection over seeds.
