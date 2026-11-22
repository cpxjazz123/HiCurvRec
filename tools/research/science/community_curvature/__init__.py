"""Curvature-guided soft community learning (experiment 1).

The question this package answers is narrow and comes before any RQ-VAE change:
**can a mixed free + hyperbolic encoder learn item communities that agree with
the Augmented Forman-Ricci (AF3) curvature of the behaviour graph better than a
pure Euclidean encoder of the same capacity?**

Nothing here touches the quantizer, the Semantic IDs or the recommender. The
inputs are the frozen 768-dimensional sentence-T5 item embeddings and the
train-split behaviour graph; the output is a soft community distribution per
item plus the measurements that say whether the communities are real.
"""
