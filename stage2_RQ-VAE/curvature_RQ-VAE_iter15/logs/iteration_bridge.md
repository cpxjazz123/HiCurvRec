# iteration_bridge (post-iter14 → iter15)
Dominant bottleneck: learnable `c_layer_scale` may drift from residual-intended u_l, confounding the layer-wise curvature hypothesis.
Forbidden: behavior branching (iter14).
Next: iter15 freeze u_l=[0.001,0.932889,1.0]; log c_l at 0/10k/50k/100k.
