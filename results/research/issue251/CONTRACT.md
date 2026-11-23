# Issue 251 frozen contract — 2026-11-24

Supersedes issue250's single-descriptor MLP gate. No formal Stage2/Stage3 or TIGER training. New candidate TIGER_gnn_attention_sectional_rq starts from frozen baseline/TIGER_RQ-VAE; semantic encoder/decoder, 768/32 dimensions, three 256-entry codebooks and original loss remain intact. Reusable geometry, full-reference loss and selection helpers are implementation components, not a new baseline or extra codebook capacity.

## Local tokens and attention

For each global item, two distinct induced directed subgraphs are used: full closed one-hop ego and full closed two-hop ego. Each subgraph separately passes through the same six-feature input projection and two width32 directed mean-message layers (self, incoming, outgoing, tanh), then its own whole-subgraph mean readout. This yields h[i,t,32], t=one-hop,two-hop. They are NOT pooled together or concatenated into a single gate input. Two scales are not called all Graphlets. Overlap reuses global node identity but each induced subgraph has local messages. Isolates have root-only valid tokens. Padding is masked. Frozen visible graph, node features and standardization from #249 are read, not recomputed.

Five trainable independent Queries Q[5,32], shared key/value Linear(32,32). For original token count T, append a fixed zero null token; it is valid only if all original tokens are invalid. Thus effective attention includes T+1 slots with an explicitly identified null slot. a=softmax_over_tokens(Q K^T / sqrt32), masked before normalization, shape [N,5,T+1]. H=sum_t a V, shape[N,5,32]. Shared score MLP32->32->1, tanh hidden, supplies s[N,5]; w=softmax_over_experts(s), shape[N,5]. No expert-specific score networks or topology-role labels. Initial independent orthogonal Queries are not claimed to know geometric meaning. Q/K/V/score/GNN all trainable. Null token is a fixed buffer, not an unused parameter.

Subgraph input permutations permute original attention slots and preserve w; the appended null slot remains fixed. Padding tokens receive exactly zero attention. Empty inputs are finite and assign all attention to the explicit null. Single valid token gives attention1 independently of Q (degenerate Q gradient zero is mathematically expected); attention discrimination is tested with multiple unequal tokens.

## Geometry and real hard assignment

Curvatures c=(-3,-1,0,1,3) fixed. Shared tangent 32, common frozen scale s=sqrt(mean_i ||encoder(x_i)||²) on initial real catalogue. Poincare exp uses tanh(sqrt(-c)||u||/2) and spherical exponential uses angle sqrt(c)||u||, u=x/s, making all origin distances ||u|| and Euclidean zero-curvature limit consistent. Hyperbolic and spherical reference computations use independent Lorentz and ambient-sphere formulas. Stable coincident, extreme, spherical-wrap and finite-gradient behavior required; no expert-specific fitted scales.

C_ij=sum_k w_ik d_k(exp_c(r_i/s),exp_c(e_j/s))². The actual frozen VQLayer._indices owns hard selection: original levels1/2 argmin, last training Sinkhorn(.003,50); framework-installed assignment, margin, codebook and separation guards are called on this same cost/indices. Inference catalogue is assigned globally before any subset query, preserving global SID even under framework floor guards. Training's balanced batch assignments are not called inference-global SIDs. Behavior-off supplies uniform weights on BOTH batch and catalogue routes. Curvature-off makes all expert costs Euclidean. No ablation training.

Smooth P uses same centered C: nonbalanced softmax with temperature .003 (the frozen Sinkhorn epsilon); balanced differentiable log-domain Sinkhorn with original marginals and iterations. q_soft=P E. q=hard.detach()+(q_soft-q_soft.detach()) is bit-exact hard forward and only soft-path derivative. dL/dtheta=(dL/dq)(d(P(C)E)/dC)(dC/dw)(dw/ds)(ds/dH)(dH/dQ,K,V,GNN). The original commitment/codebook terms use actual selected hard codewords. Residual r_next=r-q and q_final=sum three actual hard-forward codewords. No direct gate-dependent curvature metric supplies a bypass gradient.

## Full-reference supervision and DDP

Supervision/evaluation fixed squared dissimilarity is equal-weight mean of five expert squared distances, independent of trainable w. Its square root is reported as distance; no unsupported intrinsic metric claim. Kq=A.1 mean over every reference in the same frozen centre component, including centre/endpoints. KG and masks are from all 28,391,948 frozen rows/13 shards. No sampled or batch-local reference replacement.

Exact moment identity mu(x)=mean_a D²(a,x); K=mu(m)-.5(mu(b)+mu(c))+.25D²(b,c). Required endpoint moments stream full references once. Triple targets stream with original masks and centre denominators. Elementwise endpoint distances avoid unnecessary pairwise matrices. Non-reentrant activation checkpointing retains exact gradients across global GNN, cost, and reference chunks. Lcurv=mean_centres mean_valid_triples Huber((Kq-KG)/target_scale), delta1; target_scale=RMS from frozen mean/variance/count sufficient statistics. Missing centres excluded, not zero-filled.

Catalogue curvature q uses inference-global hard SIDs and their same-cost surrogate. Original reconstruction uses original training assignment protocol. Full-target centre schedule visits each valid centre once per epoch; step partitions followed by global ID modulo4 ownership. DDP average compensated by steps_per_epoch*world_size*local_centre_count/global_valid_centre_count. Empty ownership is an empty sum, not invented targets.

Lambda=||grad_encoder reconstruction|| / ||grad_encoder curvature|| from frozen initial real model and fixed real calibration batch/full references, then frozen. Nonfinite/zero norms fail. Report raw/weighted module norms, cosine and pre/post clipping. Controlled microstep uses original LR/clip, no search. Negative tests detach selection cost and freeze attention; positive chain must reach GNN,Q,K,V,score through actual A.1.

## Evidence and acceptance

C1–C9 are copied verbatim in ACCEPTANCE.json. Known-answer tolerance float64 1e-9 (well-conditioned smooth finite differences 1e-6); float32 hard forward bit-identical to decoded codeword sum. Four real GPUs and actual entry smoke required; short verification only, no 3000-epoch dispatch. Checkpoint/export bind source, run config, graph, features, targets, frozen scales and lambda. Missing or stale provenance fails reload. Independent hard-q and continuous-z diagnostics are labelled separately; historical TIGER Stage3 geometry remains NA when no same-generation checkpoint is available. No recommendation or fidelity improvement claim in this issue.

Paper/official implementation comparison is recorded separately in PAPER_COMPARISON.md and is part of this contract. Numerical and routing mutations must be caught by named assertions, never import errors/timeouts.
