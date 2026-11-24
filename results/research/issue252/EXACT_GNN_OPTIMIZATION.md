# Prospective exact GNN execution optimization

The first complete60-epoch benchmark measured mean2.2026225s including all actual work/eval50, not the required<=0.5s. No formal run starts. Next optimization preserves the phase2 mathematical contract, every node/edge, feature statistics, all learned parameters, full subgraph mean, optimiser order and actual per-step updates.

Let F be frozen standardized6-feature rows and h0=F W0^T+b0. For a complete ego's direction-normalized sparse operator A, A h0=(A F) W0^T+(A1)b0; A1 is1 for nonempty directional rows and0 for empty ones. Thus first DirectedMean preactivation can be evaluated as

F (Ws W0)^T + Ws b0 + bs + (Ain F)(Wi W0)^T + (Ain1)(Wi b0) + (Aout F)(Wo W0)^T + (Aout1)(Wo b0).

Only Ain F, Aout F and nonempty masks are static and may be cached. They derive solely from the already-verified complete frozen layouts and6 features; no learned tensor or activation is cached across updates. All matrix products involving W0/Ws/Wi/Wo/b0/bs are recomputed in the live autograd graph. The existing second nonlinear directional layer and per-ego mean stay unchanged. No additional parameter or feature dimension is introduced.

Compute static feature means once per immutable layout/device, in bounded blocks; preserve complete root/global item order and both scales. No new graph or disk asset is rebuilt. The frozen v9 generator/hash remains unchanged. Activation checkpoint replay may be removed only with independent values/all-parameter/input-gradient equality and observed memory headroom; retention is an execution choice, not a changed gradient or objective.

First write production-path failing tests for the avoidable32-wide first-layer sparse aggregation/replay, retaining existing independent dense value/gradient references. Float64 tolerances stay1e-9 abs/1e-8 rel and float32 1e-5 abs/1e-4 rel; no relaxing tests or claiming stochastic/full-loss equivalence. Re-measure inclusive60epoch benchmark after all exact changes. A faster instrumented step alone is not admission.
