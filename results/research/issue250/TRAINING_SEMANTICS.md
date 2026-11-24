# Frozen training semantics

The original reconstruction batch is unchanged (four ranks, 1024 per rank, fixed original RQVAE loss). Each sample additionally carries its global item ID. No embedding-content ID guessing.

Training assignment on reconstruction batches preserves the actual original rules: first two argmin; last Sinkhorn with .003/50 and original global min/max cost centering. Balanced assignment is intentionally batch-dependent during optimization; this is not called an inference-global SID.

Curvature uses the inference-defined hard catalogue SID, not the reconstruction batch's balanced assignments. Every catalogue item is encoded with its fixed visible ego/gate, and all three codebooks assign in one global catalogue call. This remains meaningful if the framework installs assignment-floor guards: the model calls each real VQLayer._indices, not an unguarded duplicate rule. Structural-message chunks are checkpointed for backward; quantization is not performed independently in reference chunks. The curvature metric is fixed uniform expert geometry and has no direct gate input.

Full-reference moments are computed once per required endpoint, from all nodes in that endpoint's frozen component. Frozen target blocks are visited exactly, with their per-centre valid counts. Curvature centres are scheduled exactly once per epoch, split by step then global ID modulo world size. Multiplying the local centre mean by steps_per_epoch * world_size * local_centre_count / global_valid_centre_count compensates DDP rank averaging and unequal centre counts. Empty ownership contributes an explicit zero sum, not invented target values.

Initial target scale comes from existing full-target sufficient statistics. Common tangent scale is sqrt(mean catalogue latent squared norm). Initial lambda uses actual reconstruction and curvature gradients with respect to the encoder on fixed calibration samples, before any update. Calibration records are dynamic run evidence; they do not overwrite shared graph assets. No formal training is authorized by this issue.

Full catalogue gate creation, geometry costs and reference distance blocks use non-reentrant activation checkpointing. Runtime and peak memory must be reported from the real validation, not inferred from unit-test timing.
