"""Agent contracts for the Issue #120 round: ideate, probe, score, implement.

The prompts state what each role must return and what it must not do. There is
no implementation agent and no review agent: the main agent writes the selected
mechanism and registers it, and verification is the framework's MACHINE
admission (``tools/research/stage2/checks.py``), which executes real checks at
launch instead of asking a model for an opinion.
"""
from __future__ import annotations


#: The research objective of Issue #121, hard coded into every role. It is not
#: an optional free-text objective: the ideas, the scores, the implementations
#: and the reviews are all about this one coupling.
RESEARCH_OBJECTIVE = (
    "Discover mechanisms in which the Graphlet TYPE and per-node ROLE labels of strict "
    "induced subgraphs (K3, P3, C4, P4, K1,3) are learned by a joint cross-entropy that "
    "passes through the REAL RQ-VAE quantizer -- hard indices forward, a codebook "
    "gradient backward -- so that the exported hard Semantic IDs keep the behaviour "
    "graph's local topology and Stage3 test Recall@10 stays above 0.065."
)

#: The scope contract new proposals are judged against (Issue #220 mainline).
#: The historical behaviour x hyperbolic contract of Issue #121 lives in
#: `scope.check_scope`; new work uses `scope.check_mainline_scope`.
SCOPE_CONTRACT = (
    "A proposal is IN SCOPE only when all of this holds:\n"
    "  1. SUPERVISION: an instance-level joint cross-entropy over the Graphlet TYPE "
    "label of {v1..vk} and the per-node ROLE (orbit) labels. A pairwise recast "
    "(six-pair P4 edge classification, BCE/InfoNCE/margin over item pairs) is NOT "
    "the mainline objective; graphlets are the sampling and diagnostic unit.\n"
    "  2. PREDICTOR INPUT: it names which representation the head reads (encoder z, "
    "q_L1, q_L1+L2, q_full) and keeps type logits permutation-invariant and role "
    "logits permutation-equivariant; the raw adjacency, labels or node-position ids "
    "must not be fed into the head.\n"
    "  3. QUANTIZER PATH: the same loss must act on the real quantizer -- hard "
    "indices from the training assignment (Sinkhorn / assignment floor per the "
    "frozen protocol) with a hard-forward / soft-backward code assignment proxy -- "
    "so the gradient reaches the encoder AND the actually selected codewords. A "
    "second argmin quantizer in an auxiliary branch does not qualify.\n"
    "  4. ASSIGNMENT STAYS: the baseline's Sinkhorn-balanced level assignment must "
    "not be replaced by a plain argmin (CLAUDE.md 19h).\n"
    "  5. OBSERVABLES: heldout type/role macro-F1, hard-SID structure retention "
    "(null-adjusted pair gain, true-edge/triangle retention), collision and "
    "reconstruction, and Stage3 Recall when the arm is run downstream. Classifier "
    "accuracy alone is not success.\n"
    "  6. ONE RUN: each candidate is trained once (seed=42, 3000 epochs, four-card "
    "global batch 4096); no ablation sweep, no multi-seed selection, no tuning "
    "against heldout SID or Stage3 results.\n"
)

