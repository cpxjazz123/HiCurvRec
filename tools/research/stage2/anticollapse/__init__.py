"""Make codeword death unreachable for any Stage2 mechanism.

Every mechanism hands its quantiser one thing: a cost matrix `(items, codewords)`. The
assignment that follows decides which codewords stay alive, and the baseline's rule --
plain `argmin` -- has an absorbing state: a codeword that loses its last item receives
exactly zero codebook gradient (the codebook term detaches the residual), so it can never
come back. Round 1 measured that this kills 193 of 256 codewords within 50 epochs.

This package replaces *only* the assignment step with the cheapest one that also gives every
codeword at least `floor` items. Loss, distance, curvature, codebook initialisation, encoder
and every hyper-parameter stay exactly as the candidate wrote them, which is why it works for
any mechanism rather than for one.

Two halves:

    floor.constrained_assignment          the algorithm and its guarantee
    patch_tree.apply_floor / verify_tree  install it into a candidate tree, then prove it
                                          works by running the patched code on a collapsed
                                          cost matrix in a fresh interpreter

The patcher is idempotent and fails closed: a tree whose quantiser it does not recognise is
reported as such instead of being silently left unprotected.
"""
from .floor import InfeasibleFloor, constrained_assignment
from .patch_tree import FAMILIES, UnknownTree, apply_floor, verify_tree

__all__ = ["InfeasibleFloor", "UnknownTree", "constrained_assignment", "FAMILIES",
           "apply_floor", "verify_tree"]
