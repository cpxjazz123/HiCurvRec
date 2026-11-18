"""CNCQ: certified non-collapse quantisation.

Three layers, each with a different kind of guarantee:

    capacity.hard_capacity_assignment   constructive: every codeword >= m items
    geometry.bounded_codes             structural: no two codewords can coincide
    acceptance.certify                 adjudicative: a collapsed model is never accepted

The first two hold for any training mechanism; the third cannot, because no quantiser can
recover semantics the encoder never produced. What it can do is refuse to deliver such a
model.
"""
from .acceptance import Policy, certify
from .capacity import InfeasibleCapacity, hard_capacity_assignment
from .geometry import (bounded_codes, certify_separation, mixed_min_distance,
                       separation_bound)

__all__ = ["Policy", "certify", "InfeasibleCapacity", "hard_capacity_assignment",
           "bounded_codes", "certify_separation", "mixed_min_distance", "separation_bound"]
