"""Independent reference for the issue-273 same-centre difference loss.

The production helper (``model/delta.py``) builds the full difference matrix and takes the upper
triangle.  This reference visits every unordered pair explicitly in Python and converts through
plain floats, so the two computations follow different code paths; they must agree to float
tolerance (the same independence pattern the frozen issue-271 reference uses).
"""
from __future__ import annotations


def brute_force_same_centre_difference_loss(predicted, target, centre_population):
    """Brute-force evaluation of the centre-balanced same-centre difference loss."""
    values = []
    count = len(predicted)
    for left in range(count):
        for right in range(left + 1, count):
            difference_p = float(predicted[left]) - float(predicted[right])
            difference_y = float(target[left]) - float(target[right])
            values.append((difference_p - difference_y) ** 2)
    if not values:
        return 0.0
    return sum(values) / (float(centre_population) * float(len(values)))
