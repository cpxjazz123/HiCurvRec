"""The parts of a Stage 2 candidate that are not research.

The frozen contract (data, results, launcher, ports, hyperparameters) lives
here, so that a candidate only carries the mechanism it studies.
"""
from . import checks, contract, launcher

__all__ = ["checks", "contract", "launcher"]
