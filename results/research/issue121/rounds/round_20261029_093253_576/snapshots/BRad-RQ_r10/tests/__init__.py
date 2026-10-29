"""Tests of the candidate's own mechanism.

They are part of the candidate, not of the framework: the geometry, the codebook
estimator, the switches, the trainer's paths and the distributed step each get a
check that fails when the mechanism is wrong, not when it merely looks unusual.
"""
