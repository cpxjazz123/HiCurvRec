"""Every Stage-2 run has to be reproducible, and cuBLAS is what breaks it.

The fault this refuses: ``torch.use_deterministic_algorithms`` is on in the
trainers, but on CUDA >= 10.2 a cuBLAS GEMM is not deterministic without
``CUBLAS_WORKSPACE_CONFIG``, and torch says so in a warning that is printed on
every run. The consequence was measured: arms without a sparse propagation
reproduce their numbers to the last digit across runs, arms with one drift by
several percent -- larger than the differences between the arms, so every real
effect disappears into the noise.
"""
from __future__ import annotations

from tools.research.stage2 import contract


def test_the_launch_environment_fixes_the_cublas_workspace():
    env = contract.launch_env()
    assert env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_an_explicit_workspace_configuration_is_not_overridden():
    env = contract.launch_env({"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_the_environment_is_otherwise_unchanged():
    env = contract.launch_env()
    assert env.get("OMP_NUM_THREADS") == "4"
