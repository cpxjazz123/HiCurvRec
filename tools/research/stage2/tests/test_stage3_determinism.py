"""A Stage-3 run has to be reproducible, and cuBLAS is what breaks it.

The fault this refuses: the trainer pins ``torch.use_deterministic_algorithms``
and ``SEED = 42``, yet two runs of the frozen baseline -- the same code, the
same seed, the same data -- ended at test_R@10 0.0555 and 0.0402. A spread of
38% relative, larger than every difference the mechanism campaigns have been
reporting, so no comparison against those numbers means anything until the
training is deterministic.

On CUDA >= 10.2 a cuBLAS GEMM is not deterministic unless the workspace
configuration is set, and torch says so in a warning on every run. The fix is
to pin it in the environment of the training subprocess.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SERVER = Path(__file__).resolve().parents[4] / "tools" / "genrec-mcp" / "server.py"

_spec = importlib.util.spec_from_file_location("genrec_mcp_server", str(SERVER))
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def test_the_stage3_environment_pins_the_cublas_workspace():
    env = server.stage3_child_env(Path("/tmp"), {"mode": "ddp4"})
    assert env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_an_explicit_workspace_configuration_is_not_overridden():
    import os

    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        env = server.stage3_child_env(Path("/tmp"), {"mode": "ddp4"})
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        if previous is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous
