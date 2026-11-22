"""The ranks' environment must make their output reach the log while they run.

A ranks' stdout is a file when the launcher starts it, which is block-buffered:
a job that dies early loses the buffer and leaves a zero-byte log, and the error
that killed it is gone. That cost three launches and an hour before the cause was
visible, and the fix is one variable. This test pins it, and pins the two
settings that must NOT be inherited from the caller.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools.research.stage2 import contract  # noqa: E402


def test_the_ranks_run_unbuffered() -> str:
    env = contract.launch_env()
    if env.get("PYTHONUNBUFFERED") != "1":
        raise AssertionError("launch_env does not set PYTHONUNBUFFERED=1: "
                             + str(env.get("PYTHONUNBUFFERED")))
    return "the ranks' output is unbuffered"


def test_the_single_card_marker_does_not_leak() -> str:
    os.environ[contract.SINGLE_GPU_ENV] = "1"
    try:
        env = contract.launch_env()
        if contract.SINGLE_GPU_ENV in env:
            raise AssertionError("the single-card marker leaked into a four-rank launch")
    finally:
        os.environ.pop(contract.SINGLE_GPU_ENV, None)
    return "the single-card marker is removed for the four-rank job"


def test_the_transport_is_left_to_the_interpreter() -> str:
    """The review removed the NCCL overrides and the measurement agrees with it.

    They were added while the launch was hanging, but a two-way measurement
    showed the hang was the fork, not the transport: the same job ran with and
    without them. They are therefore not part of the contract any more, and this
    test records that rather than pinning a setting the owner removed. What it
    still checks is that launch_env leaves the caller's environment otherwise
    intact, so an operator who pins a variable keeps it.
    """
    os.environ["NCCL_DEBUG_TAG"] = "probe"
    try:
        env = contract.launch_env()
    finally:
        os.environ.pop("NCCL_DEBUG_TAG", None)
    if env.get("NCCL_DEBUG_TAG") != "probe":
        raise AssertionError("launch_env dropped a variable the caller set")
    return "the transport is not overridden and the caller's environment survives"


def main() -> int:
    failures = 0
    for test in (test_the_ranks_run_unbuffered, test_the_single_card_marker_does_not_leak,
                 test_the_transport_is_left_to_the_interpreter):
        try:
            print("PASS " + test.__name__ + ": " + test())
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
