"""Launch the iter81 quantized-behavior Stage2 condition."""
import os
import sys

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SOURCE_DIR)

import curvature_config as experiment

MECHANISM_ITER = os.path.basename(SOURCE_DIR).rsplit("_iter", 1)[-1]
for name in ("STAGE2_RESULT_DIR", "STAGE2_LOG_DIR", "STAGE3_RESULT_DIR"):
    value = str(getattr(experiment, name))
    if f"_iter{MECHANISM_ITER}" not in value:
        raise SystemExit(
            f"[iter{MECHANISM_ITER}] {name} points outside this iteration: {value}"
        )
if f"iter{MECHANISM_ITER}" not in experiment.MECHANISM_NAME:
    raise SystemExit(
        f"[iter{MECHANISM_ITER}] MECHANISM_NAME is {experiment.MECHANISM_NAME!r}"
    )

from scripts.run_stage2_curvature import configure_run, _launch_via_torchrun, main

configure_run(__file__)
if __name__ == "__main__":
    _launch_via_torchrun()
    main()
