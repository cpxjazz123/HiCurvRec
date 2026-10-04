"""Launch the C3-Train Stage2 condition with the fixed DDP configuration."""
import os
import sys

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SOURCE_DIR)

from scripts.run_stage2_curvature import (
    configure_run,
    _launch_via_torchrun,
    main,
)

configure_run(__file__)
if __name__ == "__main__":
    _launch_via_torchrun()
    main()
