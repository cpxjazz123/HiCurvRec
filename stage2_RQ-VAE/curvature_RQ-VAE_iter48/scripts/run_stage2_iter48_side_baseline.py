"""Train the matched 40k fixed-curvature baseline for the Iter48 side study."""
import os
import sys

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOURCE_DIR)

from train_rqvae import configure_run, _launch_via_torchrun, main

configure_run("BASELINE", __file__)

if __name__ == "__main__":
    _launch_via_torchrun()
    main()
