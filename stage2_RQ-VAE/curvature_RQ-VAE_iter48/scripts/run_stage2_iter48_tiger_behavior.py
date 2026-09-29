"""Run TIGER with Euclidean residual behavior contrastive loss."""
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
import train_euclidean_tiger as trainer

trainer.configure_arm("BEHAVIOR", __file__)

if __name__ == "__main__":
    trainer.launch_and_run()
