"""One actual sampled production step with nested allocator and CUDA profiling."""
import time
BOOT_TIME = time.perf_counter()
import train_rqvae as training
from tools.issue252.profiling import run

if __name__ == '__main__':
    run(training, training.REPO_ROOT / 'results/research/issue252/optimized_profile', BOOT_TIME)
