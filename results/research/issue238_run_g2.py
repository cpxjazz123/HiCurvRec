import sys
sys.path.insert(0, '/fs04/ar57/wenyu/GeneRec')
from tools.research.stage2 import launcher
code = launcher.launch(
    'stage2_RQ-VAE/TIGER_g237_sem_rq/train_rqvae.py',
    'results/stage2_RQ-VAE/TIGER_g237_sem_rq')
print('exit', code)
