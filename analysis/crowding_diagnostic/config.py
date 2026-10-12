from pathlib import Path

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
OUT_DIR = ROOT / "results/crowding_diagnostic"
EMBEDDINGS = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN_PARQUET = ROOT / "results/stage0_build_parquet/train.parquet"
VALID_PARQUET = ROOT / "results/stage0_build_parquet/valid.parquet"
ARMS = {
    "Exp1a": {
        "checkpoint": ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth",
        "sid_json": ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json",
        "sha256": "ff1623df671be0fe5443a9e5d9c922f256fe7e4905623c1b48af9fe43ec993fa",
    },
    "TIGER": {
        "checkpoint": ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth",
        "sid_json": ROOT / "results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json",
        "sha256": "018ef7b19d471714da3055395cd49f1920cd12efa14c72dd9ac8f0e04f6f54fb",
    },
}
PRIMARY_PAIRS = ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz"
SECONDARY_PAIRS = ROOT / "results/stage3_T5Train/monotonicity/behavior_pairs_s42.npz"
PAIR_SHA256 = {
    "primary_valid": "04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0",
    "secondary_train": "28eee576863bde599cfde76c99133a6732f9de8c31324c3d4603f32f1977709f",
}
LEVELS = ("L1", "L2", "L3")
TOP_K = (5, 10, 20, 50)
BOOTSTRAP_REPLICATES = 200
SEED = 42
