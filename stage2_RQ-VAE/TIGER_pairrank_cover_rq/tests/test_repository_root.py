"""Issue #274: the trainer's repository-root resolution must survive the immutable run layout.

The formal run executes the tree's copy under
``results/stage2_RQ-VAE/_runs/<run_id>/<arm>/src/``.  Walking up from there reaches
``results/`` first, and ``results/`` contains a ``stage2_RQ-VAE`` directory but no
``tools/research/stage2/contract.py``.  A resolution that stops at the first ``stage2_RQ-VAE``
marker therefore points the embedding file at ``results/stage1_GeneEmbedding/...``, which does not
exist, and the real entry dies with a FileNotFoundError before the first batch.

The resolution must require **both** markers, exactly as the admitted trees do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_rqvae as training


def test_the_repository_root_has_both_markers():
    root = training._repository_root()
    assert (root / "stage2_RQ-VAE").is_dir()
    assert (root / "tools/research/stage2/contract.py").is_file()
    assert (root / "stage1_GeneEmbedding/output/sentence_t5.npy").is_file()


def test_the_resolution_would_stop_at_the_results_directory_without_the_second_marker():
    # the intermediate directory that the run layout walks through
    results = training._repository_root() / "results"
    assert (results / "stage2_RQ-VAE").is_dir()
    assert not (results / "tools/research/stage2/contract.py").exists()


def test_the_embedding_path_resolves_to_the_real_file():
    assert training.EMBEDDING_FILE.is_file()
    assert training.EMBEDDING_FILE.name == "sentence_t5.npy"
