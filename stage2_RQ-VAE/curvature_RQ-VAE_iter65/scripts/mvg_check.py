"""Verify the unchanged TIGER RQ-VAE loss paths before the full DDP run."""
import sys
import tempfile
from pathlib import Path

import torch

ITER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ITER_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def require_nonzero_gradient(loss, parameters, label):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    require(
        any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        ),
        f"{label} produced no finite nonzero model gradient",
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training.set_seed(training.SEED)
    embeddings = training.maybe_apply_pca(
        training.load_embeddings(experiment.EMBEDDING_FILE), training.PCA_DIM, training.SEED
    )
    train_frame = training.pd.read_parquet(experiment.TRAIN_FILE)
    train_ids = training.np.unique(train_frame["target"].to_numpy(dtype=training.np.int64))
    train_embeddings = torch.from_numpy(embeddings[train_ids]).to(device)
    require(len(train_embeddings) >= max(training.CODEBOOK_SIZE), "Insufficient train rows for codebook init")

    model = RQVAE(training._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    if training.XAVIER_INIT:
        training.initialize_tiger_weights(model)
    with torch.no_grad():
        model.init_codebook(train_embeddings)

    with tempfile.NamedTemporaryFile(prefix="iter65_mvg_", suffix=".pth") as checkpoint:
        torch.save(model.state_dict(), checkpoint.name)
        model.load_state_dict(torch.load(checkpoint.name, map_location=device))

    model.train()
    batch = train_embeddings[: training.BATCH_SIZE_PER_RANK]
    reconstructed, quant_loss, _, _ = model(batch)
    total_loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
    require(total_loss.requires_grad and total_loss.grad_fn is not None, "Total loss is detached")
    parameters = tuple(model.parameters())
    require_nonzero_gradient(total_loss, parameters, "total loss")
    require_nonzero_gradient(recon_loss, parameters, "reconstruction loss")
    require_nonzero_gradient(quant_loss, parameters, "quantization loss")
    print(
        "MVG PASS: checkpoint reload; total/reconstruction/quantization losses have "
        "finite nonzero gradients on one training batch",
        flush=True,
    )


if __name__ == "__main__":
    main()
