"""Build the round-5 dynamics arms from the frozen F0 tree.

Round 5 keeps the quantiser exactly as the baseline has it -- argmin, no Sinkhorn --
and only changes how fast the encoder's *output* is allowed to move between epochs.
The three arms isolate the two candidate knobs:

    D1_enc_lr        encoder learning rate x0.1 for the first 20 epochs
    D2_drift         L_drift on the per-item output, argmin untouched
    D3_drift_cb_lr   D2 plus a codebook learning rate x3

The trees are built by the same exact-string replacement helper round 2 uses, and the
manifest records every file's sha256 so "only the intervention changed" stays checkable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / "round2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = _load("qda_build", ROUND2 / "build_arms.py")

SCREEN = build.SCREEN
OUT = HERE / "trees"

#: The constants, inserted once per arm.  LAMBDA_DRIFT is calibrated from numbers that
#: were already recorded: at epoch 1 the reconstruction loss is 4.67e-4 and the mean
#: relative squared drift is (0.105)^2 = 1.1e-2, so lambda = 0.042, rounded to 0.04.
#: The penalty then equals the reconstruction term at the observed initial drift and
#: decays quadratically as the drift falls.  No outcome enters the choice.
def dynamics(enc_lr_scale: float, cb_lr_scale: float, tau: float = 0.0) -> str:
    """The per-arm constants.  Only D1 lowers the encoder's learning rate; D2/D3 must
    leave it alone so the drift penalty is the only thing acting on the encoder."""
    return f"""
#: Round 5 dynamics constants (derived; see round5/build_d_arms.py).
LAMBDA_DRIFT = 0.04
ENC_LR_EPOCHS = 20
ENC_LR_SCALE = {enc_lr_scale!r}
CB_LR_SCALE = {cb_lr_scale!r}
#: Hinge threshold: 0 means the plain quadratic penalty (D2/D3).
DRIFT_TAU = {tau!r}
"""

ARMS = {
    "D1_enc_lr": {"enc_lr": (20, 0.1), "drift": False, "cb_lr": 1.0},
    "D2_drift": {"enc_lr": None, "drift": True, "cb_lr": 1.0},
    "D3_drift_cb_lr": {"enc_lr": None, "drift": True, "cb_lr": 3.0},
    #: Only the displacement above the codebook's measured tracking limit is penalised.
    "E1_hinge_tau": {"enc_lr": None, "drift": "hinge", "cb_lr": 1.0, "tau": 0.01},
}

GROUPED_OPTIMIZER = (
    """    if OPTIMIZER.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)""",
    """    # Round 5: encoder / codebook / decoder get their own learning rates.  AdamW
    # normalises each coordinate's step by its own second moment, so scaling the
    # gradient does not change the step size; only a separate lr does.
    encoder_ids = {id(parameter) for parameter in raw_module.encoder.parameters()}
    codebook_ids = {id(parameter) for layer in raw_module.rq.levels
                    for parameter in layer.parameters()}
    encoder_params, codebook_params, decoder_params = [], [], []
    for parameter in model.parameters():
        if id(parameter) in encoder_ids:
            encoder_params.append(parameter)
        elif id(parameter) in codebook_ids:
            codebook_params.append(parameter)
        else:
            decoder_params.append(parameter)
    groups = [
        {"params": encoder_params, "lr": LR},
        {"params": codebook_params, "lr": LR * CB_LR_SCALE},
        {"params": decoder_params, "lr": LR},
    ]
    if OPTIMIZER.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(groups, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)""",
)

GROUPED_SCHEDULER = (
    """    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)""",
    """    steps_per_epoch = max(1, len(loader))

    def encoder_lambda(step: int) -> float:
        epoch = step // steps_per_epoch + 1
        return lr_lambda(step) * (ENC_LR_SCALE if epoch <= ENC_LR_EPOCHS else 1.0)

    def codebook_lambda(step: int) -> float:
        return lr_lambda(step)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[encoder_lambda, codebook_lambda, lr_lambda])""",
)

DRIFT_IMPORT = (
    """from qda_gradsrc import measure""",
    """from qda_gradsrc import measure
from qda_drift import drift_loss, hinge_drift_loss, snapshot""",
)

DRIFT_DATASET = (
    """    train_dataset = TensorDataset(train_embeddings)""",
    """    # The drift term needs each item's own previous representation, so the loader
    # must carry the row index as well as the embedding.
    train_dataset = TensorDataset(train_embeddings,
                                  torch.arange(len(train_embeddings)))""",
)

DRIFT_LOOP = (
    """        for batch_index, (batch,) in enumerate(loader):""",
    """        for batch_index, (batch, item_index) in enumerate(loader):""",
)

DRIFT_LOSS = (
    """            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)""",
    """            loss, recon_loss = raw_module.compute_loss(batch, reconstructed, quant_loss)
            if LAMBDA_DRIFT > 0:
                # One extra encoder pass: the quantiser stays argmin, only the output's
                # epoch-to-epoch displacement is penalised.
                if DRIFT_TAU > 0:
                    loss = loss + LAMBDA_DRIFT * hinge_drift_loss(
                        raw_module.encoder(batch), z_prev[item_index.to(device)], DRIFT_TAU)
                else:
                    loss = loss + LAMBDA_DRIFT * drift_loss(
                        raw_module.encoder(batch), z_prev[item_index.to(device)])""",
)

DRIFT_INIT = (
    """    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))""",
    """    with torch.no_grad():
        raw_module.init_codebook(train_embeddings.to(device))
    #: Every training item's representation at the end of the previous epoch.
    z_prev = snapshot(raw_module.encoder, train_embeddings.to(device))""",
)

DRIFT_REFRESH = (
    """            recons.append(float(recon_loss.detach()))
        epoch_time_s""",
    """            recons.append(float(recon_loss.detach()))
        with torch.no_grad():
            z_prev.copy_(raw_module.encoder(train_embeddings.to(device)))
        epoch_time_s""",
)


def digest_tree(root: Path) -> dict:
    return build.digest_tree(root)


def main() -> None:
    manifest = {"source": str(build.SOURCE), "arms": {}}
    for name, spec in ARMS.items():
        target = OUT / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(build.SOURCE, target,
                        ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"))

        pairs = [p for p in build.COMMON]
        pairs[0] = (pairs[0][0], pairs[0][1].replace("{full!r}", repr(SCREEN))
                    .replace("{gradsrc!r}", "False"))
        # round-5 runs live beside round 2, not inside it
        pairs[1] = (pairs[1][0], pairs[1][1].replace('"round2"', '"round5"'))
        pairs += [
            ('ARM_NAME = "QDA_F0_seed42"', f'ARM_NAME = "{name}"'),
            ('EPOCHS              = 3000', 'EPOCHS              = 300'),
            ("""OBSERVATIONS   = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                  2250, 2500, 2750, 3000)""",
             f"OBSERVATIONS   = {SCREEN!r}"),
            # constants first: the patches below reference them
            ("QDA_GRADSRC_EPOCHS = 60",
             "QDA_GRADSRC_EPOCHS = 60" + dynamics(
                 spec["enc_lr"][1] if spec["enc_lr"] else 1.0, spec["cb_lr"],
                 spec.get("tau", 0.0))),
            DRIFT_IMPORT,
        ]
        if spec["enc_lr"] or spec["cb_lr"] != 1.0:
            pairs += [GROUPED_OPTIMIZER, GROUPED_SCHEDULER]
        if spec["drift"]:
            pairs += [DRIFT_DATASET, DRIFT_LOOP, DRIFT_LOSS, DRIFT_INIT, DRIFT_REFRESH]

        patches = {"train_rqvae.py": build.rewrite(target / "train_rqvae.py", pairs, name)}
        patches["qda_observer.py"] = build.rewrite(
            target / "qda_observer.py",
            [build.schedule_patch(SCREEN)] + build.OBSERVER, name)
        shutil.copy2(ROUND2 / "telemetry_src.py", target / "qda_telemetry.py")
        shutil.copy2(HERE.parent / "round3" / "gradsrc_src.py", target / "qda_gradsrc.py")
        shutil.copy2(HERE / "drift_src.py", target / "qda_drift.py")
        patches["qda_drift.py"] = ["installed from round5/drift_src.py"]

        manifest["arms"][name] = {"epochs": 300, "observations": list(SCREEN),
                                  "patches": patches, "digests": digest_tree(target)}
        print(f"built {name}")

    (HERE / "arm_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                            encoding="utf-8")
    print("wrote arm_manifest.json")


if __name__ == "__main__":
    main()
