"""Build the round-2 diagnostic arms from the frozen F0 tree.

Every arm is a copy of `trees/F0_seed42` plus a fixed list of exact string
replacements. The script refuses to write a tree when any replacement does not apply
exactly once, and records the per-file sha256 of both sides in `arm_manifest.json`, so
"the only difference is the intervention" is checkable rather than asserted.

Arms
  A_fine          F0 + per-epoch telemetry + 5-epoch observations over 0..100, 3000 epochs
  B0_replay       A's instrumentation, 300 epochs, no intervention (the matched baseline)
  B1_enc_freeze   encoder frozen for epochs < 50 (the observed collapse window)
  B2_code_freeze  codebook entries frozen for epochs < 50
  B3_l1_balanced  level 1 gets the baseline's own Sinkhorn assignment
  B4_revival      L1 codes with no assignment are re-seated every 25 epochs (L1 only)
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "trees" / "F0_seed42"
OUT = HERE / "trees"

#: Epochs whose collapse window is already measured (QDA round 1: 201 -> 9 codes by
#: epoch 50). Both freeze arms use exactly that window; revival runs at half of it,
#: the coarsest schedule that acts twice inside the window.
FREEZE_EPOCHS = 50
REVIVE_EVERY = 25

COMMON = [
    # ---- configuration -------------------------------------------------
    ("""QDA_MONITOR = True""",
     """QDA_MONITOR = True
#: Per-epoch telemetry (round 2): who moves faster, encoder or codebook, and whether
#: dead codes still receive gradient.  One extra probe forward per epoch.
QDA_TELEMETRY = True
QDA_PROBE_ROWS = 4096
#: Gradient attribution (round 3): split the encoder's gradient by loss term.
QDA_GRADSRC = {gradsrc!r}
QDA_GRADSRC_EPOCHS = 60
#: The dense points keep only the summary; these keep the full npz snapshot.
QDA_FULL_OBSERVATIONS = {full!r}
#: The diagnostic round writes beside the first QDA round, never into an arm tree.
QDA_ROUND2 = QDA_ROOT / "round2"
"""),
    ("""RESULT_DIR = QDA_ROOT / "runs" / ARM_NAME""",
     """RESULT_DIR = QDA_ROOT / "round2" / "runs" / ARM_NAME"""),
    ("""from model.af3_runtime import (input_identity, resolve_shared_readout,
                               snapshot_sha256, snapshot_state)""",
     """from model.af3_runtime import (input_identity, resolve_shared_readout,
                               snapshot_sha256, snapshot_state)
from qda_telemetry import EpochTelemetry
from qda_gradsrc import measure"""),
    # ---- telemetry construction ----------------------------------------
    ("""    final_collision = float("inf")
    final_raw_unique = 0""",
     """    final_collision = float("inf")
    telemetry = (EpochTelemetry(raw_module, all_embeddings[:QDA_PROBE_ROWS].to(device))
                 if QDA_TELEMETRY else None)
    final_raw_unique = 0"""),
    # ---- the batch loop keeps its tokens -------------------------------
    ("""            reconstructed, quant_loss, _, _ = model(batch)""",
     """            reconstructed, quant_loss, _, tokens = model(batch)"""),
    ("""            loss.backward()
            if watching:""",
     """            loss.backward()
            if telemetry is not None:
                telemetry.step(tokens, [layer.embed.weight.grad
                                        for layer in raw_module.rq.levels])
                if last_batch:
                    with torch.no_grad():
                        telemetry.step_z(raw_module.encoder(batch))
            if watching:"""),
    # ---- per-epoch record ----------------------------------------------
    ("""        epoch_time_s = round(time.time() - _epoch_t0, 3)""",
     """        epoch_time_s = round(time.time() - _epoch_t0, 3)
        if telemetry is not None:
            _record(rank, "qda_epoch", **telemetry.finish(epoch))"""),
    # ---- observation keeps the npz only at the frozen full points ------
    ("""            summary = observe(raw_module, all_embeddings.to(device), 0, QDA_DIR, CURVATURE)""",
     """            summary = observe(raw_module, all_embeddings.to(device), 0, QDA_DIR, CURVATURE,
                              light=0 not in QDA_FULL_OBSERVATIONS)"""),
    ("""                summary = observe(raw_module, all_embeddings.to(device), epoch, QDA_DIR, CURVATURE)""",
     """                summary = observe(raw_module, all_embeddings.to(device), epoch, QDA_DIR, CURVATURE,
                                  light=epoch not in QDA_FULL_OBSERVATIONS)"""),
    # ---- lifecycle at the end ------------------------------------------
    ("""        _record(rank, "train_end",""",
     """        if telemetry is not None:
            _record(rank, "qda_lifecycle", **telemetry.close(EPOCHS))
        _record(rank, "train_end","""),
]

#: The trainer's observation guard is `epoch in QDA_OBSERVATIONS`, imported from
#: `qda_observer.OBSERVATIONS` -- so the schedule must be rewritten there.
def schedule_patch(points):
    return ("""OBSERVATIONS = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                2250, 2500, 2750, 3000)""",
            f"OBSERVATIONS = {points!r}")


#: `observe(..., light=True)` must not write the 8.7 MB snapshot.
OBSERVER = [
    ("""def observe(model, embeddings: torch.Tensor, epoch: int, out_dir: Path,
            curvature: float = 1.0) -> dict:""",
     """def observe(model, embeddings: torch.Tensor, epoch: int, out_dir: Path,
            curvature: float = 1.0, light: bool = False) -> dict:"""),
    ("""    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / ("epoch_%04d.npz" % epoch), **record)""",
     """    out_dir.mkdir(parents=True, exist_ok=True)
    if light:
        # The dense points of the fine window need the summary only; a 9 MB snapshot
        # per epoch would be 300 MB per arm for numbers nothing reads.
        (out_dir / ("epoch_%04d.json" % epoch)).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        return summary
    np.savez_compressed(out_dir / ("epoch_%04d.npz" % epoch), **record)"""),
]

DEFAULT = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000)
FINE = tuple(range(0, 105, 5)) + (150, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000)
SCREEN = (0, 50, 100, 200, 300)
FULL_FINE = (0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000)
FULL_SCREEN = SCREEN

#: Which levels get the baseline's own Sinkhorn during training. F0 = {2}; B3 = {0, 2};
#: C1 = {0, 1, 2}; C2 = {1, 2} -- the last level has always been balanced.
def balanced_patch(levels):
    condition = "index in %r" % (tuple(levels),)
    return ("""                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,""",
            f"""                # Round 3: the balanced levels are {sorted(levels)} (index 2 is the
                # last level, which the baseline always balanced). The rule itself is
                # the baseline's Sinkhorn; only its coverage changes.
                sk_epsilon=sk_epsilon if ({condition}) else -1.0,
                sk_iters=sk_iters if ({condition}) else -1,""")


ARMS = {
    # A_fine ran with the observer's stock schedule (the fine points live in
    # A_window); its spec says so, so the manifest describes the tree that produced
    # the data rather than the tree we meant to build.
    "A_fine": dict(
        epochs=3000, observations=DEFAULT, full=DEFAULT, extra=""),
    "A_window": dict(
        epochs=300, observations=FINE, full=SCREEN, extra=""),
    "B0_replay": dict(
        epochs=300, observations=SCREEN, full=FULL_SCREEN, extra=""),
    "B1_enc_freeze": dict(
        epochs=300, observations=SCREEN, full=FULL_SCREEN,
        extra="""        _freeze = epoch < FREEZE_EPOCHS
        for parameter in raw_module.encoder.parameters():
            parameter.requires_grad = not _freeze
"""),
    "B2_code_freeze": dict(
        epochs=300, observations=SCREEN, full=FULL_SCREEN,
        extra="""        _freeze = epoch < FREEZE_EPOCHS
        for layer in raw_module.rq.levels:
            layer.embed.weight.requires_grad = not _freeze
"""),
    "B3_l1_balanced": dict(
        epochs=300, observations=SCREEN, full=FULL_SCREEN, extra="",
        mixed_rq=("""                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,""",
                  """                # B3: level 1 is balanced too.  The rule itself is the baseline's
                # Sinkhorn; only which level it covers changes.
                sk_epsilon=sk_epsilon if (last or index == 0) else -1.0,
                sk_iters=sk_iters if (last or index == 0) else -1,""")),
    "B3_long": dict(
        epochs=3000, observations=FULL_FINE, full=FULL_FINE, extra="",
        mixed_rq=("""                sk_epsilon=sk_epsilon if last else -1.0,
                sk_iters=sk_iters if last else -1,""",
                  """                # B3: level 1 is balanced too.  The rule itself is the baseline's
                # Sinkhorn; only which level it covers changes.
                sk_epsilon=sk_epsilon if (last or index == 0) else -1.0,
                sk_iters=sk_iters if (last or index == 0) else -1,""")),
    "B4_revival": dict(
        epochs=300, observations=SCREEN, full=FULL_SCREEN,
        extra="""        if epoch >= REVIVE_EVERY and epoch % REVIVE_EVERY == 0:
            _revived = revive_l1(raw_module, all_embeddings[:8192].to(device), CURVATURE)
            _record(rank, "qda_revive", epoch=epoch, revived=_revived)
"""),
}

ARMS.update({
    # Round 3: which loss term moves the encoder. 60 epochs covers the whole
    # collapse window (complete by epoch 10) with an order of magnitude of margin.
    # 3000 epochs on purpose: the LR schedule must be F0's, otherwise the first 60
    # epochs are a different optimisation. Attribution stops at QDA_GRADSRC_EPOCHS.
    "G1_gradsrc": dict(epochs=3000, observations=(0, 5, 10, 20, 40, 60),
                       full=(0, 60), extra="", gradsrc=True),
})
ARMS.update({
    "C1_all_balanced": dict(epochs=300, observations=SCREEN, full=FULL_SCREEN, extra="",
                            mixed_rq=balanced_patch((0, 1, 2))),
    "C2_l2_balanced": dict(epochs=300, observations=SCREEN, full=FULL_SCREEN, extra="",
                           mixed_rq=balanced_patch((1, 2))),
})
ARMS.update({
    f"{base}_s{seed}": dict(ARMS[base], seed=seed)
    # C1 is not a candidate after the 300-epoch screen (worse than B3 on both L3 and
    # the encoder), so it gets no seed replicates.
    for base in ("B0_replay", "B3_l1_balanced")
    for seed in (43, 44)
})

REVIVAL_SOURCE = '''

def revive_l1(model, sample: torch.Tensor, curvature: float) -> int:
    """Re-seat every L1 code that no item selected, at the worst-fit item's residual.

    Deterministic: the sample, the cost and the greedy assignment contain no draw, so
    the intervention adds no seed noise of its own. Only level 1 is touched.
    """
    from model.mixed_rq import expmap0, logmap0, project, split

    with torch.no_grad():
        encoded = model.encoder(sample)
        z_e, z_t = split(encoded)
        residual_h = expmap0(z_t, curvature) if curvature > 0 else z_t
        layer = model.rq.levels[0]
        alpha = model.rq.alpha_table.lookups(1, []).to(encoded.device)
        cost = layer.distances(z_e, residual_h, alpha)
        index = torch.argmin(cost, dim=-1)
        alive = torch.zeros(layer.n_embed, dtype=torch.bool)
        alive[index.unique()] = True
        dead = (~alive).nonzero().flatten()
        if dead.numel() == 0:
            return 0
        worst = torch.argsort(cost.gather(1, index.view(-1, 1)).squeeze(1), descending=True)
        spare = worst[torch.isin(worst, index.unique(), invert=True)]
        codes_e, codes_h_tangent = split(layer.get_code_embs())
        take = min(int(dead.numel()), int(spare.numel()))
        for slot in range(take):
            item = spare[slot]
            target_e = z_e[item]
            target_h = (logmap0(residual_h[item].unsqueeze(0), curvature).squeeze(0)
                        if curvature > 0 else residual_h[item])
            layer.embed.weight.data[dead[slot], :codes_e.shape[1]] = target_e
            layer.embed.weight.data[dead[slot], codes_e.shape[1]:] = target_h
        return take
'''


def rewrite(path: Path, pairs, label: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    applied = []
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{label}: pattern appears {count} times, expected 1:\n{old[:120]}")
        text = text.replace(old, new)
        applied.append(old.splitlines()[0][:100])
    path.write_text(text, encoding="utf-8")
    return applied


def digest_tree(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="build only these arms; the manifest then covers them alone")
    chosen = parser.parse_args().only
    source_digests = digest_tree(SOURCE)
    manifest = {"source": str(SOURCE), "source_digests": source_digests,
                "freeze_epochs": FREEZE_EPOCHS, "revive_every": REVIVE_EVERY, "arms": {}}
    for name, spec in ARMS.items():
        if chosen and name not in chosen:
            continue
        target = OUT / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(SOURCE, target,
                        ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"))
        patches = {}

        trainer = target / "train_rqvae.py"
        pairs = list(COMMON)
        # the common block carries the per-arm full-observation set
        pairs[0] = (pairs[0][0],
                    pairs[0][1].replace("{full!r}", repr(spec["full"]))
                               .replace("{gradsrc!r}", repr(bool(spec.get("gradsrc")))))
        pairs += [
            (f'ARM_NAME = "QDA_F0_seed42"', f'ARM_NAME = "{name}"'),
            ('EPOCHS              = 3000', f'EPOCHS              = {spec["epochs"]}'),
            ("""OBSERVATIONS   = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                  2250, 2500, 2750, 3000)""",
             f"""OBSERVATIONS   = {spec["observations"]!r}
#: Freeze / revival windows, derived from the measured collapse (201 -> 9 codes by
#: epoch 50): the intervention must cover that window, not a tuned length.
FREEZE_EPOCHS = {FREEZE_EPOCHS}
REVIVE_EVERY = {REVIVE_EVERY}"""),
        ]
        if spec.get("seed"):
            pairs.append(('SEED                = 42',
                          f'SEED                = {spec["seed"]}'))
        if spec.get("gradsrc"):
            pairs.append(("""            batch = batch.to(device, non_blocking=True)""",
                          """            batch = batch.to(device, non_blocking=True)
            if QDA_GRADSRC and epoch <= QDA_GRADSRC_EPOCHS and batch_index == 0:
                # Attribution runs on the epoch's first batch only, on the model the
                # training step is about to update; `measure` restores every parameter
                # and clears the gradients it creates.
                _record(rank, "qda_gradsrc", epoch=epoch,
                        **measure(raw_module, batch, BETA,
                                  optimizer.param_groups[0]["lr"], CURVATURE))"""))
        if spec["extra"]:
            pairs.append(("""        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)""",
                          spec["extra"] + """        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)"""))
        patches["train_rqvae.py"] = rewrite(trainer, pairs, name)

        if spec["extra"].startswith("        if epoch >= REVIVE_EVERY"):
            # Insert before the entry point: appending would define revive_l1 only
            # after main() has already been called.
            body = trainer.read_text(encoding="utf-8")
            anchor = 'if __name__ == "__main__":'
            if body.count(anchor) != 1:
                raise SystemExit(f"{name}: entry-point anchor not unique")
            trainer.write_text(body.replace(anchor, REVIVAL_SOURCE.strip() + "\n\n\n" + anchor),
                               encoding="utf-8")
            patches["train_rqvae.py"].append("inserted revive_l1() before the entry point")

        patches["qda_observer.py"] = rewrite(
            target / "qda_observer.py",
            [schedule_patch(spec["observations"])] + OBSERVER, name)
        shutil.copy2(HERE / "telemetry_src.py", target / "qda_telemetry.py")
        shutil.copy2(HERE.parent / "round3" / "gradsrc_src.py", target / "qda_gradsrc.py")
        patches["qda_gradsrc.py"] = ["installed from round3/gradsrc_src.py"]
        patches["qda_telemetry.py"] = ["installed from round2/telemetry_src.py"]

        if "mixed_rq" in spec:
            patches["model/mixed_rq.py"] = rewrite(target / "model" / "mixed_rq.py",
                                                   [spec["mixed_rq"]], name)

        manifest["arms"][name] = {
            "epochs": spec["epochs"],
            "observations": list(spec["observations"]),
            "full_observations": list(spec["full"]),
            "patches": patches,
            "digests": digest_tree(target),
        }
        print(f"built {name}: {spec['epochs']} epochs, {len(spec['observations'])} observations")

    out_manifest = HERE / ("arm_manifest.json" if not chosen else "arm_manifest_partial.json")
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote arm_manifest.json")


if __name__ == "__main__":
    main()
