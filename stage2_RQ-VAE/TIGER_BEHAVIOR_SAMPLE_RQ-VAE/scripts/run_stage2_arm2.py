"""Hardcoded entry point for one of the new crowding arms.

The trainer integration is intentionally owned by the caller. Once its
``run_crowding_arm2`` hook is wired, changing ACTIVE_ARM and running this file
selects exactly one arm and its dedicated immutable output directory.
"""
from pathlib import Path

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
ACTIVE_ARM = "poslocal_w1"
STRENGTHS = {"w1": 0.5, "w2": 2.0, "w3": 8.0}
MECHANISMS = ("poslocal", "multipos", "couserw", "hncontrast", "prefixbal", "margincomp")
ARM_SETTINGS = {
    f"{name}_{level}": (name, weight)
    for name in MECHANISMS + ("random", "scale")
    for level, weight in STRENGTHS.items()
}
ARM_SETTINGS["behonly"] = ("behonly", 0.0)


def main():
    if ACTIVE_ARM not in ARM_SETTINGS:
        raise ValueError(f"unknown arm {ACTIVE_ARM!r}")
    output_dir = ROOT / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE" / f"crowd2_{ACTIVE_ARM}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing arm output: {output_dir}")
    mechanism, weight = ARM_SETTINGS[ACTIVE_ARM]
    if mechanism == "behonly":
        raise ValueError("behonly is the existing Exp1a run; do not retrain it")

    import train_rqvae

    hook = getattr(train_rqvae, "run_crowding_arm2", None)
    if hook is None:
        raise RuntimeError(
            "trainer integration is not wired: add run_crowding_arm2(mechanism, strength, output_dir) "
            "using crowding_losses2.mechanism_loss2 and crowding_losses2.additive_total"
        )
    hook(mechanism=mechanism, strength=weight, output_dir=output_dir)


if __name__ == "__main__":
    main()
