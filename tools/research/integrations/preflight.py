"""Read-only source/layout checks; training requires new MCP verification_id."""
from pathlib import Path
from .. import settings


def audit_surviving_mainline() -> dict:
    checked = {
        "train_split": settings.TRAIN,
        "stage2_baseline": settings.STAGE2_BASELINE / "train_rqvae.py",
        "stage2_behavior": settings.STAGE2_BEHAVIOR / "train_rqvae.py",
        "stage3_entry": settings.STAGE3 / "train_HG-Rec.py",
        "cgv": settings.CGV,
        "mcp": settings.MCP,
    }
    return {key: {"path": str(path), "exists": Path(path).is_file()}
            for key, path in checked.items()}


def request_training(*args, **kwargs):
    raise PermissionError("No direct training in research tools; use MCP verification_check and stage2_start(source_dir, verification_id) with independent evidence, real smoke, explicit authorization, and unchanged Stage3 recipe.")
