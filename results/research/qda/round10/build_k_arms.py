"""Build the arm where both CNCQ layers are active in the actual trainer.

    capacity   every assignment gives each codeword at least one item        (round 8)
    geometry   c_k = a_k + offset_k, ||offset_k|| <= rho, 2*rho < delta      (this round)

`delta` comes from the *data*: the prototypes are the KMeans centres of the level's initial
residual, which is where the frozen baseline puts its codebook anyway. `rho` is a quarter of
the smallest inter-centre distance, so `2*rho = delta/2 < delta` and the bound holds with
margin, while the allowed movement stays on the scale of the data rather than an arbitrary
constant.

The layer is only as good as the codebook it starts from, so the offsets start at zero and
the prototypes never move: what learns is the offset, inside the ball.
"""
from __future__ import annotations

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
ARM = "K1_floor_bounded"
FLOOR = HERE.parent / "round8" / "floor_src.py"

CONSTANTS = '''
#: Round 10: the shared capacity floor and the bounded geometry.
ASSIGNMENT_FLOOR = 1
FORCE_CONSTRAINED_ALL = True
#: rho is derived at initialisation from the data's own prototype spacing.
GEOMETRY_FRACTION = 0.25
'''

IMPORT = (
    """from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance""",
    """from .poincare_contract import expmap0, logmap0, mobius_add, project, squared_distance
from qda_floor import constrained_assignment
from qda_geometry import codebook as bounded_codebook""",
)

CONSTANT_ANCHOR = """CURVATURE = 1.0"""
CONSTANT_BLOCK = CONSTANTS + """
CURVATURE = 1.0"""

LEVEL_INIT = (
    """        # One joint table: columns 0..15 are E, columns 16..31 are tangent H.
        self.embed = nn.Embedding(self.n_embed, TOTAL_DIM)""",
    """        # One joint table: columns 0..15 are E, columns 16..31 are tangent H.
        # The table holds the *offset* from a fixed prototype; the codewords the quantiser
        # uses are `prototype + clip(offset, rho)` (see qda_geometry).
        self.embed = nn.Embedding(self.n_embed, TOTAL_DIM)
        self.register_buffer("prototypes", torch.zeros(self.n_embed, TOTAL_DIM),
                             persistent=False)
        self.rho = 0.0
        self.separation = {"delta": 0.0, "two_rho": 0.0, "margin": 0.0, "ok": False}""",
)

GET_CODES = (
    """    def get_code_embs(self) -> torch.Tensor:
        return self.embed.weight""",
    """    def get_code_embs(self) -> torch.Tensor:
        # The only codewords the quantiser may use: fixed prototypes plus a bounded offset.
        if self.rho <= 0:
            return self.prototypes
        return bounded_codebook(self.prototypes, self.embed.weight, self.rho)""",
)

INIT_CODEBOOK_HEAD = (
    """        from .utils import kmeans
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            codes_e = kmeans(residual_e.detach().float(), self.n_embed)""",
    """        from .utils import kmeans
        from qda_geometry import guarantee, separation
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not distributed or torch.distributed.get_rank() == 0:
            codes_e = kmeans(residual_e.detach().float(), self.n_embed)""",
)

INIT_CODEBOOK_TAIL = (
    """            self.embed.weight[:, EUCLIDEAN_DIM:].copy_(codes_h)
        if distributed:
            torch.distributed.broadcast(self.embed.weight, src=0)""",
    """            self.prototypes[:, :EUCLIDEAN_DIM].copy_(codes_e)
            self.prototypes[:, EUCLIDEAN_DIM:].copy_(codes_h)
            # rho: a quarter of the data's own prototype spacing, so 2*rho < delta holds
            # with a wide margin and the ball is on the scale of the representation.
            self.rho = float(GEOMETRY_FRACTION) * separation(self.prototypes)
            self.embed.weight.data.zero_()          # offsets start at the prototypes
            self.separation = guarantee(self.prototypes, self.rho)
        if distributed:
            torch.distributed.broadcast(self.prototypes, src=0)
            torch.distributed.broadcast(self.embed.weight, src=0)""",
)

INDICES = (
    """        return torch.argmin(distances, dim=-1)""",
    """        index, _ = constrained_assignment(distances, ASSIGNMENT_FLOOR)
        return index.to(distances.device)""",
)

EXPORT = (
    """            index = torch.argmin(cost, dim=-1)""",
    """            index, _ = constrained_assignment(cost, ASSIGNMENT_FLOOR)""",
)

USE_SK = (
    """        self.use_sk = sk_epsilon > 0 and sk_iters > 0""",
    """        self.use_sk = (sk_epsilon > 0 and sk_iters > 0) and not FORCE_CONSTRAINED_ALL""",
)

# The observer records the measured mixed minimum distance each level, every observation.
OBSERVE_GEOMETRY = (
    """            record[f"usage_{level}"] = _codebook_stats(index, layer.n_embed)""",
    """            record[f"usage_{level}"] = _codebook_stats(index, layer.n_embed)
            measured.append(float(mixed_min_distance(
                layer.get_code_embs().detach().cpu().numpy())))""",
)

OBSERVE_SUMMARY = (
    """            "mean_margin": [float(np.mean(record[f"margin_{level}"].astype(np.float32)))
                            for level in range(1, len(model.rq.levels) + 1)],""",
    """            "mean_margin": [float(np.mean(record[f"margin_{level}"].astype(np.float32)))
                            for level in range(1, len(model.rq.levels) + 1)],
            "min_pair_distance": measured,
            "rho": [float(layer.rho) for layer in model.rq.levels],
            "separation": [dict(layer.separation) for layer in model.rq.levels],""",
)

OBSERVE_IMPORT = (
    """    from model.mixed_rq import (expmap0, logmap0, mobius_add, project,
                                quantized_readout, split)""",
    """    from model.mixed_rq import (expmap0, logmap0, mobius_add, project,
                                quantized_readout, split)
    from qda_geometry import mixed_min_distance

    measured: list[float] = []""",
)


def main() -> None:
    manifest = {"source": str(build.SOURCE), "arms": {}}
    name = ARM
    target = OUT / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(build.SOURCE, target,
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"))

    pairs = [p for p in build.COMMON]
    pairs[0] = (pairs[0][0], pairs[0][1].replace("{full!r}", repr(SCREEN))
                .replace("{gradsrc!r}", "False"))
    pairs[1] = (pairs[1][0], pairs[1][1].replace('"round2"', '"round10"'))
    pairs += [
        ('ARM_NAME = "QDA_F0_seed42"', f'ARM_NAME = "{name}"'),
        ('EPOCHS              = 3000', 'EPOCHS              = 300'),
        ("""OBSERVATIONS   = (0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                  2250, 2500, 2750, 3000)""",
         f"OBSERVATIONS   = {SCREEN!r}"),
        ("QDA_GRADSRC_EPOCHS = 60", "QDA_GRADSRC_EPOCHS = 60"),
    ]
    patches = {"train_rqvae.py": build.rewrite(target / "train_rqvae.py", pairs, name)}
    patches["qda_observer.py"] = build.rewrite(
        target / "qda_observer.py",
        [build.schedule_patch(SCREEN)] + build.OBSERVER + [OBSERVE_IMPORT, OBSERVE_GEOMETRY,
                                                           OBSERVE_SUMMARY], name)
    mixed = target / "model" / "mixed_rq.py"
    patches["model/mixed_rq.py"] = build.rewrite(
        mixed, [(CONSTANT_ANCHOR, CONSTANT_BLOCK), IMPORT, LEVEL_INIT, GET_CODES,
                INIT_CODEBOOK_HEAD, INIT_CODEBOOK_TAIL, INDICES, EXPORT, USE_SK], name)

    shutil.copy2(ROUND2 / "telemetry_src.py", target / "qda_telemetry.py")
    shutil.copy2(HERE.parent / "round3" / "gradsrc_src.py", target / "qda_gradsrc.py")
    shutil.copy2(FLOOR, target / "qda_floor.py")
    shutil.copy2(HERE / "geometry_src.py", target / "qda_geometry.py")
    patches["qda_floor.py"] = ["installed from round8/floor_src.py"]
    patches["qda_geometry.py"] = ["installed from round10/geometry_src.py"]

    manifest["arms"][name] = {"epochs": 300, "observations": list(SCREEN),
                              "patches": patches, "digests": build.digest_tree(target)}
    (HERE / "arm_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                            encoding="utf-8")
    print(f"built {name}")


if __name__ == "__main__":
    main()
