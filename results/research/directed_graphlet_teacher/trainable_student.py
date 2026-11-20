"""Issue #223: the structure loss wired into the real RQ-VAE graph.

`gate.py` trained a Student head on a *frozen* SID: the head learned, the encoder
and the codebooks never moved.  This module closes that loop:

    items (768) -> model.encoder -> z -> model.rq (real three-level quantiser)
        -> hard tokens -> q_st (hard value forward, soft onehot backward)
        -> StudentQuantisedHead -> L_structure -> backward
        -> encoder parameters AND the three raw codebook tables

The quantiser's own `forward` implements straight-through on the *encoder* only
(`x + (x_q - x).detach()`), so a structure loss taken on that tensor reaches the
encoder and never the codebooks.  The codebook path therefore uses the arm's
`q_st`: the forward value is exactly the sum of the picked code words (verified
below against `hard_cumulative` and against `get_indices`), while the backward
pass carries gradient into each level's raw `embed.weight`.

Every failure mode that would make this look wired without being wired is a
detector here: a detached quantiser, a frozen codebook, an optimiser that does
not hold the codebook parameters, or a Student bundle that carries structure.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARM_DIR = REPO / "stage2_RQ-VAE" / "TIGER_drq_D1"

BANNED_KEYS = ("adjacency", "adj", "edges", "edge", "graph", "teacher_logits",
               "teacher_hidden", "node_hidden", "subtype", "orbit", "label",
               "canonical")


class LearningPathError(RuntimeError):
    """A required parameter group is not on the gradient path or not optimised."""


def _arm():
    if str(ARM_DIR) not in sys.path:
        sys.path.insert(0, str(ARM_DIR))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import drq  # noqa: F401  (the arm's quantiser helpers)
    import losses  # noqa: F401
    from distances import build_distance
    from model import RQVAE
    return {"RQVAE": RQVAE, "build_distance": build_distance, "drq": drq, "losses": losses}


def _tokenizer_config(narrow: int, levels: int, codes: int, dim: int, dropout: float = 0.0):
    from types import SimpleNamespace
    return SimpleNamespace(hidden_sizes=[narrow], codebook_num=levels,
                           codebook_size=[codes] * levels, codebook_dim=dim,
                           beta=0.25, vq_type="vq", ema_decay=0.99,
                           sk_epsilon=0.003, sk_iters=50, dropout=dropout,
                           fix_code_embs=False)


def hard_cumulative(tokens, codebooks):
    """The arm's own hard-value helper, re-exported so the equivalence is checkable."""
    return _arm()["drq"].hard_cumulative(tokens, codebooks)


def build_small_model(in_dim: int, narrow: int = 16, levels: int = 3, codes: int = 16,
                      dim: int = 8) -> torch.nn.Module:
    """The arm's own RQVAE at test widths (the smoke uses the frozen 768->...->32)."""
    arm = _arm()
    return arm["RQVAE"](_tokenizer_config(narrow, levels, codes, dim), in_dim=in_dim,
                        distance=arm["build_distance"]())


def build_head(dim: int, n_subtypes: int, n_orbits: int, hidden: int = 32):
    from student_head import StudentQuantisedHead
    return StudentQuantisedHead(dim, hidden, n_subtypes, n_orbits)


def build_small_teacher(dim: int, hidden: int = 16, levels: int = 2,
                        n_subtypes: int = 5, n_orbits: int = 7):
    from teacher_model import build_teacher
    return build_teacher({"dim": dim, "hidden": hidden, "layers": levels,
                          "n_subtypes": n_subtypes, "n_orbits": n_orbits})


def random_adjacency(instances: int, nodes: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    adjacency = (torch.rand(instances, nodes, nodes, generator=generator) < 0.4).float()
    adjacency[:, torch.arange(nodes), torch.arange(nodes)] = 0.0
    return adjacency


class QuantisedPath:
    """What the Student is allowed to read: hard codes and the codes' vectors."""

    def __init__(self, q: torch.Tensor, tokens: torch.Tensor, residuals: list,
                 probabilities: list, codebooks: list):
        self.q = q
        self.tokens = tokens
        self.residuals = residuals
        self.probabilities = probabilities
        self.codebooks = codebooks


def quantised_student_input(model, items: torch.Tensor, *, instances: int, nodes: int,
                            detach_quantised: bool = False) -> QuantisedPath:
    """Run the REAL encoder and the REAL three-level quantiser; return hard codes.

    The forward value is the hard code-word sum; the backward path is `q_st`
    (soft one-hot times the raw codebook table), which is the only way the
    codebooks can receive gradient from a loss on `q`.
    """
    arm = _arm()
    z = model.encoder(items)
    _, _, _, tokens = model.rq(z)                      # real assignment, real hard indices
    residuals = model.rq.last_residuals
    codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]
    probabilities = []
    for level, layer in enumerate(model.rq.vq_layers):
        temperature = arm["drq"].measure_temperature(layer, residuals[level])
        probabilities.append(arm["drq"].level_softmax(layer, residuals[level], temperature))
    flat = arm["drq"].q_st(probabilities, tokens, codebooks, depth=len(codebooks))
    if detach_quantised:
        flat = flat.detach()
    q = flat.view(instances, nodes, -1)
    token_view = tokens.view(instances, nodes, tokens.shape[-1])
    return QuantisedPath(q=q, tokens=token_view, residuals=residuals,
                         probabilities=probabilities, codebooks=codebooks)


def assert_no_structure(bundle: dict) -> None:
    for key in bundle:
        lowered = str(key).lower()
        for banned in BANNED_KEYS:
            if banned in lowered:
                raise LearningPathError("student bundle carries %r (matches %r)"
                                        % (key, banned))


def structure_loss(model, head, items: torch.Tensor, *, instances: int, nodes: int,
                   targets: dict | None = None, teacher=None, adjacency: torch.Tensor | None = None,
                   temperature: float = 2.0, alpha: float = 1.0,
                   detach_quantised: bool = False):
    """L_structure from the current, trainable RQ-VAE; targets are stop-gradient."""
    import torch.nn.functional as F

    path = quantised_student_input(model, items, instances=instances, nodes=nodes,
                                   detach_quantised=detach_quantised)
    bundle = {"q": path.q}
    assert_no_structure(bundle)
    type_logits, orbit_logits = head(path.q)
    if teacher is not None:
        if adjacency is None:
            raise LearningPathError("a teacher needs the instance adjacency on the supervision side")
        with torch.no_grad():
            teacher_type, teacher_orbit, _ = teacher(items.view(instances, nodes, -1), adjacency)
        from kdistill import joint_kd
        loss = joint_kd(type_logits, orbit_logits, teacher_type.detach(), teacher_orbit.detach(),
                        temperature=temperature, alpha=alpha)["total"]
        supervision = "teacher_kd"
    else:
        if targets is None:
            raise LearningPathError("hard-label supervision needs type and orbit targets")
        loss = F.cross_entropy(type_logits, targets["type"]) + \
            F.cross_entropy(orbit_logits.reshape(-1, orbit_logits.shape[-1]),
                            targets["orbit"].reshape(-1))
        supervision = "hard_ce"
    info = {"student_bundle": bundle, "tokens": path.tokens, "supervision": supervision,
            "type_logits": type_logits, "orbit_logits": orbit_logits}
    return loss, info


def codebook_parameters(model) -> dict:
    """Each level's RAW table (`embed.weight`), not `get_code_embs()`'s output."""
    parameters = {}
    for level, layer in enumerate(model.rq.vq_layers):
        raw = [(name, value) for name, value in layer.named_parameters()
               if name.startswith("embed") and value.ndim == 2]
        if not raw:
            raise LearningPathError("level %d has no raw codebook parameter" % level)
        parameters["level%d" % level] = raw[0][1]
    return parameters


def _group_norm(loss, parameters):
    live = [parameter for parameter in parameters if parameter is not None]
    if not live or not any(parameter.requires_grad for parameter in live):
        return 0.0
    grads = torch.autograd.grad(loss, [p for p in live if p.requires_grad],
                                retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum())
    return float(np.sqrt(total))


def gradient_report(loss, model, head, teacher=None) -> dict:
    """Grad norm per parameter group, read from the raw parameters."""
    report = {"encoder": _group_norm(loss, list(model.encoder.parameters())),
              "codebooks": [_group_norm(loss, [parameter])
                            for parameter in codebook_parameters(model).values()],
              "head": _group_norm(loss, list(head.parameters()))}
    if teacher is not None:
        frozen = all(not parameter.requires_grad for parameter in teacher.parameters())
        report["teacher"] = 0.0 if frozen else _group_norm(loss, list(teacher.parameters()))
    else:
        report["teacher"] = 0.0
    return report


def assert_learning_path(report: dict) -> None:
    if report["encoder"] <= 0.0:
        raise LearningPathError("the structure loss gives the encoder no gradient "
                                "(detached quantiser or detached q)")
    if report["head"] <= 0.0:
        raise LearningPathError("the structure loss gives the student head no gradient")
    for level, norm in enumerate(report["codebooks"]):
        if norm <= 0.0:
            raise LearningPathError("codebook level %d receives no gradient: frozen or "
                                    "bypassed table" % level)
    if report.get("teacher", 0.0) > 0.0:
        raise LearningPathError("the teacher is receiving gradient; its targets must be detached")


def assert_optimizer_covers(optimizer, model, head) -> None:
    known = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    missing = []
    if id(next(model.encoder.parameters())) not in known:
        missing.append("encoder")
    for name, parameter in codebook_parameters(model).items():
        if id(parameter) not in known:
            missing.append(name)
    if id(next(head.parameters())) not in known:
        missing.append("head")
    if missing:
        raise LearningPathError("the optimiser does not hold these parameters: %s"
                                % ", ".join(missing))


def safe_aux_weight(*, encoder_reference: float, encoder_aux: float,
                    codebook_reference: list, codebook_aux: list,
                    share: float = 0.10) -> float:
    """The largest weight that keeps EVERY group inside `share` of its main gradient.

    The arm's own `calibrate_weight` matches the encoder norm and says nothing about
    the codebooks; on the smoke that left the auxiliary codebook gradient at 1.3x the
    main one and the L1 table collapsed within ten steps.  Capping each group and
    taking the minimum makes the pre-registered share a bound on the whole path, not
    on one third of it.
    """
    bounds = []
    if encoder_aux > 0 and encoder_reference > 0:
        bounds.append(share * encoder_reference / encoder_aux)
    for reference, auxiliary in zip(codebook_reference, codebook_aux):
        if auxiliary > 0 and reference > 0:
            bounds.append(share * reference / auxiliary)
    if not bounds:
        return 0.0
    return float(min(bounds))


def parameter_digest(module) -> dict:
    digest = {}
    for name, parameter in module.named_parameters():
        digest[name] = hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest()
    return digest


def _snapshot(module) -> dict:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def parameter_divergence(before: dict, after: dict) -> dict:
    encoder, codebooks = 0.0, 0.0
    for name in before:
        if name not in after:
            continue
        difference = float((before[name] - after[name]).abs().max())
        if name.startswith("encoder"):
            encoder = max(encoder, difference)
        elif "embed" in name:
            codebooks = max(codebooks, difference)
    return {"encoder": encoder, "codebooks": codebooks}


def prove_update(*, seed: int, items: torch.Tensor, instances: int, nodes: int,
                 targets: dict, lambda_value: float, steps: int = 1) -> dict:
    """One real optimiser step on L_main + lambda * L_structure, lambda=0 included."""
    arm = _arm()
    torch.manual_seed(seed)
    model = build_small_model(in_dim=items.shape[-1])
    head = build_head(dim=model.rq.codebook_dim, n_subtypes=int(targets["type"].max()) + 1,
                      n_orbits=int(targets["orbit"].max()) + 1)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    assert_optimizer_covers(optimizer, model, head)

    before = _snapshot(model)
    structure_norms = {"encoder": 0.0, "codebooks": [0.0] * len(model.rq.vq_layers)}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstructed, parts, _, _ = model(items)
        main = arm["losses"].reconstruction_loss(reconstructed, items) + \
            arm["losses"].quantization_total(parts, 0.25)
        aux, _ = structure_loss(model, head, items, instances=instances, nodes=nodes,
                                targets=targets)
        if lambda_value > 0.0:
            report = gradient_report(aux, model, head)
            structure_norms = {"encoder": report["encoder"], "codebooks": report["codebooks"]}
            total = main + lambda_value * aux
        else:
            total = main
        total.backward()
        optimizer.step()
    after = _snapshot(model)
    changed = {
        "encoder": any(not torch.equal(before[name], after[name])
                       for name in before if name.startswith("encoder")),
        "codebooks": all(any(not torch.equal(before[name], after[name])
                             for name in before if "vq_layers.%d.embed" % level in name)
                         for level in range(len(model.rq.vq_layers))),
    }
    return {"structure_grad_norms": structure_norms, "changed": changed,
            "before": before, "after": after, "lambda": lambda_value}


def assert_update_proof(proof: dict) -> None:
    if not proof["changed"]["encoder"]:
        raise LearningPathError("the optimiser step did not change the encoder")
    if not proof["changed"]["codebooks"]:
        raise LearningPathError("the optimiser step did not change every codebook table")
