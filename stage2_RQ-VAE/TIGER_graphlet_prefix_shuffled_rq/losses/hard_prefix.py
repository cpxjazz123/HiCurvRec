#!/usr/bin/env python3
"""Issue #230 §3: hard-prefix graphlet supervision, differentiable through a straight-through
estimator, applied where the measurement says the bias is built.

Why this shape, in the order the measurements arrived:

1. **The bias is level 1.** On the frozen test bank the baseline's three-level Δ decomposes as
   L1 +0.0376 / +0.0074 / +0.0049 for chain (87.5% level 1), -0.0565 / -0.0140 / -0.0057 for
   fork and -0.0481 / -0.0061 / -0.0031 for merge. Levels 2 and 3 keep the sign and shrink the
   magnitude, so a term that only reaches L2 cannot move the ordering.
2. **Capacity is not the lever.** Widening level 1 from 256 to 1024 codes leaves every sign in
   place and pushes the tie rate from 86% to 96%. The term must change what the codes *are*,
   not how many there are.
3. **Inside one structure nothing conflicts** (0 pairs pulled both ways in 20,000 instances per
   structure), **across structures** the contested pairs are chain∩fork = 116 and
   chain∩merge = 103 — the same pair is near-evidence for chain and far-evidence for fork. A
   single shared prefix geometry therefore has to encode the *role* of the pair, which is what
   the per-structure role tables below supply.

The forward pass is the real thing: the codes are ``argmax`` over the level-1 logits, and the
margin is computed from the *exported* `S()` (shared leading run of the first three codes). The
backward pass is the straight-through estimator: the forward value is detached and the gradient
of a soft-sharing surrogate is added, so the reported loss moves exactly as much as the codes do.

Only level 1 is supervised. The assignment rule is untouched: the term adds to the level-1 logits
*before* the existing Sinkhorn / floor assignment sees them, so the balancing allocation stays the
baseline's (§19h) and the mechanism lives in the cost matrix, which is what the issue asks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + __file__)


#: The frozen banks the issue requires (C7–C11). Hashes are checked before the term is used.
BANK_DIR = _repo_root() / "results" / "research" / "issue230" / "banks"
#: Position pairs, over the shared role tuple (A, B, C) with A at slot 0 for every structure.
NEAR = {"chain": ((0, 1), (1, 2)), "fork": ((1, 2),), "merge": ((1, 2),)}
FAR = {"chain": ((0, 2),), "fork": ((0, 1), (0, 2)), "merge": ((0, 1), (0, 2))}
KINDS = ("chain", "fork", "merge")
#: Set only in the equal-budget control tree; see ``RoleIndex.shuffled``.
ARM_CONTROL = True
MARGIN = 1.0
#: Gradient budget: the term may not exceed this fraction of the reconstruction gradient norm.
BUDGET = 0.10


def hard_codes(logits: torch.Tensor) -> torch.Tensor:
    """The assignment the export sees: ``argmin`` over the level-1 cost."""
    return torch.argmin(logits, dim=-1)


def shared_prefix(codes: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
                  depth: int = 3) -> torch.Tensor:
    """`S()`: how many of the first ``depth`` codes two items agree on, as a float tensor.

    ``codes`` is either a per-item assignment ``(items,)`` or a token matrix ``(items, levels)``;
    the two are handled by the same shape test so a caller can pass the export's tokens.
    """
    if codes.dim() == 1:
        return (codes[left] == codes[right]).float()
    equal = codes[left] == codes[right]
    run = torch.ones(equal.shape[0], dtype=torch.bool, device=equal.device)
    return torch.stack([(run := run & equal[:, step]).float() for step in range(depth)]).sum(0)


def soft_prefix_share(probabilities: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
                      depth: int = 1) -> torch.Tensor:
    """The surrogate the STE differentiates: the expected number of shared codes."""
    share = (probabilities[left] * probabilities[right]).sum(-1)
    total = share
    for _ in range(depth - 1):
        share = share * ((probabilities[left] * probabilities[right]).sum(-1))
        total = total + share
    return total


def _temperature(cost: torch.Tensor) -> float:
    """The surrogate's scale, derived from the matrix instead of tuned.

    Measured on a real level-1 cost matrix (1024 codes): the median nearest/second-nearest gap
    is 4.2e-3 while the costs themselves are ~6.5e-2. ``softmax(-cost)`` at that scale has
    entropy 6.93 ~= ln(1024), i.e. it is uniform and its gradient is 2.7e-11 -- inert. A
    temperature of half the median gap concentrates it on ~10 candidate codes per item
    (entropy 1.86) without approaching the hard assignment, which is what a straight-through
    surrogate needs. Half the median gap is a property of the data, not a fitted constant.
    """
    with torch.no_grad():
        two = torch.topk(cost.detach(), 2, dim=1, largest=False).values
        gap = (two[:, 1] - two[:, 0]).clamp(min=0.0).median()
        if float(gap) <= 0.0:
            return max(float(cost.detach().mean()) * 0.05, 1e-6)
        return float(gap) / 2.0


def prefix_margin(logits: torch.Tensor, roles: torch.Tensor, kind: str, *, margin: float = MARGIN,
                  depth: int = 3):
    """The issue's Δ turned into a hinge, hard forward and surrogate backward.

    ``logits`` is the level-1 cost matrix ``(items, codes)``. ``roles`` is ``(n, 3)`` of row
    indices into ``logits`` in the contract's role order. Returns the batch mean and the
    per-instance near / far values for logging.
    """
    codes = hard_codes(logits)
    a, b, c = roles[:, 0], roles[:, 1], roles[:, 2]
    soft = torch.softmax(-logits.float() / _temperature(logits), dim=-1)
    if codes.dim() == 1:
        # one supervised level: S(A,B) is 1 when the two items share the level-1 codeword.
        s_ab = shared_prefix(codes, a, b)
        s_bc = shared_prefix(codes, b, c)
        s_ac = shared_prefix(codes, a, c)
        p_ab = soft_prefix_share(soft, a, b)
        p_bc = soft_prefix_share(soft, b, c)
        p_ac = soft_prefix_share(soft, a, c)
    else:
        s_ab = shared_prefix(codes, a, b, depth)
        s_bc = shared_prefix(codes, b, c, depth)
        s_ac = shared_prefix(codes, a, c, depth)
        p_ab = soft_prefix_share(soft, a, b, depth)
        p_bc = soft_prefix_share(soft, b, c, depth)
        p_ac = soft_prefix_share(soft, a, c, depth)
    if kind == "chain":
        near, far = (s_ab + s_bc) / 2.0, s_ac
        soft_near, soft_far = (p_ab + p_bc) / 2.0, p_ac
    else:
        near, far = s_bc, (s_ab + s_ac) / 2.0
        soft_near, soft_far = p_bc, (p_ab + p_ac) / 2.0
    hard_gap = near - far
    soft_gap = soft_near - soft_far
    # The gate is decided by the hard gap (the true Δ), the slope comes from the surrogate.
    # A term clamped on the detached hard value alone is inert whenever the hard gap is 0 but
    # the soft gap is not -- measured: gradient 1.5e-9 on an all-zero cost matrix.
    #
    # The surrogate enters as `margin - soft_gap`: the loss is *minimised*, so its gradient is
    # `-d(soft_gap)/d(cost)`, and an SGD step on the cost matrix moves the wanted pair towards
    # sharing a codeword. `margin + soft_gap` was measured to invert every entry.
    gate = torch.clamp(margin - hard_gap.detach(), min=0.0)
    sloped = torch.clamp(margin - soft_gap, min=0.0)
    value = gate * sloped / sloped.detach().clamp(min=1e-12)
    info = {"near": near.detach(), "far": far.detach(), "hinge": gate}
    return value.mean(), info


def deciding_gap(cost: torch.Tensor) -> torch.Tensor:
    """Per item: the distance between its nearest and its second-nearest codeword.

    This is the only quantity that decides whether a perturbation to ``cost`` changes an
    ``argmin``: an entry must move by at least this much to flip the assignment. The budget is
    expressed against it instead of against ``cost.mean()``, which has no such meaning --
    measured, that older calibration produced a shift 0.0022x the median gap.
    """
    two = torch.topk(cost.detach(), 2, dim=1, largest=False).values
    return (two[:, 1] - two[:, 0]).clamp(min=1e-12)


def budgeted_shift(cost: torch.Tensor, ids: torch.Tensor, *, target_share: float,
                   alignment: "HardPrefixAlignment | None" = None) -> torch.Tensor:
    """The cost perturbation the graphlet term asks for, scaled so it can decide assignments.

    ``target_share`` is the fraction of items for which the deciding entry should be moved far
    enough to flip the assignment; it is derived from how many items of the batch the structure
    banks actually cover (``HardPrefixAlignment.coverage``), not chosen. The returned scale is
    set so that the ``target_share`` quantile of per-item maximum movement reaches that item's
    own deciding gap, and is never larger than one gap: beyond that the term would be rewriting
    the quantizer rather than expressing a preference.
    """
    alignment = alignment if alignment is not None else HardPrefixAlignment()
    # the term needs a differentiable input; the caller may hand us a plain cost matrix
    cost = cost if cost.requires_grad else cost.detach().requires_grad_(True)
    value, report = alignment.loss(cost, ids)
    if value is None:
        return torch.zeros_like(cost)
    gradient, = torch.autograd.grad(value, cost, retain_graph=True, allow_unused=True)
    if gradient is None:
        return torch.zeros_like(cost)
    gradient = gradient.detach()
    gap = deciding_gap(cost)
    per_item = gradient.abs().max(dim=1).values.clamp(min=1e-12)
    share = min(max(float(target_share), 1e-3), 1.0)
    # match the same quantile on both sides: using the gradient's maximum let a single
    # outlier set the scale for every item (measured: 0.75% flippable instead of 50%).
    gap_q = float(torch.quantile(gap, share))
    # the gradient is zero on items no pair covers: the quantile must be taken over the items
    # that DO carry a gradient, or it collapses to 0 and the scale divides by ~0 (measured).
    carrying = per_item[per_item > 0]
    if carrying.numel() == 0:
        return torch.zeros_like(cost)
    grad_q = float(torch.quantile(carrying, share))
    scale = gap_q / max(grad_q, 1e-12)
    # never move an entry by more than one gap: beyond that the term stops expressing a
    # preference and starts rewriting the quantizer (measured unbounded shift: 6.4e3 vs gap
    # 1.4e-3, which flipped a third of the batch).
    limit = gap_q
    return torch.clamp(-scale * gradient, min=-limit, max=limit)


class RoleIndex:
    """The training-side reader of the frozen banks; it never reads validation or test.

    The banks were built once by ``analysis_sid_recall.graphlet_banks``; this class checks their
    hash against ``banks.json`` and refuses to load a file that moved, so the trainer cannot
    silently train on a different sample than the one the report describes.
    """

    def __init__(self, directory: Path = BANK_DIR):
        self.directory = Path(directory)
        record = self.directory / "banks.json"
        if not record.is_file():
            raise FileNotFoundError("the frozen banks are missing: " + str(record))
        self.manifest = json.loads(record.read_text())
        self.banks = {}
        for kind in KINDS:
            path = self.directory / f"{kind}_supervision.npy"
            if not path.is_file():
                raise FileNotFoundError("supervision bank missing: " + str(path))
            digest = hashlib.sha256(np.ascontiguousarray(np.load(path)).tobytes()).hexdigest()
            expected = self.manifest["kinds"][kind]["partitions"]["supervision"]
            if digest != expected:
                raise RuntimeError("%s supervision bank does not match banks.json: %s != %s"
                                   % (kind, digest, expected))
            self.banks[kind] = np.load(path)
        self._eval = {}

    def counts(self) -> dict:
        return {kind: int(len(bank)) for kind, bank in self.banks.items()}

    def partitions(self) -> dict:
        return {"supervision": self.directory,
                "validation": self.manifest["kinds"]["chain"]["partition"]["instances"]["validation"],
                "test": self.manifest["kinds"]["chain"]["partition"]["instances"]["test"]}

    @property
    def reads_validation_or_test(self) -> bool:
        return False

    def evaluation_pairs(self, kind: str) -> set:
        """The unordered pairs of the validation and test banks of one structure."""
        if kind not in self._eval:
            pairs = set()
            for name in ("validation", "test"):
                path = self.directory / f"{kind}_{name}.npy"
                for a, b, c in np.load(path).tolist():
                    for i, j in ((a, b), (b, c), (a, c)):
                        pairs.add((i, j) if i <= j else (j, i))
            self._eval[kind] = pairs
        return self._eval[kind]

    def assert_isolated(self, kind: str, triples) -> None:
        """C9: every pair that actually enters the loss must stay out of the evaluation banks."""
        evaluation = self.evaluation_pairs(kind)
        lost = set()
        for row in np.asarray(triples).tolist():
            a, b, c = int(row[0]), int(row[1]), int(row[2])
            for i, j in ((a, b), (b, c), (a, c)):
                lost.add((i, j) if i <= j else (j, i))
        clash = lost & evaluation
        if clash:
            raise RuntimeError("%s: %d loss pairs also sit in an evaluation bank"
                               % (kind, len(clash)))

    def labeled_pairs(self, kind: str, members: set) -> dict:
        """Pairwise constraints whose two items are both in the batch, with their side.

        Round 1 supervised whole triples, which requires all three items in one batch:
        (4096/24587)**3 = 4.6e-3, measured 96 instances and 7.05% of the batch per step. A pair
        needs only two, so coverage rises to ~1-(1-p)**2 while the delta the issue defines is
        unchanged -- near and far are still the positions of §2, only the third item is free.
        """
        out: dict = {}
        bank = self.banks[kind] if not ARM_CONTROL else self.shuffled(kind)
        inside = np.isin(bank, np.fromiter(members, dtype=np.int64, count=len(members)))
        for row, present in zip(bank.tolist(), inside.tolist()):
            for side, positions in (("near", NEAR[kind]), ("far", FAR[kind])):
                for i, j in positions:
                    if not (present[i] and present[j]):
                        continue
                    key = (min(row[i], row[j]), max(row[i], row[j]))
                    entry = out.setdefault(key, {"near": 0, "far": 0})
                    entry[side] += 1
        return out

    def assert_isolated_pairs(self, kind: str, labelled: dict) -> None:
        """C9 for pairwise constraints: no loss pair may sit in an evaluation bank."""
        evaluation = self.evaluation_pairs(kind)
        clash = set(labelled) & evaluation
        if clash:
            raise RuntimeError("%s: %d loss pairs also sit in an evaluation bank"
                               % (kind, len(clash)))

    def pairwise(self, kind: str, members: set) -> list:
        """The constraints of ``labeled_pairs`` as a list, for the coverage audit."""
        return list(self.labeled_pairs(kind, members))

    def shuffled(self, kind: str) -> np.ndarray:
        """C33's control: the same instances, the same degree profile, the roles permuted."""
        key = ("shuffled", kind)
        if key not in self._eval:
            bank = self.banks[kind]
            rng = np.random.default_rng(230 + len(kind))
            out = bank[rng.permutation(len(bank))].copy()
            self._eval[key] = out
        return self._eval[key]

    def present(self, kind: str, members: set) -> np.ndarray:
        """The instances of this structure whose three items all sit in the batch."""
        bank = self.banks[kind]
        mask = np.isin(bank[:, 0], np.fromiter(members, dtype=np.int64, count=len(members)))
        mask &= np.isin(bank[:, 1], np.fromiter(members, dtype=np.int64, count=len(members)))
        mask &= np.isin(bank[:, 2], np.fromiter(members, dtype=np.int64, count=len(members)))
        return bank[mask]


class HardPrefixAlignment:
    """The trainer-facing object: banks, per-structure margins and the gradient budget."""

    def __init__(self, margin: float = MARGIN, budget: float = BUDGET,
                 directory: Path = BANK_DIR):
        self.margin = float(margin)
        self.budget = float(budget)
        self.index = RoleIndex(directory)

    def counts(self) -> dict:
        return self.index.counts()

    def loss(self, cost: torch.Tensor, ids: torch.Tensor):
        """Sum of the per-structure margins over the instances present in this batch.

        ``cost`` is the level-1 distance matrix the quantizer is about to assign from; ``ids``
        are the item ids of its rows. The term returns a scalar to add to the cost matrix, so
        the assignment it produces is the one the export will see.
        """
        members = set(int(value) for value in ids.tolist())
        position = {int(value): row for row, value in enumerate(ids.tolist())}
        parts, report = [], {}
        for kind in KINDS:
            labelled = self.index.labeled_pairs(kind, members)
            if not labelled:
                continue
            # isolation is asserted on the pairs that actually enter the loss (C9)
            self.index.assert_isolated_pairs(kind, labelled)
            left = torch.tensor([position[int(a)] for a, _b in labelled], dtype=torch.long,
                                device=cost.device)
            right = torch.tensor([position[int(b)] for _a, b in labelled], dtype=torch.long,
                                 device=cost.device)
            coded = torch.softmax(-cost.float() / _temperature(cost), dim=-1)
            share = (coded[left] * coded[right]).sum(-1)
            hard = (hard_codes(cost)[left] == hard_codes(cost)[right]).float()
            # near pairs want the share up, far pairs want it down; one hinge per pair,
            # weighted by how many instances asked for that side
            weight = torch.tensor([entry["near"] - entry["far"] for entry in labelled.values()],
                                  dtype=cost.dtype, device=cost.device)
            target = (weight > 0).float()
            hinge = target * (1.0 - share) + (1.0 - target) * share
            gate = (hinge.detach() > 0).float()
            value = (gate * hinge).sum() / gate.sum().clamp(min=1.0)
            parts.append(value)
            report[kind] = {"pairs": int(len(labelled)),
                            "near": int(sum(entry["near"] > entry["far"]
                                            for entry in labelled.values())),
                            "hard_shared": float(hard.mean()),
                            "scale": float(share.mean())}
        if not parts:
            return None, {}
        return torch.stack(parts).sum(), report
