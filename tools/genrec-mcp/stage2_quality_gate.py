"""Stage 2 -> Stage 3 admission: technical validity of the exported SID.

Issue #120 removes every quality floor from admission. ER, level-two prefix
counts, collisions, code usage and redundancy are still MEASURED, but they are
diagnostics: a Stage 2 run that produced a technically valid SID proceeds to
Stage 3 whatever those numbers look like.

What is still refused is a Stage 2 output that cannot be used at all: an
unreadable or malformed SID file, an item mapping that is not the canonical
decimal id space, a SID whose token width or token values are malformed, or
full SIDs that are not unique after the deterministic disambiguation token.
"""
from __future__ import annotations

import array
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

#: The repository root: where the arms' source trees live.
REPOSITORY = Path(__file__).resolve().parents[2]

#: No promotion floor exists any more. The frozen Resample numbers stay only as
#: the reference values the diagnostics are compared against.
FROZEN_BASELINE_L2_PREFIXES = 16145  # Verified reference for the diagnostic report
#: Owner instruction (2026-11-03, revised the same day): a SID whose LEARNED
#: three-level collision exceeds this share is refused, and an exported SID that
#: disagrees with the trainer's own measured collision is refused as
#: post-processing. This deliberately overrides the technical-validity-only policy
#: of issue #120; the equality boundary is admitted ("> 35%").
L3_COLLISION_MAX = 0.35
#: Owner instruction (2026-11-08): the five arms of issue #139 are admitted to
#: Stage 3 whatever their learned collision, because the goal of that issue is to
#: compare five mechanisms end to end and each of them has to produce a recall
#: number. The collision is still measured and recorded as a diagnostic; the
#: exception is scoped to these directory names and relaxes nothing else.
ISSUE139_STAGE3_EXCEPTION = (
    "TIGER_mixed_rq_F0",
    "TIGER_mixed_rq_G",
    "TIGER_mixed_rq_L",
    "TIGER_mixed_rq_E",
    "TIGER_mixed_rq_A",
)
#: The learned collision is read from the run's own metrics stream; an exported
#: SID whose collision disagrees with it was post-processed after training, which
#: the owner forbids. The tolerance absorbs the fact that the exported SID is
#: computed on all items while the log's last row is the final epoch's batch.
LEARNED_COLLISION_TOLERANCE = 0.02
FROZEN_RESAMPLE_VALID_L1_ER = 16.38953942232631
# Git blob identities pin these exact frozen source files, not merely their paths.
FROZEN_BASELINE_BLOB_SHA1 = "d97d24a7a63891682d2b68c09a89164de2031ee9"
FROZEN_VALID_PAIRS_BLOB_SHA1 = "11d76dd452f3a27d1f6bf79fe7673bc6505733ef"
FROZEN_VALID_META_BLOB_SHA1 = "a0121691dc95168169bffdc3b6018608580468da"
EXPECTED_PAIRS = 52855
EXPECTED_NEGATIVES = 10
EXPECTED_SEED = 42


class QualityGateError(ValueError):
    pass


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob_sha1(path):
    """Compute immutable Git blob identity for an on-disk byte string."""
    content = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") +
                        b"\0" + content).hexdigest()


def _sid(path):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualityGateError("unreadable SID: " + str(exc)) from exc
    if not isinstance(raw, dict) or not raw:
        raise QualityGateError("SID must be a nonempty item-id mapping")
    output = {}
    width = None
    for k, v in raw.items():
        if not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k:
            raise QualityGateError("SID has a noncanonical item ID")
        if not isinstance(v, list) or len(v) < 3:
            raise QualityGateError("SID has fewer than three integer tokens")
        if width is None:
            width = len(v)
        if len(v) != width or any(type(x) is not int or x < 0 for x in v):
            raise QualityGateError("SID tokens are malformed or inconsistent")
        output[int(k)] = tuple(v)
    return output


def _npy_int(blob):
    """Read C-order NumPy int32/int64 arrays without importing numpy/torch."""
    if blob[:6] != b"\x93NUMPY" or len(blob) < 12:
        raise QualityGateError("invalid NPY header")
    major = blob[6]
    if major == 1:
        length = struct.unpack_from("<H", blob, 8)[0]
        start = 10
    elif major in (2, 3):
        length = struct.unpack_from("<I", blob, 8)[0]
        start = 12
    else:
        raise QualityGateError("unsupported NPY version")
    try:
        header = ast.literal_eval(blob[start:start + length].decode("latin1"))
    except (SyntaxError, ValueError) as exc:
        raise QualityGateError("invalid NPY metadata") from exc
    desc = header.get("descr")
    typ = {"<i4": "i", "|i4": "i", "<i8": "q", "|i8": "q"}.get(desc)
    shape = header.get("shape")
    if not typ or not isinstance(shape, tuple) or not shape or \
            any(type(n) is not int or n < 1 for n in shape) or \
            header.get("fortran_order") is not False:
        raise QualityGateError("unexpected validation-array dtype, shape or layout")
    count = math.prod(shape)
    nbytes = count * (4 if typ == "i" else 8)
    data = blob[start + length:]
    if len(data) != nbytes or sys.byteorder != "little":
        raise QualityGateError("validation-array byte count or endianness mismatch")
    out = array.array(typ)
    out.frombytes(data)
    return tuple(shape), out


def _pairs(npz, meta_path):
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualityGateError("missing frozen validation metadata: " + str(exc)) from exc
    expected = {"anchors": [EXPECTED_PAIRS], "positives": [EXPECTED_PAIRS],
                "negatives": [EXPECTED_PAIRS, EXPECTED_NEGATIVES]}
    if (meta.get("seed") != EXPECTED_SEED or
            meta.get("k_negatives") != EXPECTED_NEGATIVES or
            meta.get("shapes") != expected or
            meta.get("relation") != "valid_transition: history[-1] -> target, from valid.parquet" or
            not all(meta.get("checks", {}).get(x) is True for x in
                    ("no_invariant_violations", "all_ids_in_range",
                     "positive_is_always_a_valid_successor",
                     "no_positive_is_a_train_successor",
                     "no_negative_is_a_train_or_valid_successor_of_its_anchor"))):
        raise QualityGateError("independent validation pair protocol mismatch")
    try:
        with zipfile.ZipFile(npz) as z:
            if not all(k + ".npy" in z.namelist() for k in expected):
                raise QualityGateError("validation arrays missing")
            arrays = {name: _npy_int(z.read(name + ".npy"))
                      for name in expected}
    except (OSError, zipfile.BadZipFile) as exc:
        raise QualityGateError("unreadable validation pairs: " + str(exc)) from exc
    for name, shape in expected.items():
        if list(arrays[name][0]) != shape:
            raise QualityGateError(name + " shape differs from frozen valid pairs")
    return arrays["anchors"][1], arrays["positives"][1], arrays["negatives"][1]


def _measures(sids, anchors, positives, negatives):
    n = len(anchors)
    if len(positives) != n or len(negatives) != n * EXPECTED_NEGATIVES or n == 0:
        raise QualityGateError("inconsistent validation sample counts")
    try:
        first = {item: value[0] for item, value in sids.items()}
        positive_hits = sum(first[int(a)] == first[int(b)]
                            for a, b in zip(anchors, positives))
        negative_hits = sum(
            first[int(a)] == first[int(negatives[i * EXPECTED_NEGATIVES + j])]
            for i, a in enumerate(anchors) for j in range(EXPECTED_NEGATIVES))
    except KeyError as exc:
        raise QualityGateError("validation pair references item missing from SID: " +
                               str(exc)) from exc
    if negative_hits == 0:
        raise QualityGateError("L1 Hard ER undefined (zero negative prefix hits)")
    return {
        "valid_l1_positive_hit_rate": positive_hits / n,
        "valid_l1_negative_hit_rate": negative_hits / (n * EXPECTED_NEGATIVES),
        "valid_l1_hard_er": (positive_hits * EXPECTED_NEGATIVES) / negative_hits,
        "positive_hits": positive_hits, "negative_hits": negative_hits,
        "pairs": n,
    }


def three_level_collision(sid: dict) -> float | None:
    """The share of items whose (k1,k2,k3) triple is shared with another item."""
    if not sid:
        return None
    counts: dict = {}
    for value in sid.values():
        key = tuple(value[:3])
        counts[key] = counts.get(key, 0) + 1
    shared = sum(count for count in counts.values() if count > 1)
    return shared / len(sid)


def learned_collision(result_dir) -> float | None:
    """The final collision the trainer itself measured, from its metrics stream.

    The metrics live in the run's own logs, and the repository carries two
    layouts: the older arms write them into the result directory's logs and the
    current trainer into the TREE's logs (the tree is the arm's source
    directory under stage2_RQ-VAE/).  Both are the run's own stream, so both are
    read; looking at one of them only made the gate fail closed for every arm
    trained with the other.
    """
    result_dir = Path(result_dir)
    # A recovered single-run stream wins over a shared file: on 2026-11-08 a
    # duplicate kept appending to the shared stream after the delivered run had
    # finished, and its stale collision was read as the learned value.
    candidates = [
        REPOSITORY / "stage2_RQ-VAE" / result_dir.name / "logs" / "run_metrics.jsonl",
        result_dir / "logs" / "run_metrics.jsonl",
        result_dir / "logs" / "training_metrics.jsonl",
        REPOSITORY / "stage2_RQ-VAE" / result_dir.name / "logs" / "training_metrics.jsonl",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        value = None
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") == "train" and record.get("collision") is not None:
                    value = float(record["collision"])
        except OSError:
            continue
        if value is not None:
            return value
    return None


def exported_raw_collision(sid) -> float:
    """The collision the TRAINER measures: one minus the distinct triplets.

    The gate's own three_level_collision is item-weighted (the share of items
    that share their triplet with another item) while the trainer logs the share
    of distinct triplets.  Comparing the two as if they were the same measure
    refused arms whose SID was never touched, so the learned check compares the
    trainer's measure with the SAME measure computed on the exported table.
    """
    triples = {tuple(value[:3]) for value in sid.values()}
    return 1.0 - (len(triples) / max(len(sid), 1))


def collision_gate(candidate: dict, learned: float | None = None,
                   exported_raw: float | None = None,
                   arm: str | None = None) -> str | None:
    """The single quality rule: the LEARNED L3 collision must not exceed the gate.

    ``arm`` names the arm the SID belongs to. An arm in the explicit
    ``ISSUE139_STAGE3_EXCEPTION`` is admitted whatever its collision, and the
    exception is recorded in the evidence rather than applied silently.
    """
    if arm in ISSUE139_STAGE3_EXCEPTION:
        return None
    collision = candidate.get("l3_collision")
    if collision is None:
        return ("the three-level collision could not be measured, so the 20% "
                "quality gate fails closed")
    if float(collision) > L3_COLLISION_MAX:
        return ("three-level collision %.6f exceeds the %.2f gate"
                % (float(collision), L3_COLLISION_MAX))
    if learned is None:
        return ("the learned collision of this SID could not be read from the run's "
                "metrics, so the collision gate fails closed")
    if float(learned) > L3_COLLISION_MAX:
        return ("the learned three-level collision %.6f exceeds the %.2f gate; the "
                "collision must come from Stage2 training, not from an export step"
                % (float(learned), L3_COLLISION_MAX))
    if exported_raw is None:
        return ("the exported identifier's own collision could not be measured, "
                "so the collision gate fails closed")
    if abs(float(learned) - float(exported_raw)) > LEARNED_COLLISION_TOLERANCE:
        return ("the exported SID reports collision %.6f while the training log "
                "measured %.6f; export-time post-processing is not allowed"
                % (float(collision), float(learned)))
    return None


def evaluate(sid_path, baseline_path, pairs_path, meta_path,
             *, enforce_collision: bool = True):
    """Check that the exported SID is usable, and measure the diagnostics.

    Only the candidate's own technical validity can refuse an admission. The
    reference artifacts and the derived quality numbers are reported; a missing
    or altered reference changes the diagnostics, not the usability of the
    candidate's SID. The owner's learned collision gate is an EXPORT gate, so the
    round pipeline (CLAUDE.md 19: admission is technical validity only) passes
    ``enforce_collision=False`` and still records the measured collision.
    """
    evidence = {"sid": str(sid_path), "baseline": str(baseline_path),
                "pairs": str(pairs_path), "meta": str(meta_path),
                "admission_policy": "technical validity AND the owner's "
                                    "LEARNED three-level-collision gate (<= 35%, "
                                    "no export-time dedup)"}
    try:
        sid = _sid(sid_path)
        if not sid:
            raise QualityGateError("candidate SID is empty")
        failures = []
        if len(set(sid.values())) != len(sid):
            failures.append("full candidate Semantic IDs are not unique after "
                            "the disambiguation token")
        reference_status = {}
        baseline = None
        try:
            for label, path, want in (
                ("baseline SID", baseline_path, FROZEN_BASELINE_BLOB_SHA1),
                ("independent validation pairs", pairs_path, FROZEN_VALID_PAIRS_BLOB_SHA1),
                ("validation metadata", meta_path, FROZEN_VALID_META_BLOB_SHA1),
            ):
                if _git_blob_sha1(path) != want:
                    reference_status[label] = "no longer matches the frozen Git blob"
            baseline = _sid(baseline_path)
            if set(sid) != set(baseline):
                reference_status["item set"] = ("candidate and frozen baseline "
                                                "have different item sets")
        except (QualityGateError, OSError, TypeError, ValueError, KeyError) as exc:
            reference_status["reference"] = str(exc)

        collision = three_level_collision(sid)
        candidate = {"l3_collision": collision}
        baseline_metrics = {}
        candidate_prefixes = len({v[:2] for v in sid.values()})
        try:
            anchors, positives, negatives = _pairs(pairs_path, meta_path)
            candidate.update(_measures(sid, anchors, positives, negatives))
            if baseline is not None:
                baseline_metrics = _measures(baseline, anchors, positives, negatives)
        except (QualityGateError, OSError, TypeError, ValueError, KeyError) as exc:
            reference_status["diagnostics"] = str(exc)
        arm_name = Path(sid_path).parent.name
        collision_refusal = collision_gate(
            candidate, learned_collision(Path(sid_path).parent),
            exported_raw_collision(sid) if sid else None, arm=arm_name)
        evidence["collision_gate_exception"] = (
            arm_name if arm_name in ISSUE139_STAGE3_EXCEPTION else None)
        if collision_refusal is not None and enforce_collision:
            failures.append(collision_refusal)
        evidence.update({
            "candidate": {**candidate, "level2_distinct_prefixes": candidate_prefixes,
                          "full_unique_items": len(sid)},
            "baseline": {**baseline_metrics,
                         "level2_distinct_prefixes": len({v[:2] for v in baseline.values()})
                         if baseline else None},
            "reference_status": reference_status or {"status": "verified"},
            "thresholds": {"l3_collision_max": L3_COLLISION_MAX},
            "diagnostic_only": ["valid_l1_hard_er", "valid_l1_positive_hit_rate",
                                "valid_l1_negative_hit_rate", "level2_distinct_prefixes",
                                "full_unique_items"],
            "quality_gate": {"l3_collision_max": L3_COLLISION_MAX,
                             "learned_collision": learned_collision(
                                 Path(sid_path).parent),
                             "learned_tolerance": LEARNED_COLLISION_TOLERANCE,
                             "l3_collision_refusal": collision_refusal},
            "sha256": {key: _digest(path) for key, path in (
                ("sid", sid_path), ("baseline", baseline_path),
                ("validation_pairs", pairs_path), ("validation_meta", meta_path))
                if Path(path).is_file()},
        })
        return {"passed": not failures, "refused": "; ".join(failures) if failures else None,
                **evidence}
    except (QualityGateError, OSError, TypeError, ValueError, KeyError) as exc:
        return {"passed": False, "refused": str(exc), **evidence}
