"""Fail-closed Stage2 -> Stage3 quality admission; stdlib only, no training.

Hard-L1 ER and L1 positive HitRate are measured on the frozen independent
validation pairs; both are distinct from downstream Stage3 Recall@10.
The HitRate cutoff is deliberately unset until an authorised numerical
threshold has been recovered and frozen in source. Missing policy never passes.
"""
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

MIN_VALID_L1_HARD_ER = 16.38953942232631  # STRICTLY greater; frozen Resample reference
MIN_VALID_L1_HITRATE = None  # Not recovered: fail closed, never guess or silently skip
FROZEN_BASELINE_L2_PREFIXES = 7414
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


def evaluate(sid_path, baseline_path, pairs_path, meta_path):
    """Recompute all gates from source-bound artifacts, never accept a PASS flag."""
    evidence = {"sid": str(sid_path), "baseline": str(baseline_path),
                "pairs": str(pairs_path), "meta": str(meta_path)}
    try:
        sid = _sid(sid_path)
        baseline = _sid(baseline_path)
        if set(sid) != set(baseline):
            raise QualityGateError("candidate and frozen baseline have different item sets")
        if len(set(sid.values())) != len(sid):
            raise QualityGateError("full candidate Semantic IDs are not unique")
        candidate_prefixes = len({v[:2] for v in sid.values()})
        baseline_prefixes = len({v[:2] for v in baseline.values()})
        if baseline_prefixes != FROZEN_BASELINE_L2_PREFIXES:
            raise QualityGateError("frozen baseline prefix reference has changed")
        anchors, positives, negatives = _pairs(pairs_path, meta_path)
        candidate = _measures(sid, anchors, positives, negatives)
        reference = _measures(baseline, anchors, positives, negatives)
        evidence.update({
            "candidate": {**candidate, "level2_distinct_prefixes": candidate_prefixes,
                          "full_unique_items": len(sid)},
            "baseline": {**reference, "level2_distinct_prefixes": baseline_prefixes},
            "thresholds": {"valid_l1_hard_er_strict_gt": MIN_VALID_L1_HARD_ER,
                           "valid_l1_positive_hit_rate_min": MIN_VALID_L1_HITRATE,
                           "level2_prefixes_strict_gt": baseline_prefixes},
            "sha256": {key: _digest(path) for key, path in (
                ("sid", sid_path), ("baseline", baseline_path),
                ("validation_pairs", pairs_path), ("validation_meta", meta_path))},
        })
        failures = []
        if not candidate["valid_l1_hard_er"] > MIN_VALID_L1_HARD_ER:
            failures.append("Hard-L1 ER must be strictly above frozen cutoff")
        cutoff = MIN_VALID_L1_HITRATE
        if type(cutoff) not in (float, int) or not math.isfinite(cutoff) or not 0 < cutoff <= 1:
            failures.append("HitRate hard cutoff is not yet specified in source; refusing Stage3")
        elif candidate["valid_l1_positive_hit_rate"] < cutoff:
            failures.append("independent-validation L1 positive HitRate below cutoff")
        if candidate_prefixes <= baseline_prefixes:
            failures.append("second-level SID prefixes do not exceed frozen baseline")
        return {"passed": not failures, "refused": "; ".join(failures) if failures else None,
                **evidence}
    except (QualityGateError, OSError, TypeError, ValueError, KeyError) as exc:
        return {"passed": False, "refused": str(exc), **evidence}
