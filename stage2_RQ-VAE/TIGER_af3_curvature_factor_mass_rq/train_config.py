"""The trainer's constants, read from `config/train_config.json` and nowhere else.

The budget is frozen: 3000 epochs, per-rank batch 1024, seed 42. Keeping those as
literals inside `train_rqvae.py` meant the only way to see them was to read the
trainer, and the only way to change them was to edit the file that also holds the
training loop. They now live in one JSON file beside the trainer.

The file is validated here, not in the loop: a missing key, an unknown key, a wrong
type or an out-of-range value raises before any GPU is touched. A config that
silently lost `EPOCHS` would otherwise train to some default and look successful.

Paths are NOT configurable: they derive from the repository marker and `ARM_NAME`,
both of which belong to the framework (CLAUDE.md 19b0, 67).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

#: key -> (kind, extra) with `extra` the allowed set for a string or None.
FIELDS = {
    "EPOCHS": ("int", None),
    "BATCH_SIZE_PER_RANK": ("int", None),
    "LR": ("float", None),
    "WEIGHT_DECAY": ("float", None),
    "HIDDEN_SIZES": ("int_list", None),
    "CODEBOOK_NUM": ("int", None),
    "CODEBOOK_SIZE": ("int_list", None),
    "CODEBOOK_DIM": ("int", None),
    "BETA": ("float", None),
    "VQ_TYPE": ("str", {"vq", "ema", "simvq"}),
    "EMA_DECAY": ("float", None),
    "SK_EPSILON": ("float", None),
    "SK_ITERS": ("int", None),
    "PCA_DIM": ("int", None),
    "XAVIER_INIT": ("bool", None),
    "WARMUP_EPOCHS": ("int", None),
    "GRADIENT_CLIP_NORM": ("float", None),
    "OPTIMIZER": ("str", {"AdamW", "Adagrad"}),
    "SEED": ("int", None),
    "NUM_WORKERS": ("int", None),
    "EVAL_INTERVAL": ("int", None),
}

#: key -> (minimum, maximum) with either bound optional. Counts are positive, the
#: decay factors stay inside (0, 1) and the clip norm is not negative.
RANGES = {
    "EPOCHS": (1, None),
    "BATCH_SIZE_PER_RANK": (1, None),
    "LR": (0.0, None),
    "WEIGHT_DECAY": (0.0, None),
    "CODEBOOK_NUM": (1, None),
    "CODEBOOK_DIM": (1, None),
    "BETA": (0.0, None),
    "EMA_DECAY": (0.0, 1.0),
    "SK_EPSILON": (0.0, None),
    "SK_ITERS": (0, None),
    "PCA_DIM": (0, None),
    "WARMUP_EPOCHS": (0, None),
    "GRADIENT_CLIP_NORM": (0.0, None),
    "SEED": (0, None),
    "NUM_WORKERS": (0, None),
    "EVAL_INTERVAL": (1, None),
}


class ConfigError(RuntimeError):
    """The config file cannot be used as-is; training must not start."""


def _typed(name: str, value, kind: str, allowed):
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError("%s must be true or false, got %r" % (name, value))
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError("%s must be an integer, got %r" % (name, value))
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("%s must be a number, got %r" % (name, value))
        return float(value)
    if kind == "int_list":
        if not isinstance(value, list) or not value:
            raise ConfigError("%s must be a non-empty list of integers, got %r" % (name, value))
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                raise ConfigError("%s holds a value that is not a positive integer: %r"
                                  % (name, item))
        return tuple(value)
    if kind == "str":
        if not isinstance(value, str) or not value:
            raise ConfigError("%s must be a non-empty string, got %r" % (name, value))
        if allowed is not None and value not in allowed:
            raise ConfigError("%s is %r; the supported values are %s"
                              % (name, value, ", ".join(sorted(allowed))))
        return value
    raise ConfigError("unknown field kind %r for %s" % (kind, name))


def _check_range(name: str, value):
    bounds = RANGES.get(name)
    if bounds is None:
        return
    low, high = bounds
    if low is not None and value < low:
        raise ConfigError("%s must be >= %r, got %r" % (name, low, value))
    if high is not None and value > high:
        raise ConfigError("%s must be <= %r, got %r" % (name, high, value))


def load_train_config(path: Path) -> SimpleNamespace:
    """Read and validate the config; raise ConfigError instead of guessing."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError("the trainer has no config file at " + str(path))
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ConfigError("the config is not valid JSON: %s" % exc) from None
    if not isinstance(payload, dict):
        raise ConfigError("the config must be a JSON object, got %r" % (type(payload).__name__,))

    missing = sorted(set(FIELDS) - set(payload))
    unknown = sorted(set(payload) - set(FIELDS))
    if missing or unknown:
        raise ConfigError("the config's keys are wrong: missing %s, unknown %s"
                          % (missing or "none", unknown or "none"))

    values = {}
    for name, (kind, allowed) in FIELDS.items():
        value = _typed(name, payload[name], kind, allowed)
        _check_range(name, value if not isinstance(value, tuple) else max(value))
        values[name] = value

    # The two codebook-shape relations the quantiser relies on are checked here as
    # well: a four-entry sizes list with CODEBOOK_NUM 3 would index past the end.
    if len(values["CODEBOOK_SIZE"]) not in (1, values["CODEBOOK_NUM"]):
        raise ConfigError("CODEBOOK_SIZE must hold one entry or CODEBOOK_NUM entries "
                          "(num=%d, sizes=%d)" % (values["CODEBOOK_NUM"],
                                                  len(values["CODEBOOK_SIZE"])))
    if values["SK_EPSILON"] > 0 and values["SK_ITERS"] < 1:
        raise ConfigError("Sinkhorn is enabled but SK_ITERS is %d" % values["SK_ITERS"])

    values["path"] = str(path)
    values["sha256"] = hashlib.sha256(raw).hexdigest()
    return SimpleNamespace(**values)
