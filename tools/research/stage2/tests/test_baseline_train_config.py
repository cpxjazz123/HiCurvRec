"""The trainer's constants come from one JSON file, and the file is one the digest covers.

The baseline pinned 21 hyper-parameters as literals inside train_rqvae.py. They now
live in `config/train_config.json` beside the trainer, and the trainer fails closed
when a key is missing, unknown or of the wrong type -- a config that silently drops
`EPOCHS` would train for whatever the default is, which is exactly the class of
mistake the frozen budget cannot absorb.

The second half matters as much as the first: `rounds.baseline._canonical_source`
compares a candidate with the baseline, and it only walked `*.py`. A hyper-parameter
moved into JSON would have escaped that comparison, so a candidate could have
changed the budget and still been called a faithful copy.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BASELINE = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
CONFIG = BASELINE / "config" / "train_config.json"
sys.path.insert(0, str(BASELINE))

import train_config  # noqa: E402

#: The frozen budget, written out here so the config cannot drift unnoticed.
EXPECTED = {
    "EPOCHS": 3000,
    "BATCH_SIZE_PER_RANK": 1024,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "HIDDEN_SIZES": [512, 256, 128],
    "CODEBOOK_NUM": 3,
    "CODEBOOK_SIZE": [256, 256, 256],
    "CODEBOOK_DIM": 32,
    "BETA": 0.25,
    "VQ_TYPE": "vq",
    "EMA_DECAY": 0.99,
    "SK_EPSILON": 0.003,
    "SK_ITERS": 50,
    "PCA_DIM": 0,
    "XAVIER_INIT": True,
    "WARMUP_EPOCHS": 50,
    "GRADIENT_CLIP_NORM": 1.0,
    "OPTIMIZER": "AdamW",
    "SEED": 42,
    "NUM_WORKERS": 0,
    "EVAL_INTERVAL": 50,
}


def test_the_config_file_holds_the_frozen_values() -> str:
    if not CONFIG.is_file():
        raise AssertionError("the trainer has no config file to read: " + str(CONFIG))
    values = json.loads(CONFIG.read_text())
    values.pop("schema", None)
    if set(values) != set(EXPECTED):
        raise AssertionError("key set differs: missing %r, unexpected %r"
                             % (sorted(set(EXPECTED) - set(values)),
                                sorted(set(values) - set(EXPECTED))))
    wrong = {k: (v, values[k]) for k, v in EXPECTED.items() if values[k] != v}
    if wrong:
        raise AssertionError("the frozen values changed: %r" % (wrong,))
    return "the config file holds the 21 frozen hyper-parameters"


def test_the_trainer_learns_nothing_from_a_broken_config() -> str:
    good = json.loads(CONFIG.read_text())
    cases = {}
    missing = dict(good)
    missing.pop("EPOCHS")
    cases["a missing key"] = missing
    unknown = dict(good)
    unknown["EPOCHS_EXTRA"] = 1
    cases["an unknown key"] = unknown
    bad_type = dict(good)
    bad_type["SEED"] = "42"
    cases["a wrong type"] = bad_type
    bad_size = dict(good)
    bad_size["CODEBOOK_SIZE"] = 256
    cases["a scalar codebook size"] = bad_size

    failures = []
    for label, payload in cases.items():
        with tempfile.TemporaryDirectory(dir=REPO / "results" / "research") as folder:
            path = Path(folder) / "train_config.json"
            path.write_text(json.dumps(payload))
            try:
                train_config.load_train_config(path)
            except train_config.ConfigError:
                continue
            failures.append(label)
    if failures:
        raise AssertionError("a broken config was accepted: %s" % ", ".join(failures))
    return "a missing key, an unknown key, a wrong type and a scalar codebook size are refused"


def test_the_loader_reports_the_file_it_read() -> str:
    config = train_config.load_train_config(CONFIG)
    if config.EPOCHS != EXPECTED["EPOCHS"] or str(config.SEED) != str(EXPECTED["SEED"]):
        raise AssertionError("the loader does not expose the values: %r" % (config,))
    if len(config.sha256) != 64:
        raise AssertionError("the loader did not record the file's sha256: %r" % (config.sha256,))
    if list(config.HIDDEN_SIZES) != EXPECTED["HIDDEN_SIZES"]:
        raise AssertionError("HIDDEN_SIZES was not converted: %r" % (config.HIDDEN_SIZES,))
    return "the loader returns the values, the tuples, and the sha256 of the file it read"


def test_the_trainer_defines_no_hyper_parameter_literal() -> str:
    text = (BASELINE / "train_rqvae.py").read_text(encoding="utf-8")
    offenders = []
    for name in EXPECTED:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(name + " ") or "=" not in stripped:
                continue
            right = stripped.split("=", 1)[1].split("#")[0].strip()
            if right and right[0] in "-0123456789" or right in ("True", "False"):
                offenders.append(stripped)
    if offenders:
        raise AssertionError("literals left in the trainer: %r" % (offenders[:5],))
    return "every hyper-parameter in the trainer is read from the config"


def test_the_baseline_digest_covers_the_config_file() -> str:
    import hashlib
    sys.path.insert(0, str(REPO))
    from tools.research.rounds import baseline
    table = baseline._canonical_source(BASELINE)
    key = "config/train_config.json"
    if key not in table:
        raise AssertionError("the baseline digest ignores the config: a candidate could "
                             "change the budget and pass as a faithful copy")
    expected = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if table[key] != expected:
        raise AssertionError("the digest does not hash the config's bytes: %r vs %r"
                             % (table[key], expected))
    return "the candidate/baseline comparison includes config/train_config.json"


def test_the_frozen_constant_check_reads_the_config() -> str:
    import shutil
    import tempfile
    sys.path.insert(0, str(REPO))
    from tools.research.stage2 import checks
    report = checks.check_constants(BASELINE)
    if report["status"] != "PASS":
        raise AssertionError("the config-backed baseline fails the frozen-constant check: %r"
                             % (report.get("detail") or report,))
    with tempfile.TemporaryDirectory(dir=REPO / "results" / "research") as folder:
        copy = Path(folder) / "TIGER_FAKE"
        shutil.copytree(BASELINE, copy, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "logs"))
        payload = json.loads((copy / "config" / "train_config.json").read_text())
        payload["EPOCHS"] = 10
        (copy / "config" / "train_config.json").write_text(json.dumps(payload))
        tampered = checks.check_constants(copy)
    if tampered["status"] != "FAIL":
        raise AssertionError("a tree whose config trains 10 epochs passed the check: %r"
                             % (tampered,))
    return "the constant check reads the config and refuses a changed budget"


def main() -> int:
    failures = 0
    for test in (test_the_config_file_holds_the_frozen_values,
                 test_the_trainer_learns_nothing_from_a_broken_config,
                 test_the_loader_reports_the_file_it_read,
                 test_the_trainer_defines_no_hyper_parameter_literal,
                 test_the_baseline_digest_covers_the_config_file,
                 test_the_frozen_constant_check_reads_the_config):
        try:
            note = test()
            print("PASS %s: %s" % (test.__name__, note))
        except AssertionError as exc:
            failures += 1
            print("FAIL %s: %s" % (test.__name__, exc))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
