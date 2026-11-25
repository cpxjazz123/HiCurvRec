# Issue #259 incremental-validation contract (frozen before implementation)

This contract defines the machine objects and validity rules that `tools/iteration/` must implement. It extends,
and does not replace, the research `CONTRACT.md` rules of a research issue.

## 1. Validation unit

A unit is `(test_id, declared_inputs)`:

- `test_id` — stable string supplied by the caller (e.g. `candidate_suite`, `admission_entry`); it never derives
  from a directory, run name or timestamp.
- `declared_inputs` — explicit `tests`, `code`, `assets`, `configs`, `extra` (non-Python/plugin/launch-script files
  or directories), and `roots` for import resolution.
- `runtime` — whether environment identity participates; `runtime_keys` selects which environment fields matter
  for this unit (e.g. CPU-only units exclude device count).

## 2. Dependency closure

The closure starts from `tests + code` (not from JSON configs; configs are hashed, not parsed as Python) and
follows:

1. absolute imports;
2. relative imports at any level (`from .`, `from ..pkg.mod import name`), including package `__init__.py` chains;
3. statically resolvable dynamic imports (`importlib.import_module("a.b")`, `__import__("a.b")` with a string
   literal);
4. declared `extra` files/directories.

If a dynamic import has a non-literal argument, or a declared input is missing, or a relative import cannot be
resolved, the closure is marked **unresolved** and the unit is forced to RUN (conservative expansion). The
unresolved marker and location are part of the fingerprint, so a later resolution changes the digest.

## 3. Fingerprint

`fingerprint = SHA256(canonical JSON of)`:

- `test_id`;
- `code`: repo-relative path → sha256 for every closure file, plus package `__init__` files;
- `assets`, `configs`, `extra`: path (or directory entry) → sha256; directories expanded recursively;
- `tests`: the unit's own test files;
- `runtime`: filtered `runtime_identity()` (`interpreter`, `python`, `implementation`, `torch`, `devices`, `cuda`)
  according to `runtime_keys`; `None` when `runtime=False`;
- `unresolved`: sorted markers (empty when conclusive);
- `missing`: sorted declared inputs that do not exist.

Directory or run names do not enter the fingerprint; content does. A same-name file with different bytes changes
the digest.

## 4. Receipt (evidence) validity

A receipt may be reused only if **all** hold:

1. `schema == "genrec.pipeline.receipt/1"`, `test_id` matches, `digest` matches the current fingerprint;
2. `status == "PASS"`;
3. `fingerprint` stored in the receipt is present (new receipts) and its digest matches;
4. `evidence.command` is a non-empty string or argv list; `evidence.exit_code == 0`;
5. `evidence.log` exists, is a regular file and is non-empty;
6. `evidence.output` exists and, when `output_sha256` is recorded, matches the current bytes;
7. no unresolved/missing marker exists for the current fingerprint.

Legacy receipts missing any required field are explicitly invalid (no fabricated back-fill). Failures never
overwrite earlier evidence: each execution is written under
`<cache>/<test_id>/<digest>/<generation>.json`; legacy `<test_id>/<digest>.json` remains readable.

## 5. Plan

`plan(specs, cache, changed)` returns machine-readable `run` (alias `rerun`) and `reuse` entries with a reason:

- `no receipt`, `previous status FAIL/NOT_RUN`, `receipt evidence invalid: …`, `missing dependency …`,
  `unresolved dynamic dependency …`, or `changed dependency …` (current **or previous** closure, so deletions
  and renames invalidate);
- a changed path outside both closures does not invalidate anything;
- a valid receipt for the same fingerprint is reused even when the cumulative `changed` list still contains older
  paths;
- mandatory gates (`admission_entry`, behaviour graph, real entry, final run identity) are declared with
  `runtime=True` and no cache bypass; they are always RUN unless the caller explicitly reuses a receipt whose
  evidence is valid and whose command is the real entry.

## 6. Durable execution

`execution.run(unit_id, argv, out_dir=..., ...)`:

- creates the output/log directory before spawning; refuses if it is not writable;
- writes `command.json` (argv, cwd, interpreter, versions, start time, pid, generation) before starting;
- streams stdout/stderr to `run.log`; on normal end writes exit code, end time, duration, log sha256 and status
  `PASS` only when exit code is 0;
- timeout/interrupt/exception → status `INCOMPLETE`, log retained, never PASS;
- training/evaluation never uses this wrapper; they stay on `stage2_start` / `stage3_start` / `launcher.launch`.

## 7. Infra integrity

Snapshot/compare covers the protected list (iteration package, Stage2 research package, verification package,
`assets/manifest.json`, Stage3 trainer, `CLAUDE.md`, skill). Normal run outputs (logs, checkpoints, receipts,
reports) are outside the protected list. Checks run at round start, before formal dispatch and before
archive/close; adjacent stages share one result.

## 8. Saved validation metrics

Validation recall is read only from an explicit saved artefact that carries its own run identity and SID hash.
If no artefact exists for that identity, the result is `NOT_AVAILABLE`; the test split is never substituted and
no value is fabricated. A parameterised re-inference is out of scope for this issue (no training, no GPU work).
