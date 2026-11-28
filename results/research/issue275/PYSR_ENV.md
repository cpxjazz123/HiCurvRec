# Issue #275 — PySR environment (analysis-only, isolated)

| item | value |
|---|---|
| venv | `results/research/issue275/.venv` (Python 3.9.25, git-ignored) |
| PySR | `pysr==0.19.4` |
| Julia | 1.10.14 (via `juliacall 0.9.23` + `juliapkg 0.1.27`, node-local Julia env under the venv) |
| numpy / pandas / scipy / scikit-learn / sympy | 1.26.4 / 2.3.3 / 1.13.1 / 1.6.1 / 1.14.0 |
| determinism | `random_state=42`, `procs=0`, `multithreading=False`, `deterministic=True` |
| frozen search | binary `+ - * /`, unary `square`, `maxsize=10`, `maxdepth=4`, `niterations=100`, `populations=8`, `timeout_in_seconds=60` per outer fold |
| hardware | CPU only; no GPU work in this issue |
| repository commit at freeze | `b46d60ca` |

The training environments (`genrec_env`, `genrec_env_v2`) were not modified. The venv is excluded
from git by `results/research/issue275/.gitignore`.
