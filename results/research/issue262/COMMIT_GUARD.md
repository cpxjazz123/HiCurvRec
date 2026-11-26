# Commit preflight

`python3 tools/commit_guard.py --max-paths 1000 --max-lines 70000` returned 0
with 893 changed paths and 54,305 changed lines; no path reaches 2 MiB.

The path/line counts include pre-existing user working-tree changes outside
`results/research/issue262/`, which are not authored by this issue and are not
selected by the issue commit. The issue's large aligned curvature vectors,
q files, prediction scatter and stdout logs live under
`results/research/issue262/_raw/` and `_stdout/` and are ignored by the local
`.gitignore`; only compact JSON/CSV/Markdown evidence and the analysis scripts
are intended for the commit.
