#!/usr/bin/env bash
# Remove the binary artifacts from the whole history so GitHub's LFS quota is freed.
#
# Why this is needed: LFS objects are immutable and every rerun adds one, so
# deleting the files in a new commit frees nothing. Only a history rewrite drops
# the objects from the reachable history, after which GitHub garbage-collects
# them. The checkpoints are regenerable from item_sids.json (which stays in git)
# and one Stage2 rerun of about 22 minutes.
#
# What changes: every commit SHA. Anyone with a clone has to re-clone, and the
# push has to be forced, which CLAUDE.md section 44 forbids without explicit
# authorization. Run this only with that authorization.
#
# Usage: tools/research/lfs_slim.sh --authorized
set -euo pipefail

if [[ "${1:-}" != "--authorized" ]]; then
    echo "refusing: this rewrites history and force-pushes." >&2
    echo "re-run with --authorized once the force push is explicitly approved." >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="../GeneRec_backup_${STAMP}.bundle"

# 1. Back up every ref, and record what the LFS index held before the rewrite.
git bundle create "$BACKUP" --all
git lfs ls-files -l > "../GeneRec_lfs_before_${STAMP}.txt" || true
echo "backup: $BACKUP"

# 2. Rewrite: drop the binaries from every commit. Text evidence stays.
PATH="$HOME/.local/bin:$PATH" git filter-repo --invert-paths \
    --path-glob '*.pth' --path-glob '*.pt' --path-glob '*.npy' --force

# 3. filter-repo drops the remotes; put the one allowed remote back.
git remote add origin https://github.com/cpxjazz123/HiCurvRec.git

# 4. The force push that actually releases the quota. Left explicit on purpose.
git push --force origin main

# 5. Free the local object store of everything no longer reachable.
git lfs prune --verbose || true

git rev-list --count HEAD
git ls-files '*.pth' '*.npy' '*.pt' | wc -l
