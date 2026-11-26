## Issue #267 acceptance frozen

- Acceptance: C1–C9, issue body sha256 `fe40b2ba18899a41b2ef62a121a315923cb60f4cced22f2b233d255b5fb4301e` (`RUN_LOG.md`).
- `INFRA_MODE=FROZEN`: issue text explicitly sets FROZEN and allows only candidate mechanism plus issue-local tests/diagnostics.
- Training: **required** by issue §4: one Stage2 3000-epoch four-card run and one Stage3 150-epoch run.
- Public-infra snapshot recorded in `INFRA_SNAPSHOT_START.json` (61 protected files, 0 missing); parent is #265 R2 109.
- Root-cause gate is active: full q/z curvature diagnostics and six admission questions must precede the mechanism contract/implementation.
