### #258 checklist frozen — FROZEN mode, training required

**Issue body**: 6429 bytes, sha256 `a244f1b4314956f1d22622f3f06502b685d9da2e3cdc8801d128b7735931891e`.

**INFRA_MODE = FROZEN.** The issue authorizes no infrastructure work: “本轮只使用既有 Infra,**严禁开发、修改、
扩充任何新的 Infra**”; “禁止新建针对本轮的通用控制器、launcher、评估器、统计器、依赖检测器、报告生成器、
运行状态机、数据资产或重复实现公共功能”. The only mutable surface is the candidate mechanism, its dedicated tests
and this evidence directory.

**Training: required.** Issue §3 mandates “一次四卡Stage2 3000epochs/global4096/seed42 … 再执行一次冻结Stage3
150epochs/beam20/seed42/完整57439测试”; the acceptance metrics come from new weights and cannot be recomputed from
existing artefacts.

**Acceptance checklist frozen as C1–C13** (each with criterion, threshold, data hash and accounting):
C1 preregistration/dispatch state · C2 L2 role mixture + L1/L3 gate + switches · C3 frozen training-only role table,
no new graph cache · C4 public Infra byte-identical, mechanism-only edits · C5 dependency-aware validation of the
affected items · C6 0.8 s/epoch ceiling · C7 preregistered L2-prefix / L1-carry / validation-R@10 predictions
unchanged · C8 one formal four-card 3000-epoch run + one frozen Stage3 · C9 existing analysis/replay only ·
C10 wall clock via existing TIMELINE · C11 reports/VERDICT/R2 via existing interfaces · C12 before/after public
hashes + remote check · C13 close standard. Full table: `results/research/issue258/RUN_LOG.md` §4.

**Preregistration is intact, nothing was dispatched:** `results/research/issue256/rounds/r05/STATUS.json` says
`REGISTERED_NOT_DISPATCHED` (“no R05 worker was started, no workspace was reserved and no GPU budget was
consumed”); `results/stage2_RQ-VAE/_runs/` has no `issue256_r05_formal`; no `*role_conditioned_l2*` tree exists.

**Infra baseline** (before any edit): 220 public `.py`/config files hashed into
`results/research/issue258/INFRA_BASELINE.json` (sha256 `48e4fda29314e5895415afa484e06b585adcb24263e0d4b50438f81cb39cc78c`),
covering `tools/iteration/`, `tools/research/stage2/`, `tools/research/verification/`, the three MCP servers,
`tools/graph_infra/`, `tools/issue252/`, `tools/issue253/`, `assets/manifest.json`, `CLAUDE.md` and the skill.
Reused data: `sentence_t5.npy` `6490c717…`, `items.parquet` `6408e3c0…`, `train.parquet` `80597e38…`, role table
`32f986da…` (same bytes as R04). GPUs 0–3 (L40S) idle, ports 50201/50202 free.
