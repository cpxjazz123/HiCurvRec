### #260 round 1 frozen: diagnosis before R06

**Issue body**: 9255 bytes, sha256 `69fed718a05514f752d06bfe604f3f44f12722244a85866f8745ce1d2a65f4a0`.

**INFRA_MODE = FROZEN**, quoting the issue: “禁止新增或修改公共 Infra、数据资产、Skill、公共训练/评估规则。
仅研究机制与专属测试允许改变”。No public tool/asset/Skill/CLAUDE/launcher/evaluator change.

**Training decision**: the unique close condition is one complete Stage3 test Recall@10 strictly > 0.065.
The issue’s first round is diagnosis-first; no Stage2 starts before a preregistered R06 contract with
root-cause evidence. After that, one formal candidate runs the frozen Stage2/Stage3 protocols.

**Frozen C1–C10** (full text in `results/research/issue260/RUN_LOG.md` §4): five-segment diagnosis with
≥2 competing hypotheses incl. null; `DIAGNOSIS.md` + inherited `HYPOTHESES.json`; one main mechanism with
CONTRACT and six answers; TDD; local pre-training check; legal Stage2 3000/4-card/seed42/collision<0.40;
Stage3 150/beam20 with one final test; replay/report/archive/push/comment; close only on strict >0.065;
no new Infra/assets/TIGER retrain/seed sweep/test-driven selection.

**Reused identity**: R05 checkpoint `f93da1a7…`, R05 SID `dc206dba…`, baseline TIGER SID `025ad8bd…`,
train/valid/test and embedding hashes in RUN_LOG §6. Start Infra snapshot recorded (`INFRA_BASELINE.json`,
232 protected files).

Next: finish the read-only R05 checkpoint interventions, write `DIAGNOSIS.md`, then choose exactly one
mechanism and freeze its contract.
