# MVG -- the Mechanism Verification Gate

The gate that stands between a Stage-2 mechanism and an expensive cold run. It is
**independent** of the mechanism: the reference implementation it compares against never
imports the code under test, and the specification it checks against is frozen before the
implementation exists (issue #101 §A/§B).

```
tools/mvg/
  spec.py                 the MechanismSpec parser, validator and digest
  schemas/                the normative JSON Schema of a MechanismSpec
  specs/                  the frozen specifications (one per mechanism)
  reference/              the independent reference implementations (the oracles)
  gates.py                the gates M0..M4
  mutants.py              the named defects and the gate that owns each of them
  runner.py               runs the gates, writes mvg_report.json, runs M5
  cli.py                  the command line
  tests/run_mvg_tests.py  the acceptance runner
```

## Reproducible commands

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec

# the gates over the reference mechanism, with the mutation suite
PYTHONPATH=. /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
    -m tools.mvg.cli run \
      --tree stage2_RQ-VAE/TIGER_r9_mvg_reference \
      --spec tools/mvg/specs/masked_directed_codebook_margin.json \
      --out /tmp/mvg_run

# the acceptance runner (the project's 3.9 environment has torch but no pytest,
# so the repository's convention of a standalone script applies)
PYTHONPATH=. /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
    tools/mvg/tests/run_mvg_tests.py
```

Output paths: `mvg_report.json` in `--out`; the gates write nothing else.

## The gates

| gate | question it answers | state vocabulary |
|---|---|---|
| **M0** contract and static audit | does the implementation agree with the frozen specification about what it is, and does it contain a forbidden substitution? | passed / failed / blocked |
| **M1** independent numerical reference | does it compute the declared objective, forward value *and* intermediates, where an independent implementation computes the same numbers? | |
| **M2** derivative correctness | is the gradient the derivative of the declared loss, does it reach the declared targets, does it survive the degenerate cases? | |
| **M3** metamorphic and semantic | do the declared properties hold (direction matters, permutations do not, a masked pair contributes nothing)? | |
| **M4** on/off identity and integration | is the mechanism-off tree the frozen host, and does the mechanism-on tree move the parameters the export reads? | |
| **M5** test quality | does every named defect get rejected by the gate that owns it? | |

`blocked` means the check could not be run -- a missing resource, an unavailable device. It
is never a pass, and a report with a blocked gate has the verdict `failed`.

## The reference mechanism

`stage2_RQ-VAE/TIGER_r9_mvg_reference/` hosts `masked_directed_codebook_margin`: a masked
directed margin between the source's and the target's nearest codebook entry, plus a
relational term that compares the two codebook assignments through the codebook's own
geometry. It exists to give the gates something faithful to check.

Its specification (`tools/mvg/specs/masked_directed_codebook_margin.json`) carries the
equation, the symbols, the parameters, the forbidden substitutions, the invariants, the
synthetic cases (positive, negative, boundary, degenerate), the tolerances per dtype, the
CGV seven-group contract and the provenance. A change to the specification is a new
specification version and invalidates a prior admission.

## The named defects

`tools/mvg/mutants.py` holds the seven defects the issue names. Each is injected into a
**scratch copy** of the tree by a source replacement, and the gate named in the entry has
to reject it. The mutants are never applied to the repository's tree and never committed.

| mutant | the gate that must reject it | why |
|---|---|---|
| `gw_to_mse` | M1 | a relational discrepancy is not a per-element distance error |
| `transpose_pc_cc` | M1 | the assignment's axis is the codebook axis, not the pair axis |
| `latent_only_zero_codebook_gradient` | M2 | the codebook gradient is declared nonzero |
| `no_op_loss` | M1 | a term that contributes nothing is not a mechanism |
| `symmetric_for_directed` | M1 | the mechanism is directed, not a similarity |
| `detached_gradient` | M2 | a detached loss trains nothing |
| `wrong_masked_ddp_normalization` | M1 | the denominator is the number of valid pairs |

## Budget and limitations

* The gates run on CPU. A full run -- five gates plus seven mutants, each mutant running
  the gates again -- takes about **10 seconds** warm and about **4 minutes** on a cold
  filesystem (the first `import torch` dominates).
* The gates do not train anything. M4 executes the real model, the real host terms and the
  real mechanism term on a tiny case, and one optimizer step, which is what proves the
  mechanism reaches the parameters the export reads; it is not a 3000-epoch run.
* M1 compares against an independent implementation, not against a proof. A defect that is
  invisible to the reference on the declared cases is not excluded, and the report says so
  in `residual_uncertainty`.
* M2 needs a differentiable evaluation point: a case whose nearest codebook entry is unique
  for every pair. The gate refuses to measure a derivative at a tie and says which case to
  add.
* The specification's forbidden substitutions are **static**: they catch the defect in the
  source. A substitution that is invisible in the source is caught by M1 and M2, not by M0.
