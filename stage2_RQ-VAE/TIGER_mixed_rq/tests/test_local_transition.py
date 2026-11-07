"""A stale local entry is deleted, not skipped, and the table converges.

The reviewed code refused a round whose local table carried a prefix that fell
below the activation rule, because skipping the entry would leave every item
under it with a weight its support no longer justifies. Deleting it makes
AlphaTable.get inherit the nearest ancestor, which is the documented rule for a
prefix without an entry, so the reviewed transition is: delete, re-encode,
repeat until nothing is stale, and judge the round against the state that
results. This test pins that behaviour, including the fabricated prefix that no
item carries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.af3_engine import StructuralState  # noqa: E402
from model.af3_selector import MIN_PREFIX_ITEMS, prefix_groups, resolve_local_table  # noqa: E402
from model.mixed_rq import AlphaTable, MixedRQ, encode_tokens  # noqa: E402


def _fixture():
    torch.manual_seed(0)
    n, codes = 4000, 6
    rq = MixedRQ([codes] * 3, AlphaTable())
    z = torch.randn(n, 32)
    rq.init_codebooks(z, torch.device("cpu"))
    rng = np.random.default_rng(0)
    pairs = set()
    while len(pairs) < 8000:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    pairs = np.array(sorted(pairs))
    is_edge = rng.random(len(pairs)) < 0.6
    kappa = np.where(is_edge, rng.normal(-20, 8, len(pairs)), 0.0)
    state = StructuralState(pairs, is_edge, kappa, n, tau=0.3, intercept=-0.5, scale=8.0)
    tokens = encode_tokens(z, rq)
    state.full(torch.cat([z[:, :16], z[:, 16:]], dim=-1))
    return rq, z, state, tokens


def test_a_stale_entry_is_deleted_and_the_table_converges() -> str:
    rq, z, state, tokens = _fixture()
    groups = prefix_groups(tokens, 1)
    supported = [prefix for prefix, items in groups.items() if len(items) >= MIN_PREFIX_ITEMS]
    if not supported:
        raise AssertionError("the fixture produced no supported prefix to keep")
    rq.alpha_table.set(2, supported[0], 0.75)
    stale = (10 ** 6,)
    rq.alpha_table.set(2, stale, 0.25)
    outcome = resolve_local_table(
        rq.alpha_table, z, rq, state, tokens,
        encode=lambda: encode_tokens(z, rq),
        readout=lambda current: torch.cat([z[:, :16], z[:, 16:]], dim=-1))
    if outcome["status"] != "converged":
        raise AssertionError("the transition did not converge: " + str(outcome))
    if not any(d["level"] == 2 and tuple(d["prefix"]) == stale for d in outcome["deleted"]):
        raise AssertionError("the stale entry was not reported as deleted: " + str(outcome))
    if stale in rq.alpha_table.levels[2]:
        raise AssertionError("the stale entry is still in the table")
    if supported[0] not in rq.alpha_table.levels[2]:
        raise AssertionError("a supported entry was deleted as well")
    return "the stale entry was deleted, the supported one kept, the table converged"


def test_a_clean_table_is_left_alone() -> str:
    rq, z, state, tokens = _fixture()
    groups = prefix_groups(tokens, 1)
    supported = [prefix for prefix, items in groups.items() if len(items) >= MIN_PREFIX_ITEMS][:1]
    for prefix in supported:
        rq.alpha_table.set(2, prefix, 0.75)
    before = {level: dict(entries) for level, entries in rq.alpha_table.levels.items()}
    outcome = resolve_local_table(
        rq.alpha_table, z, rq, state, tokens,
        encode=lambda: encode_tokens(z, rq),
        readout=lambda current: torch.cat([z[:, :16], z[:, 16:]], dim=-1))
    if outcome["deleted"]:
        raise AssertionError("a clean table lost entries: " + str(outcome["deleted"]))
    if outcome["passes"] != 0:
        raise AssertionError("a clean table needed passes: " + str(outcome))
    if before != {level: dict(entries) for level, entries in rq.alpha_table.levels.items()}:
        raise AssertionError("a clean table was modified")
    return "a clean table is untouched and needs no pass"


def main() -> int:
    failures = 0
    for test in (test_a_stale_entry_is_deleted_and_the_table_converges,
                 test_a_clean_table_is_left_alone):
        try:
            print("PASS " + test.__name__ + ": " + test())
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
