"""The collision gate is relaxed for exactly the five arms the owner named.

The owner's learned collision gate refuses a SID above 0.35, and it deliberately
overrides the technical-validity-only policy. The owner then asked for the five
arms of issue #139 to be carried into Stage 3 whatever their collision, so that
five mechanisms can be compared end to end. That is a scoped exception, not a
silent bypass: this test pins both halves -- the excepted arms pass, and an
unrelated arm with the same collision is still refused.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "stage2_quality_gate", REPO / "tools" / "genrec-mcp" / "stage2_quality_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_the_excepted_arms_are_admitted() -> str:
    candidate = {"l3_collision": 0.40}
    for arm in gate.ISSUE139_STAGE3_EXCEPTION:
        refusal = gate.collision_gate(candidate, learned=0.40, exported_raw=0.40, arm=arm)
        if refusal is not None:
            raise AssertionError(arm + " is in the exception list but was refused: " + refusal)
    return "all five excepted arms are admitted"


def test_any_other_arm_is_still_refused() -> str:
    candidate = {"l3_collision": 0.40}
    refusal = gate.collision_gate(candidate, learned=0.40, exported_raw=0.40,
                                  arm="TIGER_some-other-arm")
    if refusal is None:
        raise AssertionError("an unrelated arm with a 0.40 collision was admitted")
    if "0.40" not in refusal:
        raise AssertionError("the refusal does not report the measured collision: " + refusal)
    return "an unrelated arm is still refused at the gate"


def test_the_default_is_unchanged() -> str:
    candidate = {"l3_collision": 0.40}
    if gate.collision_gate(candidate, learned=0.40, exported_raw=0.40) is None:
        raise AssertionError("the gate admits a 0.40 collision without an arm name")
    ok = gate.collision_gate({"l3_collision": 0.10}, learned=0.10, exported_raw=0.10)
    if ok is not None:
        raise AssertionError("the gate refuses a 0.10 collision: " + str(ok))
    return "without an excepted arm name the gate behaves as before"


def main() -> int:
    failures = 0
    for test in (test_the_excepted_arms_are_admitted, test_any_other_arm_is_still_refused,
                 test_the_default_is_unchanged):
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
