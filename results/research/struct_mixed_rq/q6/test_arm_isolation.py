"""The fix tree must not derive the old experiment's arm identity."""
import re
import sys
from pathlib import Path

FIX = Path(__file__).resolve().parents[1]
OLD = FIX.parent / "TIGER_struct_mixed_rq"


def arm_name(path: Path) -> str:
    text = path.read_text()
    match = re.search(r'^ARM_NAME\s*=\s*"([^"]+)"', text, re.M)
    assert match, f"no ARM_NAME in {path}"
    return match.group(1)


def test_the_fix_tree_has_its_own_arm_name():
    assert arm_name(FIX / "train_rqvae.py") != arm_name(OLD / "train_rqvae.py")
    assert arm_name(FIX / "train_rqvae.py") == "TIGER_struct_mixed_rq_ste_fix"


if __name__ == "__main__":
    test_the_fix_tree_has_its_own_arm_name()
    print("ok")
