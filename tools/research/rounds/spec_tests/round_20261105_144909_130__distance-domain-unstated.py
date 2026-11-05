"""The distance must state the domain on which it is defined.

Round: round_20261105_144909_130
The reviewer found that MOBIUS(-x, y) is only defined inside the ball while the
specification stated no domain, so an implementation was free to call it on a
point the lift could have pushed onto the boundary. The check refuses a
specification whose distance names no domain, or names one that does not hold
the arguments inside the ball.
"""

FINDING = "the distance states no domain although MOBIUS is only defined inside the ball"
ROUND = "round_20261105_144909_130"
EVAL_ID = ""


def check(spec: dict) -> list:
    problems = []
    computations = {str(entry.get("name")): entry
                    for entry in spec.get("computations") or []}
    distance = computations.get("distance")
    if not distance:
        return ["the specification states no distance computation"]
    domain = str(distance.get("domain") or "").strip()
    if not domain:
        problems.append(
            "the distance states no domain: MOBIUS(-x, y) is only defined inside "
            "the ball, so the specification must say where DIST is defined")
        return problems
    lowered = domain.lower()
    if "ball" not in lowered and "c||x||^2 < 1" not in domain:
        problems.append(
            "the distance's domain never mentions the ball: the arguments must "
            "stay inside it, which is what the lift guarantees")
    if "lift" not in lowered and "lifted" not in lowered:
        problems.append(
            "the domain does not say that the arguments are lifted, which is "
            "what makes it hold by construction")
    return problems
