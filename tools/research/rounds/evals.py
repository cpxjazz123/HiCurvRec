"""The eval gate: a caught defect becomes a check no later round can repeat.

The loop is the one OpenAI's agent-improvement cookbook describes, applied to
this repository's own evidence:

* a **trace** is what the round already keeps - ``agents/<task>/progress.jsonl``
  and ``execution.json`` for every agent call;
* the **feedback** is the Math Review, whose verdict and findings are persisted
  in ``state.json`` and in the frozen specification;
* an **eval** is this module: a deterministic check that fails on the defect the
  reviewer described, written once and run at the freeze from then on;
* the **gate** is ``phase_formulation``: a specification that trips a check is
  not frozen, whatever the two reviewers happen to say that round;
* the **regression set** is ``tools/research/tests/test_spec_evals.py``: every
  eval is proven to catch its seeded defect, and to leave the frozen
  specification of the round that produced it alone.

Two kinds of knowledge are kept apart, exactly as the cookbook argues. A defect
that is mechanically decidable (a dropped clamp, a 0/0 described as zero, a
per-batch mean described as an epoch mean) becomes an eval here. A question
that is not decidable - "does this curvature supervision collapse level 3?" -
stays research evidence for the Idea Agents, and is never turned into a gate.

An eval may only enter this list with a proven provenance:

1. it fails on the defect it was seeded from (``SEEDED`` in the tests);
2. it passes the frozen specification of the round that produced it, so the
   gate can never retroactively invalidate the round's own authority.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: The interpreter that has torch: the numeric evals execute the specification's
#: own statements, and a specification is arithmetic, not prose.
TORCH_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: Names the surface may use without declaring them.
BUILTINS = {
    "torch", "math", "float", "int", "len", "range", "min", "max", "sum",
    "abs", "print", "True", "False", "None", "def", "return", "if", "else",
    "for", "in", "and", "or", "not", "lambda", "shape", "dim", "unsqueeze",
    "sqrt", "tanh", "atanh", "log1p", "clamp", "where", "zeros", "zeros_like",
    "argmin", "softmax", "sum", "keepdim", "values", "Tensor",
}


@dataclass(frozen=True)
class Eval:
    """One mechanically decidable requirement, with the defect it came from."""

    id: str
    title: str
    source: str
    check: object  # callable[[dict], list[str]]


def _statements(spec: dict) -> list:
    """The SymPy statements of the frozen surface."""
    surface = spec.get("continuous_surface") or {}
    return [str(item) for item in surface.get("statements") or []]


def _executable(spec: dict) -> list:
    """The statements a program can run, when the surface carries them.

    The frozen surface states its mathematics as SymPy, because the reference
    contract must equal it statement by statement. The executable form is kept
    beside it, and the numeric eval runs THAT: the identities are properties of
    the arithmetic, not of a symbolic spelling.
    """
    surface = spec.get("continuous_surface") or {}
    executable = surface.get("executable_statements") or []
    return [str(item) for item in executable]


def _surface_text(spec: dict) -> str:
    surface = spec.get("continuous_surface") or {}
    parts = list(_executable(spec)) or list(_statements(spec))
    symbols = surface.get("symbols") or {}
    if isinstance(symbols, dict):
        parts.extend(str(value) for value in symbols.values())
    else:
        parts.extend(str(value) for value in symbols)
    return "\n".join(parts)


def _code(spec: dict) -> list:
    """The statements a program can run, falling back to the SymPy ones."""
    return _executable(spec) or _statements(spec)


def _section(spec: dict, name: str) -> str:
    value = spec.get(name)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(str(item) for item in value.values())
    return str(value or "")


def _full_text(spec: dict) -> str:
    return "\n".join([_surface_text(spec), _section(spec, "switches"),
                      _section(spec, "mechanism"), _section(spec, "parameters"),
                      _section(spec, "excluded")])


def _computation(spec: dict, name: str) -> str:
    for entry in spec.get("computations") or []:
        if str(entry.get("name")) == name:
            return "\n".join(str(entry.get(key) or "")
                             for key in ("definition", "note", "domain", "output"))
    return ""


def _distance_clamp(spec: dict) -> list:
    """The idea's artanh clamp, kept verbatim. A dropped clamp is a NaN at the
    ball boundary, which is reachable as tanh -> 1 rounds onto it."""
    problems = []
    line = next((item for item in _code(spec) if "atanh" in item), "")
    if line and not re.search(r"clamp|min\(", line):
        problems.append(
            "the distance computes atanh without a clamp on its argument: the "
            "idea's min(., 1 - 1e-6) guard is what keeps a latent that rounds "
            "onto the ball boundary from producing inf/NaN")
    if not line and "atanh" not in _full_text(spec):
        problems.append("no distance statement computes atanh at all")
    return problems


def _lift_zero(spec: dict) -> list:
    """LIFT(0) = 0 must hold by arithmetic, not by an unreachable branch."""
    problems = []
    line = next((item for item in _code(spec) if "def lift" in item), "")
    if not line:
        return ["no lift statement is given"]
    if "where" in line and "_norm(v)" in line.split("where", 1)[1][:60]:
        problems.append(
            "the lift's zero branch is masked on _norm(v), which carries an eps "
            "and is therefore never 0: the branch is dead. Guard the squared "
            "norm, or drop the branch and let the numerator vanish")
    return problems


def _curvature_zero_branch(spec: dict) -> list:
    """The geometry switch must state its Euclidean arm, not allude to it."""
    problems = []
    text = _section(spec, "switches")
    if not re.search(r"c\s*=\s*0", text):
        problems.append("the switches never state what c = 0 does")
    if not re.search(r"2\s*\|\|", text):
        problems.append("the switches never state the Euclidean distance 2||x - y||")
    return problems


def _behaviour_off_arithmetic(spec: dict) -> list:
    """b_k = 0 under BEHAVIOUR_OFF is arithmetically false: with m_i = 1 the
    demand is a constant, b_k = log 2, and the offset is k-independent."""
    problems = []
    text = _section(spec, "switches")
    if re.search(r"b_k\s*=\s*0", text):
        problems.append(
            "the behaviour-off arm claims b_k = 0: with m_i = 1 the demand is a "
            "constant, so b_k = log 2 and log(1 + b_k) = 0.5267, not 0. The arm "
            "must be justified by the offset being k-independent")
    if not re.search(r"k-independent|codeword-independent|constant across|"
                     r"the same for every codeword|independent of the codeword", text):
        problems.append(
            "the behaviour-off arm does not say that an offset which is the same "
            "at every codeword cannot change the argmin, which is the only "
            "correct justification")
    return problems


def _behaviour_chain(spec: dict) -> list:
    """The listing must carry a behaviour chain: a per-item value and a per-code term.

    Which names the chain uses is the mechanism's business; what the framework
    requires is that the behaviour is COMPUTED (a per-item quantity), that it
    meets the geometry in the cost (a subtraction or addition of a term that
    depends on both the item's value and the code), and that the assignment is
    the baseline's balanced one.
    """
    problems = []
    code = "\n".join(_code(spec))
    if not code:
        return problems
    lowered = code.lower()
    has_item_value = bool(re.search(r"rho_i|demand_i|behaviour|cohesion|coh_", lowered))
    has_per_code = bool(re.search(r"s_k|off_k|stratum|demand_k|b_k", lowered))
    has_cost = bool(re.search(r"c_ik\s*=", lowered))
    if not has_item_value:
        problems.append("the listing computes no per-item behaviour quantity: the "
                        "behaviour side of the coupling is missing")
    if not has_per_code:
        problems.append("the listing forms no per-codeword term: the behaviour "
                        "cannot meet the geometry in the cost")
    if not has_cost:
        problems.append("the listing has no cost matrix that carries both sides: "
                        "the coupling is not in one expression")
    if not re.search(r"sinkhorn", lowered):
        problems.append("the listing does not use the baseline's balanced "
                        "assignment: the mechanism may change the cost, never the "
                        "assignment rule (CLAUDE.md 19h)")
    return problems


def _epoch_one_weight(spec: dict) -> list:
    """w needs a mean_scale that only exists after epoch 1."""
    problems = []
    text = _computation(spec, "offset weight")
    if text and not re.search(r"epoch 1", text, re.IGNORECASE):
        problems.append(
            "the offset weight never says what w is during epoch 1, when "
            "mean_scale does not exist yet: the mechanism would be undefined "
            "for the first epoch it runs in")
    return problems


def _demand_surface(spec: dict) -> list:
    """The demand is measured on the distance, not on the offset-adjusted score."""
    problems = []
    text = _computation(spec, "behaviour demand") + _surface_text(spec)
    if "p_ik" in text and not re.search(r"pure geodesic distance|not score_k|distanc", text):
        problems.append(
            "the specification does not say whether the soft assignment p_ik "
            "uses the distance or the offset-adjusted score; the two give "
            "different demands and the implementation would have to guess")
    return problems


def _scale_definition(spec: dict) -> list:
    """s is the RMS of the per-item vector norm, and it must be fail-closed."""
    problems = []
    text = _computation(spec, "scale")
    if text and not re.search(r"vector norm|per-item latent", text):
        problems.append(
            "the scale s does not say that it is the RMS of the per-item latent "
            "VECTOR norm: a per-coordinate RMS is different by sqrt(D) and "
            "saturates the lift")
    if text and not re.search(r"s\s*<=\s*0|not finite", text):
        problems.append(
            "the scale has no guard: a zero or non-finite s makes the curvature "
            "diverge, and a silent default would change the mechanism")
    return problems


def _reconstruction_unlifted(spec: dict) -> list:
    """The residual recursion uses the codebook vector, not its lift."""
    problems = []
    integration = spec.get("training_integration") or {}
    reconstruction = str(integration.get("reconstruction") or "")
    if not reconstruction:
        problems.append("the training integration states no reconstruction")
    elif not re.search(r"unlifted|not lifted", reconstruction, re.IGNORECASE):
        problems.append(
            "the reconstruction does not state that it uses the UNLIFTED "
            "codeword: lifting it moves the reconstruction point and the "
            "residual recursion, which the mechanism does not own")
    return problems


def _code_identity_not_input_identity(spec: dict) -> list:
    """Levels 2 and 3 share the baseline's CODE, not its inputs: the level-1
    choice changed, so their inputs changed with it."""
    problems = []
    text = _section(spec, "mechanism") + "\n" + _computation(spec, "residual recursion")
    if re.search(r"levels?\s*2\s*(and|&)\s*3\s+see\s+exactly\s+the\s+baseline",
                 text, re.IGNORECASE):
        problems.append(
            "the specification says levels 2 and 3 SEE the baseline's inputs: "
            "their code is identical, but their inputs follow the level-1 "
            "choice, which the mechanism changes")
    return problems


def _demand_buffer_not_frozen(spec: dict) -> list:
    """b_raw is a buffer updated per epoch, not a frozen parameter."""
    problems = []
    text = _section(spec, "parameters") + "\n" + _computation(spec, "behaviour demand")
    if re.search(r"frozen demand|frozen b_raw", text, re.IGNORECASE) and not re.search(
            r"not frozen|NOT frozen", text):
        problems.append(
            "the parameters call b_raw a frozen demand: it is a detached buffer "
            "updated once per epoch, and an implementation that freezes it would "
            "never see the measurement")
    return problems


def _rho_label(spec: dict) -> list:
    """rho is a ratio of lengths, so its label must not invert the scale."""
    problems = []
    text = _computation(spec, "scale") + "\n" + _section(spec, "parameters")
    if re.search(r"radius-to-scale", text, re.IGNORECASE) and not re.search(
            r"s/R_c|R_c = s/rho", text):
        problems.append(
            "rho is called a radius-to-scale ratio while R_c = s/rho: the label "
            "inverts the ratio, and R_c is the radius, s the scale")
    return problems


def _surface_defines_what_it_uses(spec: dict) -> list:
    """Every name the listing uses at the top level must be assigned in it.

    A listing that reads a name it never defines is not a program: an
    implementation that follows it raises, and one that guesses the name is not
    implementing the specification. The names the surface declares as its
    inputs (the residual, the codebook, the item mass, the carried buffers and
    the two switches) are known; the bodies of the helper functions are not
    part of the listing's own assignments.
    """
    import builtins
    import re

    code = _code(spec)
    if not code:
        return []
    import keyword
    known = (set(dir(builtins)) | set(keyword.kwlist)
             | {"torch", "math", "True", "False", "None"})
    for line in code:
        stripped = line.strip()
        if stripped.startswith("def "):
            head = stripped.split("def ", 1)[1]
            known.add(head.split("(")[0].strip())
            arguments = head.split("(", 1)[1].split(")")[0] if "(" in head else ""
            for argument in arguments.split(","):
                name = argument.split("=")[0].strip()
                if name.isidentifier():
                    known.add(name)
    # the surface's own inputs
    # The mechanism's inputs: the residual and the codebook, the frozen scale and
    # curvature, the behaviour buffer and the per-code stratum, and the two
    # switches. A listing that ASSIGNS any of them shadows the input, which the
    # assignment scan above already handles.
    known |= {"r", "a", "K1", "B", "D", "N", "s", "c", "R_c", "rho", "epoch",
              "BEHAVIOUR_OFF", "GEOMETRY_OFF", "m_i", "b_raw", "ms_sum", "ms_count",
              "rho_i", "s_k", "beta", "tau", "sk_iters", "sk_epsilon", "K", "iters",
              "epsilon", "BASELINE_SINKHORN", "a_k", "alpha_k", "K1", "B",
              "center", "centre"}
    for line in code:
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("def "):
            continue
        if line[:1].isspace():
            continue                      # inside a helper function
        if "=" in stripped:
            name = stripped.split("=", 1)[0].strip()
            if name.isidentifier():
                known.add(name)
    problems = []
    for line in code:
        stripped = line.split("#", 1)[0]
        if line[:1].isspace() or not stripped.strip():
            continue
        if "=" in stripped:
            stripped = stripped.split("=", 1)[1]
        # the attribute names (torch.argmax, .sum, ...) belong to the object
        # they are called on, not to the listing's own names
        stripped = re.sub(r"\.[A-Za-z_][A-Za-z0-9_]*", " ", stripped)
        # a keyword argument (keepdim=True) is a name of the called function
        stripped = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)", " ", stripped)
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", stripped):
            if name not in known:
                problems.append("the surface uses %r without ever assigning it: the "
                                "listing is not a program an implementation can follow"
                                % name)
    return sorted(set(problems))


def _surface_assigns_the_demand(spec: dict) -> list:
    """The demand's chain must be in the listing: epoch, blend, normalisation."""
    problems = []
    code = "\n".join(_code(spec))
    for name in ("b_raw_epoch", "b_raw", "b_k"):
        if not any(re.match(r"%s[_A-Za-z0-9]*\s*=" % name, line.strip())
                   for line in _code(spec)):
            problems.append("the surface never assigns %s, so the offset and the "
                            "weight cannot be computed from it" % name)
    return problems


def _surface_applies_the_lift(spec: dict) -> list:
    """The lift is applied to the residual and to the codebook, separately."""
    problems = []
    code = "\n".join(_code(spec))
    if "lift(" not in code:
        return ["the surface never applies the lift"]
    if "u = lift(r" not in code.replace(" ", " ") and "lift(r," not in code:
        problems.append("the surface never lifts the residual")
    if "lift(a," not in code and "lift(a," not in code.replace(" ", ""):
        problems.append("the surface never lifts the codebook")
    return problems


def _surface_couples_the_behaviour(spec: dict) -> list:
    """The behaviour half of the coupling must be in the listing."""
    problems = []
    code = "\n".join(_code(spec))
    for name, why in (("p_ik", "the soft assignment of the demand"),
                      ("num_k", "the epoch's accumulation of the demand"),
                      ("off_k", "the offset the behaviour contributes")):
        if name not in code:
            problems.append("the surface omits %s: %s" % (name, why))
    if "score_k" in code and "off_k[None" not in code and "off[None" not in code:
        problems.append("the surface's score does not subtract the offset")
    return problems


def _symbolic_distance_keeps_the_guards(spec: dict) -> list:
    """The SymPy surface is the idea's expression: eps in the denominator and
    the clamp on the artanh argument."""
    problems = []
    statements = _statements(spec)
    if not statements:
        return []
    denominator = next((line for line in statements if line.strip().startswith("den")),
                       "")
    if denominator and "eps" not in denominator and "1e-12" not in denominator:
        problems.append("the symbolic Mobius denominator omits the eps the idea's "
                        "expression carries")
    distance = next((line for line in statements if "atanh" in line), "")
    if distance and not re.search(r"min\(|Min\(", distance):
        problems.append("the symbolic distance omits the idea's min(., 1 - 1e-6) "
                        "clamp on the artanh argument")
    return problems


def _surface_interface_is_consistent(spec: dict) -> list:
    """The callable, the arguments, the symbols and the output must agree.

    A surface that names one callable and declares another's arguments is not a
    surface: the reference contract would compare a distance against a score,
    and the implementation would have to guess which one the specification means.
    """
    problems = []
    surface = spec.get("continuous_surface") or {}
    callable_name = str(surface.get("callable") or "")
    args = surface.get("args") or []
    arg_names = [str(entry.get("name")) for entry in args
                 if isinstance(entry, dict) and entry.get("name")]
    if callable_name and args:
        head = callable_name.split("(", 1)[0].strip()
        inside = callable_name.split("(", 1)[1].rsplit(")", 1)[0] if "(" in callable_name else ""
        parameters = [item.strip().split("=")[0].strip()
                      for item in inside.split(",") if item.strip()]
        if parameters and parameters != arg_names:
            problems.append(
                "the surface names the callable %r but declares the arguments %s: "
                "the two must be the same surface" % (head, arg_names))
    statements = _statements(spec)
    output = str(surface.get("output") or "")
    if output and statements:
        assigned = {line.split("=", 1)[0].strip() for line in statements
                    if "=" in line}
        if output not in assigned:
            problems.append("the surface's output %r is not one of its statements"
                            % output)
    symbols = surface.get("symbols") or []
    if isinstance(symbols, list) and args:
        consumed = 0
        for entry in args:
            if not isinstance(entry, dict):
                continue
            if entry.get("scalar") is not None:
                consumed += 1
            else:
                consumed += int(entry.get("width") or 1)
        if len(symbols) != consumed:
            problems.append(
                "the surface declares %d symbols for %d argument columns: the "
                "reference rebuild would read the wrong ones" % (len(symbols), consumed))
    return problems


def _surface_uses_its_arguments(spec: dict) -> list:
    """The executable listing must use the arguments the surface declares.

    A listing that computes the curvature from s while the surface declares a
    curvature argument makes the contract compare two different surfaces: the
    one the implementation runs and the one the reference rebuilds.
    """
    problems = []
    surface = spec.get("continuous_surface") or {}
    args = surface.get("args") or []
    names = [str(entry.get("name")) for entry in args
             if isinstance(entry, dict) and entry.get("name")]
    code = "\n".join(_code(spec))
    if not names or not code:
        return []
    for name in names:
        if name == "curvature":
            continue
        if not re.search(r"\b%s\b" % re.escape(name), code):
            problems.append("the surface declares the argument %r but the listing "
                            "never uses it" % name)
    return problems


def _assignment_keeps_the_balancing(spec: dict) -> list:
    """The level index may not be taken by a plain argmin over the score.

    The baseline's level assignment is Sinkhorn-balanced, and the balancing is
    what keeps the codebook's usage even. A specification that replaces it with
    argmin_k(score_k) collapses the level-1 codebook: the round that measured it
    used 42 of 256 level-1 codes against the reference arm's 241, the largest
    code took 31% of the items, and the three-level collision reached 0.703
    against the reference's 0.065.
    """
    problems = []
    code = "\n".join(_code(spec))
    if not code:
        return problems
    balanced = bool(re.search(r"sinkhorn|balanced", code, re.IGNORECASE))
    # an argmin over the mechanism's score, taken as the level index
    if re.search(r"=\s*torch\.argmin\s*\(\s*score", code):
        if not balanced:
            problems.append(
                "the level index is taken by a plain argmin over the score: the "
                "baseline's Sinkhorn-balanced assignment has been replaced, and "
                "the balancing is what keeps the codebook's usage even. The "
                "mechanism's score must be the cost matrix of the balanced "
                "assignment, not a replacement for it")
    if not balanced:
        text = _section(spec, "computations")
        if re.search(r"k\s*_?\s*1?\s*=\s*arg\s*min", text, re.IGNORECASE):
            problems.append(
                "the specification assigns the level index by argmin and never "
                "mentions the baseline's balanced assignment: the balancing must "
                "stay")
    return problems


def _geometry_arm_keeps_the_measured_scale(spec: dict) -> list:
    """The Euclidean arm must differ from the curved one ONLY in the distance.

    The offset's amplitude beta is measured once, in the curved configuration. If
    the c = 0 arm re-measured beta from its own (Euclidean) distances, the two
    arms would differ in the offset's amplitude as well, and the control would
    not be a matched one.
    """
    problems = []
    text = _section(spec, "switches")
    if not text:
        return problems
    for match in re.finditer(r"GEOMETRY_OFF[^\n]*", text, re.IGNORECASE):
        arm = match.group(0)
        if re.search(r"re-measured|remeasured|measured again|its own median", arm,
                     re.IGNORECASE):
            problems.append(
                "the Euclidean arm re-measures the offset's amplitude, so it "
                "differs from the curved arm in the amplitude as well as in the "
                "distance: the frozen beta of the curved configuration has to be "
                "carried into it")
    cost = _computation(spec, "cost matrix")
    if re.search(r"beta", cost) and not re.search(r"frozen|measured once", cost,
                                                 re.IGNORECASE):
        problems.append("the offset's amplitude is not stated as measured once and "
                        "frozen")
    return problems


def _offset_scale_is_one_global_quantile(spec: dict) -> list:
    """The offset's amplitude must be ONE quantile of the pooled batch.

    A mean of the ranks' medians is not a median of anything: with a skewed
    distance distribution the two differ, and the amplitude is what sets the
    offset's range.
    """
    problems = []
    cost = _computation(spec, "cost matrix")
    if not cost:
        return problems
    if re.search(r"mean of (the )?(per-)?rank", cost, re.IGNORECASE) or \
            re.search(r"medians are averaged", cost, re.IGNORECASE):
        problems.append("the offset's amplitude is the mean of the ranks' medians, "
                        "which is not a quantile of anything: it must be the median "
                        "of the POOLED step-0 distances")
    if not re.search(r"pooled|pool ", cost, re.IGNORECASE):
        problems.append("the offset's amplitude does not say it is measured over the "
                        "pooled step-0 batch")
    return problems


def _the_ordering_is_not_claimed_as_maintained(spec: dict) -> list:
    """The strata are free parameters: their order is an initialisation."""
    problems = []
    text = _section(spec, "mechanism") + "\n" + _computation(spec, "code stratum")
    if re.search(r"codewords are ORDERED", text) and not re.search(
            r"initialis|start ordered|INITIALIS", text):
        problems.append("the mechanism claims the codewords are ORDERED by the rank, "
                        "but the strata are free parameters whose order nothing "
                        "maintains: the claim must be an initialisation")
    return problems


def _declared_intervals_match_the_definitions(spec: dict) -> list:
    """A declared range must be the range the definition gives for THAT symbol.

    The check is about one symbol: a quantity the definition computes with a
    sigmoid lies in the open interval, so an output that declares it closed
    contradicts its own definition. Intervals that belong to other quantities are
    none of this check's business.
    """
    problems = []
    for entry in spec.get("computations") or []:
        output = str(entry.get("output") or "")
        definition = str(entry.get("definition") or "")
        for symbol in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s+in\s+\[", output):
            if re.search(r"%s\s*=\s*sigmoid\(" % re.escape(symbol), definition):
                problems.append(
                    "the computation %r declares %s in a closed interval while its "
                    "definition computes it with a sigmoid, whose range is open"
                    % (entry.get("name"), symbol))
    return problems


def _no_unattested_fallback_is_attributed_to_the_idea(spec: dict) -> list:
    """A fallback the idea does not contain must not be attributed to it."""
    problems = []
    text = "\n".join(str(entry.get("definition") or "") + "\n" +
                      str(entry.get("note") or "")
                      for entry in spec.get("computations") or [])
    text += "\n" + _section(spec, "gradient_rules")
    if re.search(r"idea(?:'s)? (?:documented|recorded|own) fallback", text,
                 re.IGNORECASE):
        problems.append("the specification attributes a fallback to the idea that the "
                        "idea does not contain: a rule the idea does not state must "
                        "not be presented as the idea's")
    return problems


def _the_geometry_note_states_properties(spec: dict) -> list:
    """The note about the sign must state the properties, not a counterfactual."""
    problems = []
    for entry in spec.get("computations") or []:
        if entry.get("name") != "geometry":
            continue
        note = str(entry.get("note") or "")
        if re.search(r"with the \+ sign|with \+2c|would return|returns 0 at", note,
                     re.IGNORECASE):
            problems.append("the geometry note argues the sign with a counterfactual "
                            "instead of stating the properties the correct form has "
                            "(d_c(x, x) = 0 and the c -> 0 limit 2||x - y||)")
    return problems


def _the_surface_does_not_bind_its_own_temperature(spec: dict) -> list:
    """The plan's temperature is the baseline's; the listing must not add one."""
    problems = []
    code = "\n".join(_code(spec))
    if re.search(r"^\s*tau\s*=\s*[0-9]", code, re.MULTILINE) and \
            "sinkhorn" in code.lower():
        problems.append("the listing binds its own temperature while the plan is the "
                        "baseline's sinkhorn, which exponentiates the cost with the "
                        "baseline's epsilon: a second temperature changes the plan")
    return problems


def _the_mechanisms_list_agrees_with_its_computations(spec: dict) -> list:
    """The mechanism may not claim a change the computations deny."""
    problems = []
    mechanism = _section(spec, "mechanism")
    computations = "\n".join(str(entry.get("definition") or "")
                              for entry in spec.get("computations") or [])
    if re.search(r"NO deviation from the framework's canonical straight-through",
                 computations) and re.search(r"the ESTIMATOR -", mechanism):
        problems.append("the mechanism lists the estimator as one of its changes "
                        "while its computation states that the estimator IS the "
                        "framework's canonical one: the two must agree")
    return problems


def _no_claim_that_the_ranking_ignores_the_scale(spec: dict) -> list:
    """A degree-dependent statistic does not become scale-free by being ranked."""
    problems = []
    for entry in spec.get("computations") or []:
        text = str(entry.get("definition") or "") + " " + str(entry.get("note") or "")
        if re.search(r"never its magnitude|cannot dominate|the scaling cannot", text,
                     re.IGNORECASE):
            problems.append(
                "the specification claims the ranking is free of the statistic's "
                "scale, but the statistic carries an item-dependent degree factor: "
                "a monotone transform does not remove it")
    return problems


def _the_surface_exercises_the_curvature_it_declares(spec: dict) -> list:
    """The declared argument and the pinned constant must be the same curvature."""
    problems = []
    surface = spec.get("continuous_surface") or {}
    declared = None
    for entry in surface.get("args") or []:
        if isinstance(entry, dict) and entry.get("name") == "curvature":
            declared = entry.get("scalar")
    pinned = None
    for line in _code(spec):
        match = re.match(r"^\s*c\s*=\s*([0-9.]+)", line)
        if match:
            pinned = float(match.group(1))
    if declared is not None and pinned is not None and float(declared) != pinned:
        problems.append(
            "the surface declares the curvature %s while the listing pins c = %s: "
            "the surface must exercise the curvature it declares"
            % (declared, pinned))
    return problems


def _the_graph_drops_self_pairs(spec: dict) -> list:
    """The behaviour graph must not count an item as its own neighbour."""
    problems = []
    for entry in spec.get("computations") or []:
        text = str(entry.get("definition") or "")
        if re.search(r"A_ii\s*>\s*0|unless \(i,i\) occurs|self-adjacenc", text):
            if not re.search(r"self-pairs dropped|self-adjacencies are dropped|i\s*!=\s*j",
                             text):
                problems.append(
                    "the adjacency counts may contain a self-pair, so an item can be "
                    "its own neighbour: the self-pairs must be dropped")
    return problems


def _the_edge_less_items_are_stated(spec: dict) -> list:
    """An item with no neighbour has no cohesion: the rank must say what it gets."""
    problems = []
    text = "\n".join(str(entry.get("definition") or "")
                      for entry in spec.get("computations") or [])
    if re.search(r"1/deg_i", text) and not re.search(
            r"no (train(ing)? )?edge|without one|deg_i > 0 only|outside P", text):
        problems.append("the cohesion divides by deg_i but nothing says what an item "
                        "with no training edge gets: the rank must state it and "
                        "exclude it from the counts")
    return problems


def _the_degree_is_described_as_it_is_used(spec: dict) -> list:
    """The note about a quantity must describe the role the definition gives it."""
    problems = []
    for entry in spec.get("computations") or []:
        note = str(entry.get("note") or "")
        definition = str(entry.get("definition") or "")
        if re.search(r"deg", note) and re.search(r"top-D|top_D|D highest-count",
                                                 definition):
            if re.search(r"tie-break|top-D key", note) and "sqrt(deg_i * deg_j)" not in note:
                problems.append(
                    "the note describes the degree only as the top-D tie-breaker, "
                    "while the definition also divides by it: the note must "
                    "describe every role the quantity has")
    return problems


def _the_specification_does_not_change_the_idea(spec: dict, idea: dict = None) -> list:
    """Every quantity the idea fixes must be fixed the SAME way here.

    The reviewers kept finding the same class of defect: the specification
    quietly changed the idea's formula - a different denominator, a different
    population, a different constant. This check compares the two texts
    quantity by quantity: the numbers each states for the same symbol have to
    agree, or the specification has to name the change as an amendment.
    """
    problems = []
    if not idea:
        return problems
    equations = str(idea.get("spec_equations") or "")
    body = "\n".join(str(entry.get("definition") or "") + " " +
                      str(entry.get("note") or "") + " " +
                      str(entry.get("output") or "")
                      for entry in spec.get("computations") or [])
    body += "\n" + _section(spec, "switches") + "\n" + _section(spec, "parameters")
    for line in equations.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            continue
        numbers = set(re.findall(r"\d+\.?\d*", line.split("=", 1)[1]))
        numbers -= {"0", "1"}
        if not numbers:
            continue
        candidates = [segment for segment in body.splitlines()
                      if segment.strip().startswith(name)]
        if not candidates:
            continue
        stated = set()
        for segment in candidates:
            stated |= set(re.findall(r"\d+\.?\d*", segment))
        stated -= {"0", "1"}
        if numbers and not (numbers & stated):
            problems.append(
                "the specification states %s with different numbers than the idea "
                "does: the idea has %s, the specification has %s" % (name,
                                                                   sorted(numbers),
                                                                   sorted(stated)))
    return problems


def _the_specification_is_internally_consistent(spec: dict, idea: dict = None) -> list:
    """The specification must agree with itself: the symbols, the cost's sign,
    the order of the listing and the claims it makes about itself."""
    problems = []
    code = "\n".join(_code(spec))
    body = "\n".join(str(entry.get("definition") or "") + " " +
                      str(entry.get("note") or "") + " " +
                      str(entry.get("output") or "")
                      for entry in spec.get("computations") or [])
    switches = _section(spec, "switches")
    integration = _section(spec, "training_integration")
    # the cost subtracts the offset, and every switch says so with the same sign
    if re.search(r"C_ik\s*=\s*D_k\s*-\s*O_ik", body) and re.search(
            r"C_ik\s*=\s*2\|\|u[^\n]*\+\s*beta", switches):
        problems.append("the cost subtracts the offset while a switch writes it "
                        "with a plus sign")
    # the listing must define the plan before it is used
    order = [line.strip() for line in _code(spec)]
    plan_at = next((index for index, line in enumerate(order)
                    if "sinkhorn(" in line), None)
    use_at = next((index for index, line in enumerate(order)
                   if "argmax(P_ik" in line), None)
    if plan_at is not None and use_at is not None and use_at < plan_at:
        problems.append("the listing uses the plan before it computes it")
    # the loss must not claim an additive term the mechanism does not have
    if re.search(r"plus the mechanism's contribution|objective plus", integration) \
            and not re.search(r"NO (new |auxiliary )?term|no new term|no auxiliary",
                              body, re.IGNORECASE):
        problems.append("the training integration claims an additive contribution "
                        "that the objective does not have")
    # a symbol may not carry two meanings
    dimensions = spec.get("dimensions") or {}
    if "D" in dimensions and re.search(r"\bD\s*=\s*ceil\(mean", body):
        problems.append("the symbol D is both the latent width and the "
                        "neighbourhood cap")
    if "q" in dimensions and re.search(r"q\s*=\s*lift\(", code) and re.search(
            r"q_k is the RAW codeword", body):
        problems.append("the symbol q is both the lifted codebook and the raw "
                        "codeword")
    return problems


EVALS = (
    Eval("distance-clamp", "the distance keeps the artanh clamp",
         "round 20261105_144909: a draft dropped min(., 1 - 1e-6)",
         _distance_clamp),
    Eval("lift-zero", "LIFT(0) = 0 by arithmetic, not by a dead branch",
         "round 20261105_144909: a draft masked a branch _norm() could never make reachable",
         _lift_zero),
    Eval("curvature-zero-branch", "the geometry switch states its c = 0 arm",
         "round 20261105_144909: the surface had sqrt(c) with no branch",
         _curvature_zero_branch),
    Eval("behaviour-off-arithmetic", "the behaviour-off arm is arithmetically true",
         "round 20261105_144909: 'b_k = 0' was false (it is log 2) and hid a 0/0 claim",
         _behaviour_off_arithmetic),
    Eval("behaviour-chain", "the listing carries the behaviour chain",
         "round 20261105_144909: the surface formed its demand from a single "
         "minibatch and dropped the accumulation; the check now asks for the "
         "chain's shape, not for one mechanism's names",
         _behaviour_chain),
    Eval("epoch-one-weight", "w is defined during epoch 1",
         "round 20261105_144909: w needed a mean_scale that only exists after epoch 1",
         _epoch_one_weight),
    Eval("demand-surface", "the demand names the surface it is measured on",
         "round 20261105_144909: distance against offset-adjusted score was unstated",
         _demand_surface),
    Eval("scale-definition", "s is a vector-norm RMS and fails closed",
         "round 20261105_144909: 'the RMS of the latents' was ambiguous",
         _scale_definition),
    Eval("reconstruction-unlifted", "reconstruction uses the unlifted codeword",
         "round 20261105_144909: the recursion could be read as using LIFT(a_k)",
         _reconstruction_unlifted),
    Eval("code-identity-not-input-identity", "levels 2/3 share the code, not the inputs",
         "round 20261105_144909: 'levels 2 and 3 see exactly the baseline's inputs'",
         _code_identity_not_input_identity),
    Eval("demand-buffer-not-frozen", "the demand buffer is updated, not frozen",
         "round 20261105_144909: the parameters called b_raw a frozen demand",
         _demand_buffer_not_frozen),
    Eval("rho-label", "rho's label does not invert the ratio",
         "round 20261105_144909: 'radius-to-scale ratio' against R_c = s/rho",
         _rho_label),
    Eval("surface-defines-what-it-uses", "the listing defines every name it uses",
         "round 20261105_144909: score_k read an 'off' the listing never defined",
         _surface_defines_what_it_uses),

    Eval("surface-applies-the-lift", "the listing lifts the residual and the codebook",
         "round 20261105_144909: the surface never applied the lift",
         _surface_applies_the_lift),

    Eval("symbolic-distance-keeps-the-guards", "the symbolic distance is the idea's expression",
         "round 20261105_144909: the symbolic denominator dropped eps and the distance dropped the clamp",
         _symbolic_distance_keeps_the_guards),
    Eval("surface-interface-is-consistent", "the callable, the arguments and the output agree",
         "round 20261105_144909: the surface named a score and declared a distance's arguments",
         _surface_interface_is_consistent),
    Eval("surface-uses-its-arguments", "the listing uses the arguments it declares",
         "round 20261105_144909: the listing computed the curvature from s and ignored the declared argument",
         _surface_uses_its_arguments),
    Eval("the-specification-does-not-change-the-idea", "the specification states the idea's numbers",
         "round 20261106_020541: the reviewers kept finding the specification changing "
         "the idea's formula - a denominator, a population, a constant",
         _the_specification_does_not_change_the_idea),
    Eval("the-specification-is-internally-consistent", "the specification agrees with itself",
         "round 20261106_020541: the sign of the offset, the order of the listing, the "
         "overloaded symbols and the additive-loss claim",
         _the_specification_is_internally_consistent),
    Eval("the-graph-drops-self-pairs", "an item is not its own neighbour",
         "round 20261105_144909: the adjacency counts allowed a self-pair",
         _the_graph_drops_self_pairs),
    Eval("the-edge-less-items-are-stated", "the items without an edge are stated",
         "round 20261105_144909: coh was undefined for deg = 0 and the rank "
         "overwrote the value afterwards",
         _the_edge_less_items_are_stated),
    Eval("the-degree-is-described-as-it-is-used", "a note describes every role",
         "round 20261105_144909: the note described the degree only as the "
         "tie-breaker",
         _the_degree_is_described_as_it_is_used),
    Eval("the-surface-does-not-bind-its-own-temperature", "the plan's temperature is the baseline's",
         "round 20261106_020541: a stray tau in the listing would change the plan",
         _the_surface_does_not_bind_its_own_temperature),
    Eval("the-mechanisms-list-agrees-with-its-computations", "the mechanism and its computations agree",
         "round 20261106_020541: the mechanism listed the estimator as a change while "
         "the computation said there was none",
         _the_mechanisms_list_agrees_with_its_computations),
    Eval("no-claim-that-the-ranking-ignores-the-scale", "a ranking does not remove a degree factor",
         "round 20261106_020541: the claim that the ranking is scale-free was false",
         _no_claim_that_the_ranking_ignores_the_scale),
    Eval("the-surface-exercises-the-curvature-it-declares", "the declared curvature is the pinned one",
         "round 20261106_020541: the surface declared 0.25 while the listing pinned c = 1",
         _the_surface_exercises_the_curvature_it_declares),
    Eval("offset-scale-is-one-global-quantile", "the amplitude is one quantile of the pooled batch",
         "round 20261106_020541: the reviewers found that a mean of the ranks' medians "
         "is not a quantile of anything",
         _offset_scale_is_one_global_quantile),
    Eval("the-ordering-is-not-claimed-as-maintained", "the strata's order is an initialisation",
         "round 20261106_020541: the strata are free parameters, so the ordering "
         "cannot be claimed as an invariant",
         _the_ordering_is_not_claimed_as_maintained),
    Eval("declared-intervals-match-the-definitions", "a declared range is the range the definition gives",
         "round 20261106_020541: the stratum's output said [0,1] while sigmoid gives (0,1)",
         _declared_intervals_match_the_definitions),
    Eval("no-unattested-fallback-is-attributed-to-the-idea", "the idea is not credited with rules it lacks",
         "round 20261106_020541: a fallback the idea never stated was attributed to it",
         _no_unattested_fallback_is_attributed_to_the_idea),
    Eval("the-geometry-note-states-properties", "the sign is argued with the properties",
         "round 20261106_020541: the note's counterfactual about the sign was wrong",
         _the_geometry_note_states_properties),
    Eval("geometry-arm-keeps-the-measured-scale", "the Euclidean arm is distance-only",
         "round 20261106_020541: the reviewers found that re-measuring beta in the "
         "c = 0 arm would make the control differ in the amplitude too",
         _geometry_arm_keeps_the_measured_scale),
    Eval("assignment-keeps-the-balancing", "the baseline's balanced assignment stays",
         "round 20261105_144909: the specification replaced the Sinkhorn-balanced "
         "assignment with a plain argmin and the level-1 codebook collapsed "
         "(42 of 256 codes, L3 collision 0.703 against the reference's 0.065)",
         _assignment_keeps_the_balancing),
)


#: The identities the numeric eval checks. Only the specification's FUNCTION
#: definitions are executed: the identities are properties of the maps the
#: specification names, so they can be driven with inputs of the eval's choosing
#: instead of the statement stream's own ordering.
IDENTITIES = """
import json, math, torch
torch.manual_seed(0)
eps = 1e-12
tau = 1.0
rho = 0.5
rho_off = 0.10
eta = 0.1
s = 2.0
c = (rho / s) ** 2
out = {}
%(statements)s
B, D, K1 = 4, 6, 5
out["functions"] = sorted(name for name, value in list(globals().items())
                          if callable(value) and not name.startswith("_") is False)
x = torch.randn(B, D) * 0.3
y = torch.randn(K1, D) * 0.3
out["lift_zero"] = float(lift(torch.zeros(2, D), c).abs().max())
out["dist_self"] = float(dist(x, x, c).abs().max())
out["c_zero_gap"] = float((dist(x[:, None, :], y[None, :, :], 0.0) - 2.0 * (x[:, None, :] - y[None, :, :]).norm(dim=-1)).abs().max())
out["c_small_gap"] = float((dist(x[:, None, :], y[None, :, :], 1e-9) - 2.0 * (x[:, None, :] - y[None, :, :]).norm(dim=-1)).abs().max())
out["lift_bound"] = float((lift(x, c).norm(dim=-1) * math.sqrt(c)).max())
print("@@RESULT@@" + json.dumps(out))
"""


def run_numeric(spec: dict, *, python: str = TORCH_PYTHON) -> dict:
    """Execute the specification's own geometry and check its identities.

    A specification is arithmetic. The checks below are the properties the
    mechanism claims as definitions - LIFT(0) = 0, DIST(x, x) = 0, the Euclidean
    limit of the distance, the lift staying inside the ball - so they hold for
    any candidate that implements what it wrote and fail for one that wrote
    something else. The eval drives the specification's own functions with
    inputs of its own choosing; it never trusts the statement order.
    """
    source = _executable(spec) or _statements(spec)
    statements = [item for item in source
                  if item.lstrip().startswith("def ") or item.lstrip().startswith("eps =")]
    if not any("def lift" in item for item in statements):
        return {"id": "numeric-identities", "title": "the geometry runs",
                "source": "every round",
                "problems": ["the surface defines no lift function to execute"]}
    script = IDENTITIES % {"statements": "\n".join(statements)}
    path = Path(tempfile.mkdtemp(prefix="spec-eval-")) / "run.py"
    path.write_text(script, encoding="utf-8")
    try:
        done = subprocess.run([python, str(path)], capture_output=True, text=True,
                              timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        return {"id": "numeric-identities", "title": "the geometry runs",
                "source": "every round",
                "problems": ["the numeric eval could not run: %s" % error]}
    marker = "@@RESULT@@"
    line = next((item for item in reversed(done.stdout.splitlines())
                 if item.startswith(marker)), "")
    if not line:
        return {"id": "numeric-identities", "title": "the geometry runs",
                "source": "every round",
                "problems": ["the geometry did not execute: " +
                             (done.stderr or done.stdout or "no output").strip()[-400:]]}
    values = json.loads(line[len(marker):])
    problems = []
    if values["lift_zero"] > 1e-7:
        problems.append("LIFT(0) is %g, not 0" % values["lift_zero"])
    if values["dist_self"] > 1e-6:
        problems.append("DIST(x, x) is %g, not 0" % values["dist_self"])
    if values["c_zero_gap"] > 1e-5:
        problems.append("the c = 0 arm is off the Euclidean 2||x - y|| by %g"
                        % values["c_zero_gap"])
    if values["c_small_gap"] > 1e-3:
        problems.append("the c -> 0 limit is off the Euclidean 2||x - y|| by %g"
                        % values["c_small_gap"])
    if values["lift_bound"] > 1.0 + 1e-5:
        problems.append("the lift leaves the ball: sqrt(c)||LIFT(v)|| = %g > 1"
                        % values["lift_bound"])
    if not problems:
        return {}
    return {"id": "numeric-identities", "title": "the geometry runs",
            "source": "every round", "problems": problems}


def run(spec: dict, *, numeric=None, idea: dict = None) -> dict:
    """Every eval, in order; the gate is closed when any of them reports."""
    failures = []
    for item in EVALS:
        if item.id == "the-specification-does-not-change-the-idea":
            problems = list(item.check(spec, idea) or [])
        elif item.id == "the-specification-is-internally-consistent":
            problems = list(item.check(spec, idea) or [])
        else:
            problems = list(item.check(spec) or [])
        if problems:
            failures.append({"id": item.id, "title": item.title,
                             "source": item.source, "problems": problems})
    numeric_failure = numeric(spec) if numeric else run_numeric(spec)
    if numeric_failure:
        failures.append(numeric_failure)
    return {
        "passed": not failures,
        "checked": len(EVALS) + 1,
        "failures": failures,
        "findings": [problem for failure in failures
                     for problem in failure.get("problems") or []],
    }


# ---- the ledger: a caught defect must be decided, not forgotten -------------

#: Where the decisions live: one entry per defect class, either an eval that now
#: refuses it or a documented reason it cannot be decided mechanically.
LEDGER_PATH = Path(__file__).with_name("eval_ledger.json")


def ledger(path: Path = None) -> list:
    """The recorded decisions, oldest first."""
    target = Path(path) if path else LEDGER_PATH
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    return list(payload.get("entries") or [])


def resolve(finding: str, entries: list = None) -> dict | None:
    """The ledger entry a finding falls under, or None.

    An entry matches by KEYWORDS as well as by a pattern: a class of defect is
    decided once, and the wording of the reviewer's sentence is not the class.
    """
    text = str(finding or "")
    lowered = text.lower()
    for entry in entries if entries is not None else ledger():
        pattern = str(entry.get("pattern") or "")
        if pattern and re.search(pattern, text, re.IGNORECASE):
            return entry
        keywords = [str(word).lower() for word in entry.get("keywords") or []]
        if len(keywords) >= 2 and sum(1 for word in keywords if word in lowered) >= 2:
            return entry
    return None


def undecided(findings: list, entries: list = None) -> list:
    """The findings no ledger entry covers.

    A round may not freeze while one of its findings has no decision: the
    decision is either an eval that now refuses the defect or a recorded reason
    it is not mechanically decidable. That is the whole point of the ledger -
    the next round inherits the refusal, not a memory of it.
    """
    decided = entries if entries is not None else ledger()
    return [finding for finding in findings if resolve(finding, decided) is None]
