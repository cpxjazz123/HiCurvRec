"""Static source/contract audit for Iter31 HRA-STEP6-1; runtime proof is S08-only."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = ROOT.parent / "curvature_RQ-VAE_iter29"
HRA_PATH = ROOT / "logs/hra_step6_contract_iter31.json"
FCCR_PATH = ROOT / "logs/mechanism_contract_iter31.json"
MODEL_PATH = ROOT / "modules/rqvae.py"
PARENT_MODEL_PATH = PARENT_ROOT / "modules/rqvae.py"
FIXED_C = [1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
FCCR_FIELDS = {
    "contract_version": "FCCR-1",
    "curvature_source": "closed_form",
    "curvature_trainable": False,
    "curvature_time_varying": False,
    "uses_cyclic_schedule": False,
    "uses_curvature_regularization": False,
    "new_curvature_conditioned_optimizer": False,
    "new_curvature_conditioned_aux_loss": False,
    "formula_inputs": ["behavior_branching", "raw_residual_median"],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"HRA_STEP6_PREFLIGHT FAIL: {message}")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"HRA_STEP6_PREFLIGHT FAIL: missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _pattern_dump(source: str) -> str:
    parsed = ast.parse(source).body[0]
    node = parsed.value if isinstance(parsed, ast.Expr) else parsed
    return ast.dump(node, include_attributes=False)


def _has_node(root: ast.AST, source: str) -> bool:
    expected = _pattern_dump(source)
    return any(
        ast.dump(node, include_attributes=False) == expected
        for node in ast.walk(root)
    )


def _find_method(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        child
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "RqVae"
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name == name
    ]
    _require(len(matches) == 1, f"expected one RqVae.{name}, found {len(matches)}")
    return matches[0]


def _normalized_without_step6(source: str) -> str:
    tree = ast.parse(source)
    method = _find_method(tree, "_step6_sum_embeddings")
    method.body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def _check_contracts(hra: dict, fccr: dict) -> None:
    expected_fccr_keys = set(FCCR_FIELDS) | {"final_curvature_values"}
    _require(set(fccr) == expected_fccr_keys, "FCCR-1 field set is not exactly ten fields")
    for key, expected in FCCR_FIELDS.items():
        observed = fccr.get(key)
        _require(type(observed) is type(expected) and observed == expected,
                 f"FCCR-1 field {key} differs from the registered value")
    _require(fccr.get("final_curvature_values") == FIXED_C,
             "FCCR-1 fixed curvature vector differs")

    _require(hra.get("contract_version") == "HRA-STEP6-1", "wrong HRA contract version")
    _require(
        hra.get("contract_status") == "ACTIVE_FOR_ITER31_BY_S04_BETWEEN_ITERATION_TRANSITION",
        "HRA transition is not active for Iter31",
    )
    transition = hra.get("transition", {})
    _require(
        transition.get("from_iteration") == 30
        and transition.get("from_status") == "ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE"
        and transition.get("from_scientific_result") == "NONE"
        and transition.get("from_iteration_stage2_executed") is False
        and transition.get("from_iteration_stage3_executed") is False
        and transition.get("from_iteration_gpu_training_executed") is False
        and transition.get("to_iteration") == 31
        and transition.get("parent_iteration") == 29
        and transition.get("transition_type") == "BETWEEN_ITERATION_CONTRACT_TRANSITION"
        and transition.get("transition_authority") == "CANONICAL_S04_JUDGE_ONLY"
        and transition.get("transition_scope")
        == "ACTIVATE_HRA_STEP6_ONLY_WHILE_PRESERVING_FCCR1_FIXED_CURVATURE_SUBCONTRACT",
        "Iter30-to-Iter31 transition identity/scope differs",
    )
    mechanism = hra.get("mechanism", {})
    _require(
        mechanism.get("mechanism_id") == "HRA_INSPIRED_STEP6_COMMON_REFERENCE_AGGREGATION"
        and mechanism.get("equation_id") == "S02_ITER31_HRA_COMMON_C0_RIGHT_NESTED_V1"
        and mechanism.get("paper_equivalence_claimed") is False
        and mechanism.get("d_hste_included") is False,
        "HRA mechanism identity or exclusion flags differ",
    )
    inherited = hra.get("inherited_curvature_contract", {})
    _require(
        inherited.get("path") == "logs/mechanism_contract_iter31.json"
        and inherited.get("contract_version") == "FCCR-1"
        and inherited.get("curvature_source") == "closed_form"
        and inherited.get("curvature_trainable") is False
        and inherited.get("curvature_time_varying") is False
        and inherited.get("formula_inputs") == ["behavior_branching", "raw_residual_median"]
        and inherited.get("layer_order") == ["L0", "L1", "L2"]
        and inherited.get("final_curvature_values") == FIXED_C
        and inherited.get("historical_input_json_keys") == ["branching", "raw_residual_medians"]
        and inherited.get("historical_input_replay_or_hash_proof") is False
        and inherited.get("fresh_measurement_or_fallback_allowed") is False
        and inherited.get("normalized_layer_scale_substitution_allowed") is False,
        "HRA-to-FCCR linkage or inherited provenance boundary differs",
    )
    equation = hra.get("equation", {})
    _require(
        equation.get("layer_indices") == [0, 1, 2]
        and equation.get("layer_order") == ["L0", "L1", "L2"]
        and equation.get("evaluation_order") == [
            "q_0 = exp0^{c_0}(e_0); q_0^0 = exp0^{c0}(log0^{c_0}(q_0))",
            "q_1 = exp0^{c_1}(e_1); q_1^0 = exp0^{c0}(log0^{c_1}(q_1))",
            "q_2 = exp0^{c_2}(e_2); q_2^0 = exp0^{c0}(log0^{c_2}(q_2))",
            "inner = q_1^0 ⊕_{c0} q_2^0",
            "h = q_0^0 ⊕_{c0} inner",
            "z = log0^{c0}(h)",
        ]
        and equation.get("replaces_only") == "z_E = e_0 + e_1 + e_2 at Iter29 Step6"
        and equation.get("mobius_addition_associative") is False
        and equation.get("reassociation_or_reordering_allowed") is False
        and equation.get("log0_directly_on_e_l_allowed") is False,
        "registered HRA equation/order/limits differ",
    )
    symbols = equation.get("symbols", {})
    e_l = symbols.get("e_l", {})
    q_l = symbols.get("q_l", {})
    q_l_common = symbols.get("q_l_common", {})
    h = symbols.get("h", {})
    z = symbols.get("z", {})
    _require(
        e_l.get("type") == "tangent_coordinate_quantizer_embedding"
        and e_l.get("is_ball_point") is False
        and e_l.get("is_token_id") is False
        and q_l.get("type") == "Poincare_ball_point_in_B_c_l"
        and q_l.get("source_curvature") == "c_l"
        and q_l_common.get("symbol") == "q_l^0"
        and q_l_common.get("type") == "Poincare_ball_point_in_B_c0"
        and q_l_common.get("log_source_curvature") == "c_l"
        and q_l_common.get("exp_target_curvature") == "c0"
        and h.get("type") == "Poincare_ball_aggregation_at_c0"
        and h.get("mobius_curvature_for_both_operations") == "c0"
        and z.get("type") == "tangent_coordinate_decoder_input"
        and z.get("final_log_curvature") == "c0"
        and equation.get("decoder_consumer")
        == "existing decoder receives z as tangent-coordinate input",
        "HRA tangent/ball semantics or curvature roles differ",
    )
    limits = hra.get("mathematical_limits", {})
    _require(
        limits.get("exact_paper_hra_telescope_claimed") is False
        and limits.get("cross_curvature_mobius_homomorphism_claimed") is False
        and limits.get("heterogeneous_curvature_residual_inversion_claimed") is False
        and limits.get("exact_residual_reconstruction_claimed") is False
        and limits.get("origin_exp_log_identity_unconditional") is False,
        "HRA mathematical limits were weakened",
    )
    helper_caveats = hra.get("helper_domain_caveats", {})
    _require(
        helper_caveats.get("runtime_finiteness_shape_and_ball_domain") == "must_be_verified_at_S08"
        and helper_caveats.get("projection_and_log_clamp_incidence") == "must_be_measured_at_S08"
        and helper_caveats.get("pointwise_exp_log_identity")
        == "conditional_on_valid_unclipped_domain_only",
        "HRA helper-domain caveats differ",
    )
    _require(
        helper_caveats.get("origin_exp_helper")
        == "existing helper projects output to radius (1-eps)/sqrt(c), default eps=1e-6"
        and helper_caveats.get("origin_log_helper")
        == "existing helper clamps the atanh argument sqrt(c)*norm to at most 1-1e-5"
        and helper_caveats.get("mobius_add_helper")
        == "existing helper clamps its denominator from below and does not project the output",
        "registered geometry-helper domain semantics differ",
    )
    scope = hra.get("scope_and_invariants", {})
    _require(
        scope.get("single_new_mechanism") == "Step6 aggregation replacement only"
        and scope.get("unchanged") == [
            "FCCR-1 mapping, inputs, fixed curvature values, and fixed-buffer behavior",
            "Stage1 embeddings and data",
            "Step4 residual subtraction",
            "Step5 cross-layer residual transport",
            "quantizer, codebooks, assignments, and existing STE",
            "all existing losses and loss weights",
            "optimizer, Sinkhorn configuration, training schedule, and seed",
            "decoder and reconstruction-loss path",
            "Stage3 model, data, evaluator, and protocol",
        ]
        and scope.get("forbidden_additions") == [
            "d-HSTE",
            "additional loss or auxiliary mechanism",
            "curvature change, learning, or schedule",
            "optimizer or Sinkhorn change",
            "codebook or assignment change",
            "Stage1 or Stage3 change",
            "any second new mechanism",
        ],
        "registered one-factor scope/invariants differ",
    )
    checker = hra.get("checker_contract", {})
    _require(
        checker.get("required_distinct_hra_checker") == "scripts/preflight_hra_step6_iter31.py"
        and checker.get("hra_checker_arguments") == "NONE"
        and checker.get("hra_checker_environment_overrides") is False
        and checker.get("both_checkers_must_pass_independently") is True
        and checker.get("shared_fccr_checker_modification_or_bypass_allowed") is False,
        "checker path/authorization contract differs",
    )
    _require(
        checker.get("hra_checker_must_check") == [
            "both canonical contracts and their linkage",
            "exact S02 operation, L0-L1-L2 leaf order, and right-nested expression",
            "tangent embedding versus ball-point semantics",
            "source c_l and target c0 curvature argument roles",
            "Step6 replacement and existing decoder consumer",
            "unchanged scope and absence of forbidden additions",
            "helper-domain caveats and required S08 runtime observations",
        ],
        "HRA checker required-scope contract differs",
    )


def _check_source() -> None:
    if not MODEL_PATH.is_file() or not PARENT_MODEL_PATH.is_file():
        raise FileNotFoundError("HRA_STEP6_PREFLIGHT FAIL: current/parent rqvae.py missing")
    source = MODEL_PATH.read_text(encoding="utf-8")
    parent_source = PARENT_MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODEL_PATH))
    parent_tree = ast.parse(parent_source, filename=str(PARENT_MODEL_PATH))
    method = _find_method(tree, "_step6_sum_embeddings")
    parent_method = _find_method(parent_tree, "_step6_sum_embeddings")
    _require(
        _normalized_without_step6(source) == _normalized_without_step6(parent_source),
        "RqVae source differs from Iter29 outside the approved Step6 method",
    )
    _require(
        _has_node(parent_method, "embeddings = quantized.embeddings.sum(dim=0).transpose(0, 1)"),
        "Iter29 baseline Step6 is not the registered Euclidean sum",
    )

    required_patterns = [
        "embeddings = quantized.embeddings",
        "batch_size = embeddings.shape[2]",
        "curvatures = [layer.get_c().view(1, 1) for layer in self.layers]",
        "c0 = curvatures[0]",
        "e_l = embeddings[layer_index].transpose(0, 1)",
        "q_l = _expmap0_t(e_l, curvatures[layer_index])",
        "q_l_common = _expmap0_t(_logmap0_t(q_l, curvatures[layer_index]), c0)",
        "common_points.append(q_l_common)",
        "inner = _mobius_add_t(common_points[1], common_points[2], c0)",
        "h = _mobius_add_t(common_points[0], inner, c0)",
        "result = _logmap0_t(h, c0)",
        "return result",
    ]
    for pattern in required_patterns:
        _require(_has_node(method, pattern), f"Step6 missing required source structure: {pattern}")
    forbidden_names = {
        "d_hste", "aux_loss", "loss", "reconstruction_loss", "behavior_loss",
        "optimizer", "sinkhorn",
    }
    _require(
        not any(
            isinstance(node, ast.Name) and node.id in forbidden_names
            for node in ast.walk(method)
        ),
        "Step6 contains an unregistered loss/optimizer/auxiliary mechanism",
    )
    call_counts = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            call_counts[node.func.id] = call_counts.get(node.func.id, 0) + 1
    _require(
        call_counts.get("_expmap0_t") == 2
        and call_counts.get("_logmap0_t") == 2
        and call_counts.get("_mobius_add_t") == 2,
        "Step6 adds or omits geometry operations outside the registered equation",
    )

    layer_loop = [
        node for node in ast.walk(method)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "layer_index"
    ]
    _require(
        len(layer_loop) == 1
        and isinstance(layer_loop[0].iter, ast.Tuple)
        and [elt.value for elt in layer_loop[0].iter.elts if isinstance(elt, ast.Constant)]
        == [0, 1, 2],
        "Step6 layer loop is not exact ascending L0/L1/L2 order",
    )
    _require(
        _has_node(method, "embeddings.ndim != 3")
        and _has_node(method, "embeddings.shape[0] != 3")
        and _has_node(method, "embeddings.shape[1] != self.embed_dim")
        and _has_node(method, "result.shape != (batch_size, self.embed_dim)")
        and _has_node(method, "not torch.isfinite(result).all().item()"),
        "Step6 shape/finiteness guards are missing",
    )
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "_logmap0_t" and node.args:
            _require(
                not (isinstance(node.args[0], ast.Name) and node.args[0].id == "e_l"),
                "Step6 must not apply log0 directly to tangent e_l",
            )
    forward = _find_method(tree, "forward")
    _require(
        _has_node(forward, "summed_embeddings = self._step6_sum_embeddings(quantized)")
        and _has_node(forward, "self.decode(summed_embeddings)"),
        "forward no longer sends Step6 output to the existing decoder",
    )

    for relative in ("hyperbolic.py", "quantize.py", "loss.py"):
        current = ROOT / "modules" / relative
        parent = PARENT_ROOT / "modules" / relative
        _require(current.read_bytes() == parent.read_bytes(),
                 f"inherited module changed unexpectedly: modules/{relative}")


def main() -> None:
    hra = _load_json(HRA_PATH)
    fccr = _load_json(FCCR_PATH)
    _check_contracts(hra, fccr)
    _check_source()
    print("HRA_STEP6_STATIC_PREFLIGHT PASS")
    print("FCCR-1 and HRA-STEP6-1 contracts are separate and linked")
    print("Static equation/source scope verified; runtime activation is NOT established")
    print(
        "S08_REQUIRED=shape,finiteness,ball-domain,projection/log-clamp incidence,"
        "fixed-curvature/optimizer invariance,model/codebook gradients,"
        "same-checkpoint/same-batch direct output effect"
    )


if __name__ == "__main__":
    main()
