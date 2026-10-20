"""Real-execution adapters for CGV.

An adapter is the ONLY place that touches a mechanism's implementation. It builds a small but
genuine training case, runs the mechanism's real code path, and hands the raw observations to the
engine and to the independent oracles.

What an adapter must never do is decide whether the mechanism is correct: it produces
observations, the oracle produces expectations, and the engine compares them. An adapter that
computed an expectation would put the mechanism back in the position of being its own judge.

Everything mechanism-specific is read from the contract: the executor symbol, the loss sources,
the parameter groups, the activation flags and the epoch gate. The adapter knows how to talk to
this RQ-VAE family; the contract says what to talk to.
"""
from __future__ import annotations

import inspect
from typing import Any

import torch

N_SRC = 24
IN_DIM = 32


# --------------------------------------------------------------------------------------
# loss sources: contract term name -> how the real case exposes it
# --------------------------------------------------------------------------------------
LOSS_SOURCES = {
    "base_loss": lambda T, obs: obs["base_loss"],
    "recon_loss": lambda T, obs: obs["recon_loss"],
    "quant_loss": lambda T, obs: obs["quant_loss"],
    "behavior_l1": lambda T, obs: obs["l1_loss"],
    "behavior_other": lambda T, obs: obs["other_loss"],
}


def resolve_dotted(trainer, dotted: str) -> Any:
    """Resolve ``"symbol"`` or ``"symbol.attribute"`` inside the loaded trainer module."""
    head, _, tail = dotted.partition(".")
    if not hasattr(trainer, head):
        raise AttributeError(f"{getattr(trainer, '__file__', '?')} defines no {head!r}")
    value = getattr(trainer, head)
    if not tail:
        return value
    for part in tail.split("."):
        value = getattr(value, part)
    return value


def declared_value(trainer, section: dict[str, Any], key: str, default: Any) -> Any:
    """A declared constant, either written in the contract or read from the trainer's own flag.

    ``alpha: 0.5`` states the value in the declaration; ``alpha_flag: "R5_05_ALPHA"`` points at
    the trainer constant the mechanism actually uses. The flag form is preferred for anything the
    mechanism itself reads, because it cannot drift away from the implementation.
    """
    flag = section.get(f"{key}_flag")
    if flag:
        if not hasattr(trainer, flag):
            raise AttributeError(f"the contract names {flag!r}, which the trainer does not define")
        return getattr(trainer, flag)
    return section.get(key, default)


def structural_blocks(trainer, contract: dict[str, Any],
                      obs: dict[str, Any]) -> list[list[int]]:
    """The declared parameter blocks of a blockwise mechanism.

    Structural data, not an expectation: the contract names the helper that partitions the target
    parameters, and the oracle then checks the declared per-block mathematics on those blocks.
    """
    declared = (contract.get("gradient_contract") or {}).get("blocks") or {}
    flag = declared.get("flag")
    if not flag:
        raise ValueError("gradient_contract.blocks needs a 'flag' naming the partitioning helper")
    helper = getattr(trainer, flag, None)
    if helper is None:
        raise AttributeError(f"the contract names the block helper {flag!r}, which is absent")
    group = declared.get("group") or (contract.get("gradient_contract") or {}).get(
        "target", "encoder")
    module = resolve_dotted(obs["model"], declared.get("module", group))
    return [list(block) for block in helper(module, obs["encoder"])]


def parameter_group(trainer, obs, contract, group: str) -> list[torch.nn.Parameter]:
    """The declared parameter group, resolved through the contract's attribute paths."""
    paths = (contract.get("model") or {}).get("parameter_groups") or {}
    if group not in paths:
        raise KeyError(f"the contract declares no parameter group {group!r}")
    target = resolve_dotted(obs["model"], paths[group])
    return [parameter for parameter in target.parameters()]


# --------------------------------------------------------------------------------------
# building a real mini case
# --------------------------------------------------------------------------------------
def build_observations(trainer, contract: dict[str, Any], seed: int = 1234,
                       n_src: int | None = None, model=None) -> dict[str, Any]:
    """Run the real forward once and collect everything the gates and oracles need.

    ``model`` lets a caller supply an already-created module. The multi-rank path needs it: DDP
    broadcasts parameters at construction, so the per-rank local gradients are only comparable
    with the DDP-averaged ones when they are taken at that same synchronised parameter point.
    """
    probe = (contract.get("model") or {}).get("probe") or {}
    n_src = int(n_src if n_src is not None else probe.get("n_src", N_SRC))
    in_dim = int(probe.get("in_dim", IN_DIM))
    if model is None:
        torch.manual_seed(seed)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=in_dim)
    embeddings = torch.randn(2 * n_src, in_dim,
                             generator=torch.Generator().manual_seed(seed + 1))
    reconstructed, quant_loss, _, _ = model.forward(embeddings)
    base_loss, recon_loss = model.compute_loss(embeddings, reconstructed, quant_loss)

    residuals = trainer._residuals_with_trace(model, embeddings)
    codewords = _codewords(trainer, model, embeddings)
    l1_loss, other_loss, valid_levels = trainer.behavior_contrastive_components(
        residuals, torch.arange(n_src), (torch.arange(n_src) + 1) % n_src, None)

    with torch.no_grad():
        tokens = model.get_indices(embeddings, infer_use_sk=False)

    weight = float((contract.get("gradient_contract") or {}).get("probe_weight", 0.5))
    scale = weight / max(int(valid_levels), 1)

    observations: dict[str, Any] = {
        "trainer": trainer,
        "model": model,
        "seed": seed,
        "n_src": n_src,
        "embeddings": embeddings,
        "base_loss": base_loss,
        "recon_loss": recon_loss,
        "quant_loss": quant_loss,
        "l1_loss": l1_loss,
        "other_loss": other_loss,
        "valid_levels": int(valid_levels),
        "residuals": residuals,
        "codewords": codewords,
        "scale": scale,
        "weight": weight,
        "tensors": {
            "reconstructed": reconstructed,
            "quant_loss": quant_loss,
            "base_loss": base_loss,
            "recon_loss": recon_loss,
            "tokens": tokens,
        },
    }
    observations["encoder"] = parameter_group(trainer, observations, contract,
                                              (contract.get("gradient_contract") or {})
                                              .get("target", "encoder"))
    return observations


def _codewords(trainer, model, embeddings: torch.Tensor) -> list[torch.Tensor]:
    """The quantised vectors each level actually subtracted, observed from the real quantizer."""
    residual = model.encoder(embeddings)
    out = []
    for layer in model.rq.vq_layers:
        quant, _, _, _ = layer(residual)
        out.append(quant.detach())
        residual = residual - quant
    return out


def build_model(trainer, contract: dict[str, Any], seed: int = 1000):
    """A freshly seeded model of the declared type, for callers that need only the module."""
    probe = (contract.get("model") or {}).get("probe") or {}
    in_dim = int(probe.get("in_dim", IN_DIM))
    torch.manual_seed(seed)
    return trainer.RQVAE(trainer._tokenizer_config(), in_dim=in_dim)


def loss_terms(trainer, contract: dict[str, Any], obs: dict[str, Any],
               substitute: dict[str, torch.Tensor] | None = None
               ) -> dict[str, torch.Tensor]:
    """Resolve the contract's declared loss terms to live tensors on this case."""
    declared = (contract.get("gradient_contract") or {}).get("loss_terms") or {}
    terms: dict[str, torch.Tensor] = {}
    for name, source in declared.items():
        if substitute and name in substitute:
            terms[name] = substitute[name]
            continue
        if source not in LOSS_SOURCES:
            raise KeyError(f"unknown loss source {source!r} for term {name!r}")
        value = LOSS_SOURCES[source](trainer, obs)
        if value is None:
            raise ValueError(f"loss term {name!r} resolved to None on this case")
        terms[name] = value
    return terms


# --------------------------------------------------------------------------------------
# declared activation
# --------------------------------------------------------------------------------------
def activation_open(trainer, contract: dict[str, Any], epoch: int | None = None) -> bool:
    """Whether the contract's declared activation condition holds at this epoch.

    The switch is OPTIONAL: a mechanism with no switch simply declares none and is always open.
    """
    activation = (contract.get("mechanism") or {}).get("activation") or {}
    for flag in activation.get("requires", []) or []:
        if not getattr(trainer, flag, False):
            return False
    flag = activation.get("flag")
    if flag and not getattr(trainer, flag, False):
        return False
    gate = activation.get("epoch_gate")
    if gate and epoch is not None:
        if epoch < int(getattr(trainer, gate)):
            return False
    return True


# --------------------------------------------------------------------------------------
# executing the real gradient assembly
# --------------------------------------------------------------------------------------
def executor_kwargs(trainer, contract, obs, epoch: int | None = None) -> dict[str, Any]:
    """Pass only the arguments the declared executor actually accepts.

    A mechanism that needs tensors of its own declares an ``inputs`` provider: a symbol in
    the tree that turns the observations into the tensors the mechanism reads. The provider
    is called by the verifier, not by the mechanism, and its result is handed to the
    executor as ``mechanism_inputs``.
    """
    name = (contract.get("mechanism") or {}).get("executor")
    if not name:
        return {}
    executor = resolve_dotted(trainer, name)
    accepted = inspect.signature(executor).parameters
    kwargs: dict[str, Any] = {}
    if "recon_loss" in accepted:
        kwargs["recon_loss"] = obs["recon_loss"]
    if "encoder_module" in accepted:
        kwargs["encoder_module"] = obs["model"].encoder
    if "quant_loss" in accepted:
        kwargs["quant_loss"] = obs["quant_loss"]
    if "epoch" in accepted and epoch is not None:
        kwargs["epoch"] = epoch
    provider_name = (contract.get("mechanism") or {}).get("inputs")
    if provider_name and "mechanism_inputs" in accepted:
        provider = resolve_dotted(trainer, provider_name)
        provider_parameters = inspect.signature(provider).parameters
        provided = provider(obs) if provider_parameters else provider()
        kwargs["mechanism_inputs"] = provided
    return kwargs


def executed_target_gradient(trainer, contract: dict[str, Any], obs: dict[str, Any],
                             *, epoch: int | None = None,
                             substitute: dict[str, torch.Tensor] | None = None
                             ) -> list[torch.Tensor]:
    """Run the mechanism's real assembly and return the target-parameter gradient it produced.

    When the declared activation is closed, the contract says the plain assembly must run
    instead, so the executed result is the unprojected one -- not an empty gradient.

    ``substitute`` replaces declared loss terms with probe tensors on the same graph, which is how
    the propagation gate finds out which terms the mechanism actually reads.
    """
    model = obs["model"]
    encoder = obs["encoder"]
    terms = loss_terms(trainer, contract, obs, substitute=substitute)
    model.zero_grad(set_to_none=False)

    if not activation_open(trainer, contract, epoch):
        assembled = terms["base_loss"] + obs["scale"] * (
            _sum_present(terms.get("l1_loss"), terms.get("other_loss")))
        assembled.backward()
    else:
        terms["base_loss"].backward(retain_graph=True)
        executor = resolve_dotted(trainer, (contract.get("mechanism") or {})["executor"])
        executor(terms["base_loss"], terms.get("l1_loss"), terms.get("other_loss"),
                 encoder, obs["weight"], valid_levels=obs["valid_levels"],
                 **executor_kwargs(trainer, contract, obs, epoch))
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
            for p in encoder]


def _sum_present(*terms) -> torch.Tensor:
    present = [term for term in terms if term is not None]
    if not present:
        raise ValueError("the declared assembly needs at least one behaviour term")
    total = present[0]
    for term in present[1:]:
        total = total + term
    return total


def term_gradient(trainer, obs: dict[str, Any], term: torch.Tensor) -> list[torch.Tensor]:
    """The independent route to a term's gradient: ``torch.autograd.grad`` on the real graph.

    Deliberately NOT a trainer helper. A helper that computes gradients for the mechanism is part
    of the mechanism; asking it what the gradient is would be asking the mechanism to grade
    itself.
    """
    params = obs["encoder"]
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def probe_substitution(trainer, contract, name: str, seed: int = 1234,
                       probe_seed: int = 77) -> float:
    """Replace one declared loss term with a probe built on the SAME graph and see if the
    executed gradient moves.

    The probe carries its own non-zero gradient, so a term that is genuinely unused is
    distinguishable from one that is used. The probe is anchored on its own case's base loss: a
    probe anchored on another case's parameters would carry a zero gradient and would look like
    "unused" for every mechanism.

    Both executions get a FRESH case, because the mechanism's own assembly ends with an autograd
    call that frees the graph. Reusing one case would fail with "backward through the graph a
    second time" instead of answering the question this probe exists to answer.
    """
    reference = executed_target_gradient(
        trainer, contract, build_observations(trainer, contract, seed=seed))
    probe_obs = build_observations(trainer, contract, seed=seed)
    raw = torch.randn((), generator=torch.Generator().manual_seed(probe_seed))
    probe = probe_obs["base_loss"] * 3.7 + raw
    moved = executed_target_gradient(trainer, contract, probe_obs, substitute={name: probe})
    return float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(moved, reference))))


# --------------------------------------------------------------------------------------
# EMA priming
# --------------------------------------------------------------------------------------
def prime_ema(trainer, contract, prime_seed: int
              ) -> tuple[list[list[torch.Tensor]], dict[str, Any]]:
    """Run the declared priming steps and record the base gradients the recurrence consumes.

    The priming steps are executed through the mechanism's real assembly, which is what advances
    its internal state; the recurrence itself is then folded by the ORACLE from the observed
    series, never read out of the mechanism's own buffer. Returns the observed series and the
    last primed case.
    """
    ema = (contract.get("gradient_contract") or {}).get("ema") or {}
    steps = int(ema.get("prime_steps", 1))
    series: list[list[torch.Tensor]] = []
    primed_obs: dict[str, Any] = {}
    for index in range(steps):
        primed_obs = build_observations(trainer, contract, seed=prime_seed + index)
        terms = loss_terms(trainer, contract, primed_obs)
        series.append(term_gradient(trainer, primed_obs, terms["base_loss"]))
        executed_target_gradient(trainer, contract, primed_obs)
    return series, primed_obs


# --------------------------------------------------------------------------------------
# residual-trace mechanisms: the arm rewrites _residuals_with_trace itself
# --------------------------------------------------------------------------------------
TRACE_LOSS_SOURCES = ("behavior_l1", "behavior_l2", "behavior_l3", "behavior_other")


def trace_case(trainer, contract: dict[str, Any], seed: int = 1234,
               n_src: int | None = None, model=None) -> dict[str, Any]:
    """Run the arm's REAL residual trace and expose every level with its own graph.

    A trace mechanism changes ``_residuals_with_trace`` and nothing else: there is no separate
    projection step to call afterwards. So the case here is built around the trace itself, and
    the per-level contrastive terms are reconstructed from the trainer's OWN contrastive
    function, temperature and level set. Reusing the trainer's loss primitives matters: the
    terms the oracle reasons about are then literally the terms the training loop consumes,
    not a look-alike.
    """
    probe = (contract.get("model") or {}).get("probe") or {}
    n_src = int(n_src if n_src is not None else probe.get("n_src", N_SRC))
    in_dim = int(probe.get("in_dim", IN_DIM))
    if model is None:
        torch.manual_seed(seed)
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=in_dim)
    embeddings = torch.randn(2 * n_src, in_dim,
                             generator=torch.Generator().manual_seed(seed + 1))
    reconstructed, quant_loss, _, _ = model.forward(embeddings)
    base_loss, recon_loss = model.compute_loss(embeddings, reconstructed, quant_loss)

    executor_name = (contract.get("mechanism") or {}).get("executor")
    if not executor_name:
        raise ValueError("a trace contract must declare mechanism.executor")
    executor = resolve_dotted(trainer, executor_name)
    # The executor returns its levels STACKED, and two sibling slices of one stacked tensor have
    # no differentiable path between them: indexing forward is a select, and its backward
    # scatters back to the whole stack rather than reaching a sibling slice. Probing a sliced view
    # would therefore read as exactly zero for every level -- indistinguishable from the legacy
    # rule's real defect. So the levels are captured as the list the executor actually stacks.
    captured: list[list[torch.Tensor]] = []
    real_stack = torch.stack

    def recording_stack(tensors, *args, **kwargs):
        stacked_tensor = real_stack(tensors, *args, **kwargs)
        if (tensors and isinstance(tensors, (list, tuple))
                and getattr(stacked_tensor, "dim", lambda: 0)() == 3):
            captured.append([tensor for tensor in tensors])
        return stacked_tensor

    torch.stack = recording_stack
    try:
        stacked = executor(model, embeddings)
    finally:
        torch.stack = real_stack
    if captured:
        traces = captured[-1][:int(stacked.shape[0])]
    else:
        traces = [stacked[index] for index in range(int(stacked.shape[0]))]

    temperature = float(getattr(trainer, "BEHAVIOR_TEMPERATURE", 0.07))
    source_ids = torch.arange(n_src)
    target_ids = (torch.arange(n_src) + 1) % n_src
    level_losses = []
    for trace in traces:
        source, candidates = trace[:n_src], trace[n_src:]
        logits = -torch.cdist(source, candidates, p=2) / temperature
        loss = trainer._contrastive_ce(logits, source_ids, target_ids, None)
        if loss is None:
            raise ValueError("the declared contrastive term vanished on this case")
        level_losses.append(loss)

    codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]

    # The untouched control walk. The issues designate this function as out of scope for the
    # mechanism, so it is the reference the arm's forward is measured against. The last level
    # assigns codewords with Sinkhorn, whose choice an outside observer cannot re-derive from
    # distances alone; that is why the control exists, and why the oracle also verifies the hard
    # subtraction algebraically from the codebooks and the selected indices.
    reference_traces, reference_tokens = trainer._residuals_and_tokens_with_trace(model,
                                                                                 embeddings)

    observations: dict[str, Any] = {
        "trainer": trainer,
        "model": model,
        "seed": seed,
        "n_src": n_src,
        "embeddings": embeddings,
        "base_loss": base_loss,
        "recon_loss": recon_loss,
        "quant_loss": quant_loss,
        "traces": traces,
        "codebooks": codebooks,
        "level_losses": level_losses,
        "temperature": temperature,
        "residuals": stacked,
        "codebooks": codebooks,
        # the hard vectors the quantiser actually subtracted, selected from the raw codebooks by
        # the control walk's indices -- so the residual identity is checked against an
        # independent selection, not against the arm's own arithmetic
        "hard_quantized": [codebooks[level].detach()[reference_tokens[level]]
                           for level in range(int(reference_traces.shape[0]))],
        "reference_traces": [reference_traces[index]
                             for index in range(int(reference_traces.shape[0]))],
        "reference_tokens": reference_tokens,
        "trace_indices": [reference_tokens[index]
                          for index in range(int(reference_tokens.shape[0]))],
        "tensors": {
            "reconstructed": reconstructed,
            "quant_loss": quant_loss,
            "base_loss": base_loss,
            "recon_loss": recon_loss,
            "residuals": stacked.detach(),
        },
    }
    latent = model.encoder(embeddings)
    observations["latent"] = latent
    groups = (contract.get("model") or {}).get("parameter_groups") or {}
    target = (contract.get("gradient_contract") or {}).get("target", "encoder")
    observations["encoder"] = parameter_group(trainer, observations, contract, target)
    if "codebooks" in groups:
        observations["codebooks_parameters"] = parameter_group(trainer, observations, contract,
                                                              "codebooks")
    weight = float((contract.get("gradient_contract") or {}).get("probe_weight", 0.5))
    observations["scale"] = weight / max(len(traces), 1)
    observations["weight"] = weight
    with torch.no_grad():
        observations["tensors"]["tokens"] = model.get_indices(embeddings, infer_use_sk=False)
    return observations


def level_cotangents(trainer, obs: dict[str, Any]) -> list[torch.Tensor]:
    """d(level_loss)/d(level_trace) for each level, straight from the real graph.

    This is the quantity the mechanism's Jacobian acts on. Reading it from the trace's own graph
    is not the mechanism grading itself: the mechanism decides HOW the trace depends on the
    latent, and this is the other factor of the chain rule, which no mechanism controls.
    """
    return [torch.autograd.grad(loss, trace, retain_graph=True)[0]
            for loss, trace in zip(obs["level_losses"], obs["traces"])]


def latent_to_encoder(obs: dict[str, Any], cotangent: torch.Tensor) -> list[torch.Tensor]:
    """J_encoder^T u for a cotangent on the latent, via the model's own autograd.

    The encoder is not the mechanism under test; using autograd through it keeps the oracle's
    only dependence on the mechanism to the declared Jacobian coefficients.
    """
    latent = obs["latent"]
    grads = torch.autograd.grad((latent * cotangent).sum(), obs["encoder"],
                                retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(obs["encoder"], grads)]


def executed_trace_gradient(trainer, obs: dict[str, Any]) -> list[torch.Tensor]:
    """The gradient the arm's real trace produces for the declared level terms."""
    total = obs["level_losses"][0]
    for loss in obs["level_losses"][1:]:
        total = total + loss
    params = obs["encoder"]
    grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def apply_trace_gradient(trainer, obs: dict[str, Any]) -> None:
    """Write the arm's real trace gradient into ``.grad``, so a real optimizer step can follow.

    ``torch.autograd.grad`` returns gradients without touching ``.grad``, which is exactly what
    G3 wants (a value to compare) and exactly what G4 must not do (it needs a real optimizer to
    act on the same gradient the training loop would see).
    """
    total = obs["level_losses"][0]
    for loss in obs["level_losses"][1:]:
        total = total + loss
    obs["model"].zero_grad(set_to_none=False)
    total.backward()


def codebook_level_gradient(trainer, obs: dict[str, Any], level: int) -> list[torch.Tensor]:
    """The gradient one level's behaviour term sends to the codebooks.

    Measured, not assumed: whether the mechanism opens this path at all is precisely what the
    declaration has to be checked against.
    """
    codebooks = obs["codebooks_parameters"]
    if not codebooks:
        return []
    loss = obs["level_losses"][level]
    grads = torch.autograd.grad(loss, codebooks, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(codebooks, grads)]


def codebook_total_gradient(trainer, obs: dict[str, Any]) -> list[torch.Tensor]:
    """The gradient the arm's real trace sends to the codebooks, summed over levels."""
    codebooks = obs.get("codebooks_parameters")
    if not codebooks:
        return []
    total = obs["level_losses"][0]
    for loss in obs["level_losses"][1:]:
        total = total + loss
    grads = torch.autograd.grad(total, codebooks, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(codebooks, grads)]


def level_path_state(grads) -> str:
    """Classify a declared path as disconnected, exactly zero, non-zero, or non-finite.

    The three must not be conflated. A term that is not in the graph at all and a term whose
    gradient is exactly zero look identical in a norm, yet they mean opposite things: the first
    is a wiring bug, the second may be the correct mathematics.
    """
    flat = torch.cat([g.reshape(-1) for g in grads])
    if not bool(torch.isfinite(flat).all()):
        return "nonfinite"
    norm = float(torch.linalg.vector_norm(flat))
    if norm == 0.0:
        return "zero"
    return "nonzero"


def jacobian_probe(trainer, obs: dict[str, Any], level: int, seed: int = 31337) -> dict[str, Any]:
    """Measure the arm's OWN trace-to-latent Jacobian for one level.

    Pushes a random cotangent ``v`` back from the level's trace to the latent and reports the
    ratio and the signed alignment. For a declared rule ``J = c I`` both must equal ``c``, which
    is what rejects the neighbouring wrong rules: the legacy full-detach gives 0, the plain
    straight-through gives 1, and neither may pass as 0.5.
    """
    trace = obs["traces"][level]
    # Level 0 of the trace IS the encoder output, in the trace's own graph. Differentiating
    # against a separately constructed latent would cross no shared graph at all and every level
    # would measure as zero -- a disconnection that looks exactly like the legacy rule's bug.
    latent = obs["traces"][0]
    vector = torch.randn(trace.shape, generator=torch.Generator().manual_seed(seed))
    (pullback,) = torch.autograd.grad((trace * vector).sum(), latent, retain_graph=True,
                                      allow_unused=True)
    if pullback is None:
        return {"level": level, "connected": False, "ratio": 0.0, "alignment": 0.0}
    norm = float(torch.linalg.vector_norm(vector))
    ratio = float(torch.linalg.vector_norm(pullback)) / norm if norm else 0.0
    alignment = float((pullback * vector).sum()) / (norm * norm) if norm else 0.0
    return {"level": level, "connected": True, "ratio": ratio, "alignment": alignment}
