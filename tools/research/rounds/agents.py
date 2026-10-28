"""Agent contracts for the Issue #120 round: ideate, score, implement, verify, plan.

The prompts state what each role must return and what it must not do. The
verification role is a correctness reviewer of an implementation, not a scientific
paper referee: the issue is explicit that scientific paperwork completeness is
not a pre-training gate, while actual correctness, numerical stability and a
runnable trainer are. The repair role is a **planner**: it writes the change
that would fix a finding and never edits the tree, because the operator applies
the change and owns the code.
"""
from __future__ import annotations


#: The research objective of Issue #121, hard coded into every role. It is not
#: an optional free-text objective: the ideas, the scores, the implementations
#: and the reviews are all about this one coupling.
RESEARCH_OBJECTIVE = (
    "Discover new mechanisms in which TRAINING-ONLY user behaviour and HYPERBOLIC "
    "geometry are computationally coupled to improve the RQ-VAE Semantic IDs used by "
    "a generative next-item recommender, with the real target of Stage3 test "
    "Recall@10 above 0.065."
)

#: The scope contract every proposal has to meet. It is a research-scope gate,
#: not a proof: what matters is that the two sides are used together.
SCOPE_CONTRACT = (
    "A proposal is IN SCOPE only when all of this holds:\n"
    "  1. BEHAVIOUR: it names one training-only behaviour signal (directed item "
    "transitions, user-interest branching, sequential co-consumption, "
    "popularity-adjusted graph neighbours, behaviour-conditioned assignment "
    "weights, ...) and says how it is computed.\n"
    "  2. HYPERBOLIC GEOMETRY: it names an explicit curved operation (trainable or "
    "fixed negative curvature, Poincare or Lorentz distance, exp/log maps, geodesic "
    "scoring, curved residual, curved quantization, curved regularization) and where "
    "in training or quantization it acts.\n"
    "  3. REAL COUPLING: one connected computational pathway uses both, so that "
    "removing either changes the core computation. A behaviour suffix on an "
    "unchanged Euclidean quantizer, a standalone hyperbolic loss beside an unrelated "
    "behaviour loss, a geometry name without geometric computation, or inherited "
    "unused hyperbolic code do NOT qualify.\n"
    "  4. TESTABLE EFFECT: it states what should observably change (SID assignment, "
    "gradient or parameter update, prefix behaviour predictability, or Stage3 "
    "Recall) and names the behaviour-off, geometry-off/Euclidean and both-on "
    "comparisons. The three switches must be part of the mechanism, so that the "
    "coupling can be shown to be real, but the round RUNS no ablation arm: each "
    "candidate is trained once.\n"
    "  5. MINIMAL SPECIFICATION: equations or pseudocode, inputs and outputs, "
    "numerical safeguards and the training cost. No proof and no paper section are "
    "required.\n"
    "SID quality (ER, prefix counts, collisions) is NOT a scope criterion and NOT an "
    "admission rule."
)

#: The experiment policy of the programme, stated to every role.
EXPERIMENT_POLICY = (
    "EXPERIMENT POLICY (binding on every proposal, implementation and review):\n"
    "  - ONE candidate, ONE training run, on the frozen budget (3000 Stage 2 epochs "
    "on four cards, the frozen Stage 3 protocol, no early stopping).\n"
    "  - NO ablation experiments: the behaviour-off, geometry-off and both-on "
    "switches must EXIST in the code and must be correct, but no arm is trained "
    "with them switched off.\n"
    "  - NO tuning experiments: no sweep over the coupling weight, the clip bound, "
    "the temperature, the usage weight or any other constant. A constant that "
    "enters the mechanism has to be DERIVED (a scale-relative bound, a measured "
    "quantity, a documented default), not chosen by trying values.\n"
    "  - NO multi-seed experiments: a candidate is not repeated with other seeds, "
    "so a single run decides it and the seed noise is part of the uncertainty.\n"
    "  - The behavioural diagnostics (coherence, adjacency, transfer predictability) "
    "are recorded, never used as an admission gate; Stage 3 test_recall@10 is the "
    "judgement.\n"
    "  - A proposal that needs an ablation, a sweep or a seed repetition to make "
    "its case is out of programme: it has to make the case with one run."
)

IDEA_FIELDS = (
    "idea_id, title, motivation, novelty, key_change, specification, cost_estimate, "
    "falsifiable_hypothesis, risks"
)

IDEA_PROMPT = (
    "Propose exactly {minimum} DISTINCT, implementable idea for improving the "
    "item semantic IDs that a generative recommender consumes. You are one of "
    "{agents} independent ideation agents: aim for a different part of the design "
    "space than the others, and do not copy an existing mechanism in the "
    "repository.\n"
    "THE RESEARCH OBJECTIVE:\n{objective}\n"
    "THE SCOPE CONTRACT (a proposal that misses it is sent back, not scored):\n"
    "{scope}\n"
    "Explore materially different coupling positions, for example: behaviour-weighted "
    "curved assignment or Sinkhorn; behaviour hierarchy in a negatively curved latent "
    "or quantizer; behaviour-directed curved residual or transition optimization. "
    "These are examples, not a fixed list.\n"
    "You have no write permission: print your answer in the response itself.\n"
    "Write PLAIN TEXT, one block per idea, in exactly this shape:\n"
    "IDEA: <short safe identifier, unique in your answer>\n"
    "TITLE: <one line>\n"
    "MOTIVATION: <the user-behaviour or recommendation problem it addresses>\n"
    "NOVELTY: <what is different from the mechanisms that already exist here>\n"
    "CHANGE: <the computational change, in one or two sentences>\n"
    "SPECIFICATION: <the minimal mathematics or pseudocode an implementer can write "
    "without guessing>\n"
    "COST: <the compute or wall-clock cost relative to the baseline, and the per-step "
    "cost at the FROZEN scale: 24556 training items, 24 steps per epoch, 3000 epochs>\n"
    "HYPOTHESIS: <what must be true for the idea to be worth keeping, and how it would "
    "be observed>\n"
    "RISKS: <what could make it fail>\n"
    "BEHAVIOR SIGNAL: <the training-only behaviour signal you use>\n"
    "BEHAVIOR COMPUTATION: <how that signal is computed, from which split>\n"
    "LEAKAGE BOUNDARY: <why no validation or test interaction can enter the SID "
    "construction>\n"
    "HYPERBOLIC OPERATOR: <the curved operation, with its curvature>\n"
    "HYPERBOLIC LOCATION: <where in training or quantization it acts>\n"
    "COUPLING PATH: <the single connected pathway in which the behaviour signal and "
    "the curved operation meet, so that removing either changes the computation>\n"
    "COUPLING EQUATION: <the equation of that pathway, naming the behaviour term and "
    "the curved term explicitly, for example w_behaviour(x) * d_poincare(v, c)>\n"
    "OBSERVABLES: <what should observably change when the coupling is on>\n"
    "ABLATION BEHAVIOR OFF: <the run with the behaviour signal switched off>\n"
    "ABLATION GEOMETRY OFF: <the run with the curved operation replaced by the "
    "Euclidean one>\n"
    "ABLATION BOTH ON: <the full coupled run>\n"
    "SPEC EQUATIONS: <the equations or pseudocode to implement>\n"
    "SPEC INPUTS OUTPUTS: <what the operator takes and what it returns>\n"
    "SAFEGUARDS: <the numerical safeguards you need>\n"
    "END IDEA\n"
    "A value may run over several lines; the next line that starts with one of the "
    "labels above starts the next value. Unconventional ideas and unfinished "
    "derivations are welcome: only reject an idea that cannot be made minimally "
    "specified."
)
RANK_PROMPT = (
    "This exploration produces ONE mechanism. You are one of three independent reviewers; "
    "you see the same frozen evidence bundle and the same proposals as the others, and you "
    "do not coordinate with them before handing in.\n"
    "THE RESEARCH OBJECTIVE:\n{objective}\n"
    "THE SCOPE CONTRACT:\n{scope}\n"
    "THE SELECTION POLICY ({rubric_version}): the reviewers RANK the proposals. The "
    "proposal the reviewers place best, in total, is the round's mechanism. Rank every "
    "proposal, best first, and give each one a reason that names the evidence you used.\n"
    "Read the frozen evidence bundle for each proposal before you place it: a placement "
    "you cannot ground in the evidence is not a placement.\n"
    "Write PLAIN TEXT, one line per proposal, in exactly this shape:\n"
    "RANK: <idea_id> | position=<1 for the best, then 2, 3, ...> | reason=<one line>\n"
    "RANK: <idea_id> | position=<n> | reason=<one line>\n"
    "Every proposal must appear exactly once, the positions must be 1..N with no ties and "
    "no gaps, and the reason must say what in the evidence put the proposal there. If you "
    "cannot ground a proposal in the evidence, rank it last and say so in its reason; do "
    "not leave a proposal out.\n"
    "THE PROPOSALS:\n{proposals}\n"
    "THE FROZEN EVIDENCE BUNDLE:\n{evidence}"
)

EVIDENCE_REVISION_PROMPT = (
    "The evidence probe measured the following about your proposal, and it could not "
    "establish these points:\n{gaps}\n"
    "Your proposal:\n{proposal}\n"
    "You have ONE bounded revision of the proposal (not of any code): answer the specific "
    "gaps with what the evidence shows, or state that the evidence does not support the "
    "claim. Do not add a mechanism, do not change the coupling, and do not promise a "
    "Recall. If the evidence contradicts the proposal, say so.\n"
    "You MAY write: implement the mechanism in the candidate tree, run the smoke "
    "test, and write the answer file the runner reads.\n"
    "Write PLAIN TEXT, in exactly this shape:\n"
    "REVISION: <what the proposal now claims, or what it withdraws> | <which gap it "
    "answers>\n"
    "WITHDRAWN: <the claim you withdraw, or NONE>\n"
    "EVIDENCE NOTE: <the measured numbers you relied on, or UNKNOWN>"
)

REBUTTAL_PROMPT = (
    "One of three reviewers raised this objection against your proposal:\n{objection}\n"
    "with this counterexample, when one was given:\n{counterexample}\n"
    "You have ONE bounded revision of the proposal (not of any code): either repair the "
    "specification so the objection no longer holds, or state precisely why the objection is "
    "wrong. Do not change the mechanism to a different one; do not add scores.\n"
    "You have no write permission: print your answer in the response itself.\n"
    "Write PLAIN TEXT, in exactly this shape:\n"
    "CHANGE: <what you changed in the proposal> | <why the objection no longer applies>\n"
    "UNCHANGED: <the part of the proposal you deliberately kept>\n"
    "REMAINING: <the objection you accept and cannot repair>"
)

IMPLEMENT_PROMPT = (
    "The experiment policy binds you: ONE candidate, ONE training run on the frozen "
    "budget. Do not train ablation arms, do not sweep any constant and do not repeat "
    "the run with another seed. The behaviour-off, geometry-off and both-on switches "
    "must exist in the code and be correct, and a cheap unit-level check that the "
    "switch is wired is welcome, but no experiment with them switched off.\n"
    "Implement ONE idea as a standalone Stage 2 candidate tree.\n"
    "THE RESEARCH OBJECTIVE:\n{objective}\n"
    "THE SCOPE CONTRACT:\n{scope}\n"
    "The idea:\n{idea}\n"
    "YOUR TREE IS THIS EXACT DIRECTORY, and every file you create must live inside "
    "it:\n{tree}\n"
    "Do not create a tree anywhere else, and do not write your implementation into "
    "your own workspace. The framework reads the files in that directory and ignores "
    "everything you say about them.\n"
    "Requirements:\n"
    "  1. The tree you were given is already a fresh copy of the frozen baseline "
    "TIGER_RQ-VAE. Change only what the idea needs, and keep the baseline's frozen "
    "training, data and architecture configuration: latent 32, three levels of 256 "
    "codes, 3000 epochs, the existing optimizer, schedule, batch size, seeds, the data "
    "split and the exporter.\n"
    "  2. The coupling must actually run: the behaviour signal and the curved "
    "operation must meet in one computational pathway that the training loop calls, "
    "the pathway must change the SID assignment or the gradients when it is on, and "
    "switching the behaviour off or replacing the curved operation with the Euclidean "
    "one must change that computation. Do not merely import a hyperbolic helper.\n"
    "  3. The training must run: a forward and a backward, finite losses, gradients "
    "that reach the parameters the idea changes, a valid SID export, and a minimal "
    "smoke test that proves it. Run Python with the project interpreter "
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9, which is the one "
    "that has PyTorch; the system python3 does not, and a ModuleNotFoundError from it "
    "is not a defect of this tree.\n"
    "  4. The trainer must be able to run on ONE GPU: keep the frozen trainer's "
    "single-card branch, which reads the environment variable "
    "SINGLE_GPU_ENV = \"GENREC_SINGLE_GPU\", and do not replace the launcher with an "
    "unconditional four-rank torchrun.\n"
    "  5. NO ABSOLUTE PATH MAY APPEAR ANYWHERE in the trainer, in any constant. Every "
    "path is derived from the repository root, and the repository root is found by "
    "walking up from this file to the directory that contains stage2_RQ-VAE/ "
    "(the baseline tree already has that helper: keep it). The trainer writes into "
    "REPO_ROOT / \"results\" / \"stage2_RQ-VAE\" / ARM_NAME, where ARM_NAME is "
    "\"TIGER_<the idea id above>\" as a single constant; OUTPUT_SID, OUTPUT_JSON, "
    "CHECKPOINT, METRICS_PATH and LOG_DIR all derive from it. Never "
    "Path(__file__).parents[N]: this tree is copied into stage2_RQ-VAE/ before it "
    "trains, so a parent count is correct here and wrong there. Your smoke test must "
    "pass in that promoted location, not only in this one. OUTPUT_SID, OUTPUT_JSON, CHECKPOINT, "
    "METRICS_PATH and LOG_DIR must all derive from it. The baseline tree "
    "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_RQ-VAE and its result "
    "directory are read-only: never write into them, never into the source tree and "
    "never into another candidate's directory.\n"
    "  6. Record every deviation from the idea and every assumption you had to make, "
    "with the reason.\n"
    "  7. Do not change the shared repository outside your own tree. Do not modify the "
    "baseline trees, the MCP server or the verification code.\n"
    "You MAY write: implement the mechanism in the candidate tree, run the smoke "
    "test, and write the answer file the runner reads.\n"
    "Write PLAIN TEXT about what you did, in exactly this shape:\n"
    "TREE: <the absolute path of the tree you wrote>\n"
    "CHANGED FILE: <one path relative to the tree, one line per file>\n"
    "DEVIATION: <what you changed from the idea> | <why>\n"
    "ASSUMPTION: <one assumption you had to make>\n"
    "SMOKE COMMAND: <the command you ran>\n"
    "SMOKE RESULT: <what it printed>\n"
    "NOTES: <anything the reviewers should know>"
)

#: The five verification roles. Each reviewer still reads the whole tree, but one
#: area is its primary responsibility and it must report on that area's checks.
#: Five identical full-scope reviews are a coverage lottery: what each one finds
#: depends on where it happens to look. The roles make the union of the five
#: reviews cover the five classes of defect that actually killed candidates --
#: geometry, gradients, distributed state, behaviour/SID consistency, integration.
VERIFIER_ROLES = (
    {
        "id": "math-geometry",
        "reads": ["model/frechet_poincare.py", "model/curvature_control.py",
                  "model/utils.py", "tests/test_geometry.py",
                  "tests/test_codebook_step.py"],
        "name": "mathematics and geometry",
        "scope": "model/frechet_poincare.py, tests/test_geometry.py, "
                 "tests/test_codebook_step.py",
        "must": (
            "every map (exp, log, Mobius sum, distance, Frechet mean, projection) "
            "has the formula its name implies and the domain it claims; check the "
            "identities numerically on paper (0 (+) y = y, exp_x(log_x(y)) = y, "
            "symmetry, distance against the closed form on a geodesic, the midpoint "
            "halving the arc), and check the sign of every term",
            "the curvature convention is one convention: which c, which ball radius, "
            "and no place that silently uses the Euclidean distance inside a curved "
            "path or the reverse",
            "numerical degeneracy: what happens at the boundary, at the origin, to a "
            "zero tangent, to a huge latent (saturation of tanh/atanh), and whether a "
            "clamp removes information or a gradient",
            "the weighting: is the weighted mean the one the proposal defines, is the "
            "normalisation the stated one, and does the weight of an item follow from "
            "the training data rather than from a constant",
        ),
        "checks": (
            ("mathematics and geometry: formulas, domains and identities",
             "report the identity you checked and the numbers you checked it with"),
            ("numerical behaviour at the degenerate points",
             "boundary, origin, zero tangent, saturation, clamp behaviour"),
        ),
    },
    {
        "id": "gradient-update",
        "reads": ["model/layers.py", "model/model.py", "model/curvature_control.py",
                  "model/frechet_poincare.py", "model/utils.py",
                  "train_rqvae.py", "tests/test_mechanism.py",
                  "tests/test_curvature_wiring.py"],
        "name": "gradients and parameter updates",
        "scope": "model/layers.py, model/model.py, train_rqvae.py (the optimiser, the "
                 "backward and the parameter updates), tests/test_mechanism.py",
        "must": (
            "for every parameter: which objective updates it, and does that match the "
            "proposal; a parameter that is in the optimiser but never receives a "
            "gradient, or that is trained by two estimators at once, is a blocking "
            "defect",
            "detach, straight-through and stop-gradient placements: exactly where the "
            "gradient is cut, and whether the cut still leaves the encoder the path the "
            "proposal wants",
            "the loss composition: every additive term is used, has the stated weight, "
            "and none cancels another",
            "gradient scale: could a term saturate, vanish or explode at the frozen "
            "learning rate and clip? Check the derivative of the curved term near the "
            "boundary rather than a single forward value",
        ),
        "checks": (
            ("gradients: which parameter is updated by which objective",
             "name the parameter, the objective and the code path"),
            ("detach, STE and the loss composition",
             "every cut, every additive term, every weight"),
        ),
    },
    {
        "id": "distributed-state",
        "reads": ["train_rqvae.py", "model/model.py", "model/layers.py",
                  "model/curvature_control.py", "model/frechet_poincare.py",
                  "tests/test_ddp.py"],
        "name": "distributed execution and shared state",
        "scope": "train_rqvae.py (DDP construction, the step loop, collectives), "
                 "model/model.py (the codebook step plumbing), tests/test_ddp.py",
        "must": (
            "DDP: is every parameter that requires grad reduced on every step? A "
            "parameter that is in the module but outside the loss graph fails the "
            "reducer on the second step with find_unused_parameters=False; a manual "
            ".data write is not broadcast by DDP",
            "collectives: same order and same count on every rank, no branch that only "
            "one rank takes, no collective inside a conditional that can differ",
            "shared state: a codebook, cache, counter or buffer initialised once must "
            "be identical on every rank, and a step that writes it must be reduced "
            "before it is used",
            "the single-process branch and the four-rank branch must both be reachable "
            "and must not disagree about what they compute",
        ),
        "checks": (
            ("distributed execution: reduction, collectives and rank symmetry",
             "name the collective and the step where it runs"),
            ("shared state: codebook, cache and counters across ranks",
             "how the state is initialised and how it is kept equal"),
        ),
    },
    {
        "id": "behaviour-sid",
        "reads": ["train_rqvae.py", "model/layers.py", "model/curvature_control.py",
                  "model/frechet_poincare.py", "tests/test_mechanism.py"],
        "name": "behaviour signal and SID consistency",
        "scope": "train_rqvae.py (the behaviour statistics, the export), "
                 "model/layers.py (the curved assignment), model/model.py",
        "must": (
            "data boundary: the behaviour signal is built from the TRAINING split only; "
            "no validation or test interaction, no future behaviour of the target user, "
            "no statistic that would be unavailable at prediction time",
            "the pathway: the behaviour signal and the curved operation meet in one "
            "connected computation that decides the assignment; a behaviour suffix on an "
            "unchanged Euclidean quantizer, or two unrelated losses, do not count",
            "training/export consistency: the SID exported for an item must be the code "
            "the trained quantizer assigns to it, computed the same way for every "
            "item, the same for every user and every request, with the deterministic "
            "disambiguation token kept",
            "the two switches: with the behaviour off and with the curve off the "
            "mechanism must still be the same mechanism with that one element replaced, "
            "not the untouched baseline",
        ),
        "checks": (
            ("behaviour signal: boundary, statistics and the connected pathway",
             "name the exercised path file:line and the split the statistic comes from"),
            ("SID consistency: training, export and global uniqueness",
             "how the export is computed and why it cannot differ per rank or per user"),
        ),
    },
    {
        "id": "integration-resources",
        "reads": ["train_rqvae.py", "model/utils.py", "model/curvature_control.py",
                  "tests/test_paths.py", "."],
        "name": "integration and resources",
        "scope": "train_rqvae.py (entry point, paths, run configuration, epoch loop), "
                 "model/utils.py, tests/test_paths.py",
        "must": (
            "the real entry point runs with the frozen configuration: epochs, batch, "
            "learning rate, seed, and the launch contract (the framework owns torchrun, "
            "the ports and the working directory; the candidate must not fork them)",
            "paths: the repository root comes from an unambiguous marker and not from a "
            "parent count; the result directory is the candidate's own; nothing is "
            "written into the baseline, another candidate, or outside the repository",
            "cost: the per-step work at the frozen scale (items, codes, steps per epoch, "
            "3000 epochs) must fit the budget; a Python loop over slots with a device "
            "sync per slot is a blocking performance defect",
            "interfaces: the modules still expose what the trainer and the framework "
            "call, and a changed signature has one caller shape only",
        ),
        "checks": (
            ("integration: entry point, frozen configuration and launch contract",
             "name the constant you checked and the line it lives on"),
            ("paths, cost and interfaces",
             "the path resolution, the per-step cost and the module interfaces"),
        ),
    },
)


VERIFY_PROMPT = (
    "You are verifier {role_index} of {role_total} on an implementation. Your primary "
    "responsibility is {role_name}; the other four reviewers own the other areas, so "
    "spend your effort here and go deep rather than broad. Inspect the code, do not "
    "trust a passing test.\n"
    "YOUR AREA (what you must cover):\n{role_must}\n"
    "YOUR PRIMARY READ SCOPE: {role_scope}. Read anything else you need, and report a "
    "defect you find outside your area -- the division only guarantees coverage, it "
    "does not forbid a cross-area finding.\n"
    "THE CHECKS YOU MUST REPORT, one CHECK block each:\n{role_checks}\n"
    "THE RESEARCH OBJECTIVE:\n{objective}\n"
    "THE SCOPE CONTRACT:\n{scope}\n"
    "The idea it implements:\n{idea}\n"
    "The tree:\n{tree}\n"
    "WHAT CHANGED SINCE THE LAST REVIEW (the main agent's brief; verify it, do not "
    "trust it):\n{change_brief}\n"
    "THE CODE THAT CHANGED SINCE THE LAST REVIEWED REVISION (a unified diff; read it "
    "together with the files in your scope):\n{change_diff}\n"
    "THE PREVIOUS REVIEW'S FINDINGS (are they actually fixed? A finding whose defect "
    "is still present is still blocking, and a fix that introduced a new defect is a "
    "finding of this review; the complete list is in {findings_file}):\n"
    "{previous_findings}\n"
    "Every area is still in scope for judging whether the candidate may train; the "
    "three questions the whole panel answers are:\n"
    "  a) behaviour x geometry fidelity: is there ONE connected pathway in which the "
    "training-only behaviour signal and the curved operation are used together? Name "
    "the exercised path (file and line) and say how you exercised it. An imported but "
    "unused hyperbolic helper, a behaviour suffix on an unchanged Euclidean quantizer, "
    "or two unrelated losses do not count.\n"
    "  b) trainability: does a real forward and backward produce finite losses and "
    "gradients that reach the intended parameters, with no dead term and no "
    "numerically unstable step? Does the coupled pathway change the SID assignment or "
    "the gradients when it is switched on, and change back when the behaviour is "
    "switched off or the curve is replaced by the Euclidean distance?\n"
    "  c) integration: does the real trainer entry point run, is the data path the "
    "frozen one, is the SID export valid and correctly indexed, does the trainer write "
    "only into its own result directory, and does the distributed launcher still work, "
    "including the single-card branch that reads "
    "SINGLE_GPU_ENV = \"GENREC_SINGLE_GPU\"?\n"
    "If the proposal is an ABLATION CONTROL (the idea says so), the coupling is "
    "intentionally switched off. Then judge three things instead: exactly the named "
    "switch is off, everything else is unchanged from the mechanism it derives from, "
    "and it still trains. Do not report the missing coupling as a defect.\n"
    "Report PASS or NEEDS_FIX per check. A NEEDS_FIX must name the defect, give a "
    "concrete reproduction or a minimal counterexample, say how severe it is and what "
    "the correction is. Do not demand a proof, a literature review or a quality floor: "
    "this is a correctness review of code, not a scientific paper review.\n"
    "A NEEDS_FIX without a finding is not a report: the framework treats a verdict "
    "that names no defect as a blocking finding in its own right, and a reviewer that "
    "cannot name one has to say PASS for that check instead.\n"
    "Two things are NOT defects and must not be reported as blocking findings:\n"
    "  - This sandbox has no PyTorch and no shell, so you cannot run the trainer. "
    "Judge trainability from the code: a differentiable path for the coupled term, no "
    "dead term that is computed and then unused, and no numerical hazard such as a "
    "clamp that overflows to inf or a nan in float32. The executed runtime proof is "
    "the MCP one-epoch gradient smoke, which runs the real trainer on a GPU before any "
    "full run; you do not have to provide it.\n"
    "  - The candidate's result directory IS results/stage2_RQ-VAE/<the promoted tree "
    "name>/, because that is where the admission and the Stage 3 trainer read "
    "item_sids.json. Writing there is correct and required. The defect is writing into "
    "the baseline's directory or into another candidate's directory, or resolving to a "
    "path outside the repository. Do not ask for a result directory inside the "
    "candidate tree.\n"
    "You have no write permission: print your answer in the response itself.\n"
    "Write PLAIN TEXT, in exactly this shape:\n"
    "VERDICT: PASS or NEEDS_FIX\n"
    "CHECK: <the check you are reporting>\n"
    "CHECK VERDICT: PASS or NEEDS_FIX\n"
    "EXERCISED PATH: <file:line> | <how you exercised it>\n"
    "FINDING SEVERITY: <blocking or minor>\n"
    "FINDING LOCATION: <file:line>\n"
    "FINDING DETAIL: <what is wrong>\n"
    "FINDING REPRODUCTION: <how to see it>\n"
    "FINDING CORRECTION: <what the correction is>\n"
    "Repeat the CHECK block for every check, and the finding lines for every finding.\n"
    "The severity decides whether the candidate may train, so it is not a matter "
    "of taste. Write blocking when the candidate does not do what the frozen "
    "proposal says it does, when the behaviour signal or the curved operation "
    "does not reach the computation it claims, when a gradient does not reach "
    "the parameter the proposal says it reaches, or when the implementation "
    "cannot run at all. Write minor only for a defect that changes no "
    "computation of the mechanism, such as an unused declaration or a duplicated "
    "import. If the overall verdict is NEEDS_FIX, at least one finding is "
    "blocking; a NEEDS_FIX with only minor findings is read as a contradiction "
    "of the proposal and blocks the candidate in any case.\n"
)

REPAIR_PLAN_PROMPT = (
    "You are the repair PLANNER for one implementation. The verifiers rejected it with "
    "these findings:\n{findings}\n"
    "THE RESEARCH OBJECTIVE:\n{objective}\n"
    "THE SCOPE CONTRACT:\n{scope}\n"
    "The candidate tree is:\n{tree}\n"
    "YOU MUST NOT MODIFY ANY FILE, AND YOU HAVE NO WRITE PERMISSION AT ALL: your session "
    "denies the edit and the write tools, so any attempt to change the tree is impossible "
    "and would be rolled back if it somehow happened. You do not repair the code: you plan "
    "the repair and the operator applies it. Read the tree, work out the exact change for "
    "every reported finding, and print the plan.\n"
    "For every finding give the file and the line, the change in enough detail that it can "
    "be applied without guessing (the code to replace and the code to write), and the check "
    "that proves the fix (the command to run and what it must print). Keep the frozen "
    "protocol and everything the idea requires that is not reported as defective.\n"
    "Three requirements hold whatever the findings say: the behaviour signal and the "
    "curved operation must meet in one connected pathway that the training loop "
    "calls; the trainer must be able to run on ONE GPU through the frozen single-card "
    "branch (SINGLE_GPU_ENV = \"GENREC_SINGLE_GPU\"); and it must contain NO absolute "
    "path, deriving every path from the repository root found by walking up to the "
    "stage2_RQ-VAE marker, and writing its artifacts into "
    "REPO_ROOT / \"results\" / \"stage2_RQ-VAE\" / ARM_NAME, never into the baseline "
    "tree, the baseline's result directory or another candidate's directory.\n"
    "You have no write permission: print your answer in the response itself.\n"
    "Write PLAIN TEXT, in exactly this shape:\n"
    "PLAN: <the finding> | <file:line> | <the exact change> | <the check that proves it>\n"
    "REMAINING ISSUE: <one thing you could not plan a fix for>\n"
    "NOTES: <anything the operator should know>"
)
