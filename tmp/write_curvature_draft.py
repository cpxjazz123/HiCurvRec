import json
from pathlib import Path

R = Path("results/research/issue121/rounds/round_20261106_020541_888")

# The Möbius sum of (-x) and y: the 2c<.,.> terms carry a MINUS because the
# first argument is -x. This is the sign the reviewers caught in the agent's
# draft, and it is the sign of the frozen round of 2026-11-05.
sympy_statements = [
    "xy = -(x0*y0 + x1*y1)",
    "xn = x0**2 + x1**2",
    "yn = y0**2 + y1**2",
    "num0 = (1 + 2*c*xy + c*yn)*(-x0) + (1 - c*xn)*y0",
    "num1 = (1 + 2*c*xy + c*yn)*(-x1) + (1 - c*xn)*y1",
    "den = 1 + 2*c*xy + c**2*xn*yn + 1e-12",
    "mx0 = num0/den",
    "mx1 = num1/den",
    "mn = mx0**2 + mx1**2",
    "d = 2/sqrt(c)*atanh(Min(sqrt(c)*sqrt(mn), 1 - 1e-6))",
]

executable = [
    "eps = 1e-12",
    "tau = 1.0",
    "rho = 0.5",
    "c = (rho / s) ** 2",
    "def _norm(v): return torch.sqrt(torch.clamp((v * v).sum(-1), min=0.0))",
    "def lift(v, c): return v if c == 0.0 else v * (torch.tanh(math.sqrt(c) * _norm(v)) / (math.sqrt(c) * _norm(v) + eps)).unsqueeze(-1)",
    "def mobius(x, y, c): return ((1 + 2 * c * (x * y).sum(-1, keepdim=True) + c * (y * y).sum(-1, keepdim=True)) * x + (1 - c * (x * x).sum(-1, keepdim=True)) * y) / (1 + 2 * c * (x * y).sum(-1, keepdim=True) + c * c * (x * x).sum(-1, keepdim=True) * (y * y).sum(-1, keepdim=True) + eps)",
    "def dist(x, y, c): return 2.0 * _norm(x - y) if c == 0.0 else (2.0 / math.sqrt(c)) * torch.atanh(torch.clamp(math.sqrt(c) * _norm(mobius(-x, y, c)), max=1.0 - 1e-6))",
    "s_k = torch.arange(K1).float() / (K1 - 1)          # the fixed code stratum, NOT a parameter",
    "u = lift(r, c)",
    "q = lift(a, c)",
    "D_k = dist(u[:, None, :], q[None, :, :], c)",
    "O_ik = beta * R_c * (1.0 - torch.abs(rho_i[:, None] - s_k[None, :]))   # the behaviour offset, in [0, beta*R_c]",
    "C_ik = D_k - O_ik",
    "P_ik = sinkhorn(torch.exp(-C_ik / tau), row=1.0, col=B / K1, iters=SK_ITERS, epsilon=SK_EPSILON)",
    "k1 = torch.argmax(P_ik, dim=-1)",
    "r_next = r - a[k1]",
]

draft = {
    "mechanism": (
        "The level-1 codewords are ORDERED by a train-only behaviour cohesion rank. "
        "Each item gets a bounded cohesion rank rho_i in (0,1) from the training "
        "split's undirected co-occurrence graph (how tightly its neighbourhood "
        "closes into triangles, mid-ranked). Each level-1 codeword k has a FIXED "
        "stratum s_k = k/(K-1) - it is a constant, not a parameter. The level-1 "
        "cost of an item at a codeword is the Poincare distance minus a bounded "
        "offset beta*R_c*(1 - |rho_i - s_k|), so an item is cheaper at the "
        "codewords whose stratum is close to its own rank. The codeword is chosen "
        "by the BASELINE's balanced assignment (a Sinkhorn plan over the cost "
        "matrix, then the plan's argmax), never by a plain argmin over the cost: "
        "removing the balancing is what collapsed the codebook in "
        "round_20261105_144909_130 (42 of 256 level-1 codes, L3 collision 0.703). "
        "Levels 2 and 3, the residual recursion, the reconstruction and the "
        "straight-through estimator are the baseline's, unchanged, and the "
        "reconstruction uses the UNLIFTED codeword."),
    "computations": [
        {"name": "cohesion statistic",
         "inputs": ["the training split's user sequences (history -> target)"],
         "output": "coh_i in [0,1] for every training item i",
         "definition": (
             "From the TRAINING split only: build the undirected co-occurrence "
             "count matrix A, A_ij = the number of times i and j are adjacent in "
             "a training sequence, i != j (self-adjacencies are dropped, so the "
             "diagonal is 0 by construction). deg_i = the number of distinct "
             "neighbours of i. N(i) = the D highest-count neighbours of i, "
             "D = ceil(mean_i deg_i), ties broken by the smaller item id. "
             "sim_ij = |N(i) cap N(j)| / sqrt(deg_i * deg_j) in [0,1]. "
             "coh_i = (1/|N(i)|) * sum over j in N(i) of sim_ij, in [0,1]. "
             "Every item with at least one training edge gets a value."),
         "note": "train split only; the graph is built once and frozen"},
        {"name": "cohesion rank",
         "inputs": ["coh_i for every item with at least one training edge"],
         "output": "rho_i in (0,1) for every training item",
         "definition": (
             "Let P = the set of training items with deg_i > 0, n_pos = |P|. "
             "For i in P: b_i = #{j in P : coh_j < coh_i} and "
             "t_i = #{j in P : coh_j = coh_i}; rho_i = (b_i + t_i/2 + 0.5) / n_pos, "
             "which lies in (0,1) and is the mid-rank the empirical CDF gives to "
             "the item. Both counts range over P and over P only. An item with "
             "deg_i = 0 has no cohesion and gets the neutral rho_i = 0.5; that "
             "value is a COMPLETION of the idea (which does not define the "
             "isolated items), it is stated here, and the mechanism is inert for "
             "those items (their offset is the same at every codeword only if "
             "s_k were 0.5, so it is NOT inert: the offset still orders their "
             "cost by |0.5 - s_k|)."),
         "note": "the mid-rank is over P, never over the whole catalogue"},
        {"name": "code stratum",
         "inputs": ["the number of level-1 codewords K1"],
         "output": "s_k in [0,1] for every level-1 codeword",
         "definition": (
             "s_k = k / (K1 - 1) for k = 0..K1-1, with K1 = 256 for this "
             "baseline. The stratum is a FIXED ORDERING of the codewords, not a "
             "learned parameter: the mechanism adds no trainable quantity, so "
             "there is no new gradient path and no new hyperparameter to tune. "
             "The codeword index is the stratum."),
         "note": "fixed and derivable: the codewords are read in index order"},
        {"name": "geometry",
         "inputs": ["the level-1 residual r", "the level-1 codebook a", "the curvature c"],
         "output": "D_k, the geodesic distance of every item to every codeword",
         "domain": "c > 0 for the curved arm; both arguments lifted, so c||x||^2 < 1",
         "definition": (
             "s = the RMS of the training-split latent vector norm at step 0, "
             "rho = 0.5 and c = (rho/s)^2 = 0.25/s^2, so R_c = s/rho = 2s. "
             "LIFT(v) = tanh(sqrt(c)||v||) v / (sqrt(c)||v|| + eps) with "
             "eps = 1e-12 and LIFT(0) = 0; MOBIUS is the ball sum with the "
             "2c<.,.> terms NEGATED on the first argument (-x), because the "
             "distance substitutes X = -x: "
             "(-x)(+)y = ((1 - 2c<x,y> + c||y||^2)(-x) + (1 - c||x||^2)y) / "
             "(1 - 2c<x,y> + c^2||x||^2||y||^2 + eps); "
             "DIST(x,y) = (2/sqrt(c)) artanh(min(sqrt(c)||(-x)(+)y||, 1 - 1e-6)). "
             "At c = 0 the lift is the identity and DIST(x,y) = 2||x - y||. "
             "The sign is the one the reviewers caught in the formula agent's "
             "draft: with +2c<.,.> the surface returns 0 at y = -x instead of "
             "the positive distance, and its c -> 0 limit is 2||x + y||."),
         "note": "the canonical ball at curvature -c; the same sign as the "
                 "frozen surface of round_20261105_144909_130"},
        {"name": "cost matrix",
         "inputs": ["D_k (B, K1)", "rho_i (B,)", "s_k (K1,)", "beta (scalar)", "R_c"],
         "output": "C_ik (B, K1)",
         "definition": (
             "O_ik = beta * R_c * (1 - |rho_i - s_k|), in [0, beta*R_c]; "
             "C_ik = D_k - O_ik. beta = median over the first batch of epoch 0 of "
             "the distance of every item to its chosen codeword, divided by R_c, "
             "measured once and frozen: the offset's scale is the measured "
             "level-1 distance scale, so no constant is tuned. R_c = s/rho."),
         "note": "the geodesic term and the behaviour term meet in ONE entry"},
        {"name": "balanced assignment",
         "inputs": ["C_ik (B, K1)"],
         "output": "P_ik (B, K1), the transport plan; k1 (B,)",
         "definition": (
             "P = Sinkhorn(exp(-C/tau), row mass 1 per item, column mass B/K1 per "
             "codeword, sk_iters and sk_epsilon as the baseline's), tau = 1. "
             "k1 = argmax_k P_ik, ties to the smallest k - the BASELINE's rule "
             "(its _indices). The balanced assignment is not replaced: the "
             "mechanism only changes the cost it balances."),
         "note": "the baseline's assignment rule, unchanged (CLAUDE.md 19h)"},
        {"name": "level-1 reconstruction and residual",
         "inputs": ["the chosen codeword a_{k1}", "the residual r"],
         "output": "the reconstruction of level 1 and r_next",
         "definition": (
             "q = a_{k1}, the UNLIFTED baseline codeword; r_next = r - a_{k1}. "
             "LIFT(a_{k1}) appears only inside the cost. The reconstruction, the "
             "commitment term and the straight-through estimator are the "
             "baseline's, so the encoder's gradient reaches the parameters "
             "through the STE on a_{k1} exactly once - the mechanism adds no "
             "parameter and no loss term, so this is the ONLY gradient path."),
         "note": "no new parameter, no new loss term"},
        {"name": "objective",
         "inputs": ["the baseline reconstruction and commitment terms"],
         "output": "the scalar loss the baseline optimiser minimises",
         "definition": (
             "loss = baseline_recon + baseline_commit, unchanged. The mechanism "
             "has no term of its own: it changes which codeword the balanced "
             "assignment picks, and the baseline objective then trains a "
             "different quantisation."),
         "note": "no auxiliary loss, no new optimiser"},
    ],
    "inputs": [
        "r, the residual entering level 1 (B, D) - the baseline's encoder output",
        "the level-1 codebook a (K1, D), the baseline's, learned",
        "rho_i (B,), the cohesion rank of the batch's items, a frozen buffer",
        "s_k (K1,), the fixed code stratum",
        "beta, measured once at epoch 0 and frozen; s and c, measured at step 0",
    ],
    "outputs": [
        "k1 (B,), the level-1 codeword index",
        "the diagnostics: the offset O_ik, the plan P_ik, the per-code rank mean",
    ],
    "parameters": [
        "rho = 0.5, the fixed dimensionless radius-to-scale ratio (R_c = s/rho)",
        "s = the training-split RMS of the per-item latent norm at step 0; c = (rho/s)^2",
        "beta = median_j D_j at epoch 0 divided by R_c, measured once and frozen",
        "s_k = k/(K1-1), the fixed code stratum (NOT learned)",
        "tau = 1, the softmax temperature of the plan; sk_iters and sk_epsilon as the baseline",
    ],
    "dimensions": {
        "r": "B x D", "a": "K1 x D", "u": "B x D", "q": "K1 x D",
        "D_k": "B x K1", "O_ik": "B x K1", "C_ik": "B x K1", "P_ik": "B x K1",
        "k1": "B", "rho_i": "B", "s_k": "K1", "beta": "scalar", "c": "scalar",
    },
    "gradient_rules": [
        {"operator": "the assignment argmax_k P_ik",
         "rule": "discrete, no gradient: it selects the codeword"},
        {"operator": "the encoder and the level-1 codebook",
         "rule": "the ONLY gradient path is the baseline's straight-through "
                 "estimator on the chosen codeword a_{k1}, through the baseline's "
                 "reconstruction and commitment terms. The cost matrix, the plan "
                 "and the offset are forward-only (detached), so they never "
                 "appear in a differentiable loss."},
        {"operator": "the mechanism's constants (c, s, beta, s_k, rho_i, tau)",
         "rule": "no gradient: they are constants or frozen buffers. There is no "
                 "new parameter, so no new gradient path, and encoder_gradient "
                 "must show the same 8 of 8 tensors as the baseline."},
    ],
    "training_integration": {
        "loss": "baseline_recon + baseline_commit, unchanged; no new term",
        "reconstruction": "the chosen codeword at its baseline position: a_{k1} "
                          "UNLIFTED, consistent with r_next = r - a_{k1}",
        "optimizer": "the baseline's, unchanged; no new parameter group",
    },
    "switches": [
        "BEHAVIOUR_OFF (present in code, never trained): the offset is set to 0 "
        "exactly, so C_ik = D_k; the Poincare distance and the balanced Sinkhorn "
        "assignment stay in the code path. The arm is plain hyperbolic-cost "
        "balanced quantisation with no behaviour information.",
        "GEOMETRY_OFF (present in code, never trained): c = 0, so LIFT is the "
        "identity and DIST(x,y) = 2||x - y||; the offset keeps the same scale "
        "R_c = s/rho as the curved arm, so the two arms differ in the geometry "
        "and in nothing else. This is the matched Euclidean control, and per the "
        "programme policy it is a separate arm in its own round, not an ablation "
        "of this one.",
    ],
    "excluded": [
        "any use of validation, test or future behaviour in rho_i",
        "any learned parameter of the mechanism (the stratum is fixed; there is no new loss)",
        "replacing the baseline's balanced assignment with a plain argmin (CLAUDE.md 19h)",
        "any sweep, ablation training run or seed repetition",
    ],
    "continuous_surface": {
        "callable": "distance(x, y, curvature)",
        "args": [
            {"name": "x", "width": 2, "radius": 0.4},
            {"name": "y", "width": 2, "radius": 0.4},
            {"name": "curvature", "scalar": 0.25},
        ],
        "symbols": ["x0", "x1", "y0", "y1", "c"],
        "statements": sympy_statements,
        "output": "d",
        "executable_statements": executable,
        "note": (
            "The statements are the DISTANCE in SymPy: the 2c<.,.> terms are "
            "negated because the distance substitutes X = -x, which is what "
            "makes the c -> 0 limit 2||x - y||. The executable_statements are "
            "the same surface as the trainer runs it, plus the stratum, the "
            "offset, the plan and the assignment."),
    },
    "incomplete": [],
    "missing": [],
}

(R / "math_spec_draft.json").write_text(json.dumps(draft, indent=1, ensure_ascii=False))
print("the corrected draft is written:", len(draft["computations"]), "computations,",
      len(sympy_statements), "sympy statements,", len(executable), "executable lines")
