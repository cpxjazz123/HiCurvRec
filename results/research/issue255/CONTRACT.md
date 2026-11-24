# Issue255 frozen mechanism contract

Frozen before implementation and before any formal run. Source of truth: issue #255 body SHA256 `2fa23b1ee2faa897155f2a12999b3cf4cd7c085a5fa0a5f8a99c53f54b1c361d`. Implementation may only obey this contract; any deviation must be written here first with a date and a reason.

## 1. Catalogue, codebooks and levels

`N = 24587` items, global zero-based item ids (`assets/train/item_ids.npy` must equal `arange(N)`). Input 768-d, latent 32-d, three levels of 256 codewords, plus the untouched fourth collision counter used only by the Stage3 codec. Level `l ∈ {1,2,3}` has codebook `e_{l,j} ∈ R^32` and the hard selected code `e_{l,i}` of item `i`.

## 2. Direction-aware ego tokens (four types)

For every item `i` the token population is the union of four induced subgraphs of the **frozen training-visible directed graph** (`assets/train/visible_forward.npz`, sha256 `ac7155d1…`):

| Token | Node set | Edges kept |
|---|---|---|
| T1 one-hop | `{i} ∪ N_in(i) ∪ N_out(i)` | all real directed edges with both endpoints inside the set |
| T2 two-hop | the frozen complete radius-two population `{i} ∪ N2(i)` | all real directed edges with both endpoints inside the set |
| T3 in-only | `{i} ∪ N_in(i)` | all real directed edges with both endpoints inside the set |
| T4 out-only | `{i} ∪ N_out(i)` | all real directed edges with both endpoints inside the set |

Rules: the full 1-hop and 2-hop populations are never pruned or truncated; T3 and T4 are **not** claimed to be mutually exclusive (an item can be both an in- and an out-neighbour, and reciprocal edges appear in both token edge sets); a node with an empty neighbourhood yields a single-node token (the root only) and a zero neighbourhood contribution, never a fabricated neighbour; the root is always present exactly once per token; the same global item identity is used in every token and in every table; no test-derived relation and no #254 error list may enter any feature. Token type is encoded by a fixed per-type constant, and the per-token validity mask is explicit.

## 3. Layer-wise three-curvature gating

One shared directed GNN/readout produces one token vector per type: `H_i ∈ R^{4×32}` (plus the per-token mask). Per level `l` the mechanism owns a trainable query (equivalently a layer-conditioned cross-attention over the shared K/V and score MLP) and computes

`w_i^(l) = softmax_k( score(H_i, l, k) )`, `k ∈ {-1,0,+1}`, shape `[N, 3 levels, 3 experts]`.

Normalisation: the attention over the token dimension is normalised per token; the geometry weights are normalised over the expert dimension per (item, level), so `Σ_k w_i^(l,k) = 1`. No layer is hand-assigned a curvature: each level learns its own weights, and the effective per-level weight is `w_i^(l)`.

Per-level mixed cost for the codeword `e_{l,j}` of item `i`:

`cost_i^(l)(j) = Σ_k w_i^(l,k) · d_k²( r_i^(l), e_{l,j} )`,

where `r_i^(l)` is the level-`l` representation (the same encoder output the level-`l` codebook sees) and `d_k` is the frozen ambient squared distance for curvature `k`. The actual hard code selection and the existing surrogate backward are unchanged in structure; no neighbour-distance term is added.

## 4. Cumulative prefix and the sectional proxy

Hard prefix: `q_i^(l) = Σ_{j≤l} e_{i,j}`. The hard forward must be bitwise identical to the sum of the first `l` selected codewords, i.e. the actual SID's first `l` codes.

Clarification recorded 2026-11-25 before the formal run: the prefix value used by the loss is the selected value whose FORWARD is bitwise the hard codeword and whose BACKWARD is the same-cost soft surrogate, exactly like the frozen #253 quantised path (`hard.detach() + (soft - soft.detach())`). The bitwise forward requirement is therefore unchanged, and the structural term keeps the same gradient path to the encoder and the codebooks that #253 had.

For a lawful triple `(m; b, c)` and its frozen reference set, the per-level GraphMoRE A.1 proxy uses a **fixed, gate-independent** equal-weight three-expert symmetric distance:

`K_l = mean_a[ d²(q_a^l, q_m^l) + d²(q_b^l, q_c^l)/4 − ( d²(q_a^l, q_b^l) + d²(q_a^l, q_c^l) )/2 ]`,

averaged over the same reference set `a` as the frozen #253 estimator. The three levels share one sampling and one reference set; the GNN and the quantisation are not repeated per level.

## 5. Layer-wise loss

`L_total = L_original_RQVAE + λ · Σ_l α_l · Huber( K_l / target_scale , K_G / target_scale )`, with `α_l = 1/3` for `l = 1,2,3`.

- `target_scale` and the per-level distance scale `s_l` are fixed **before** training from the training-side initial cumulative prefixes (the untrained encoder state on the training split), so an early level is not penalised merely because its prefix representation has a different scale. `s_l` is recorded in the source as a frozen constant.
- `λ` is calibrated exactly once, before the formal run, on the real training-side initial state by the frozen rule `λ = ‖∇_encoder L_original_RQVAE‖ / ‖∇_encoder L_curv‖` at initialisation (equal encoder-gradient magnitude), and then frozen as a source constant. It is never tuned on test or Recall.
- `K_G` is the frozen graph target for the same triple.
- There is no additional behaviour loss and no graphlet-to-codebook rule.

## 6. Sampling bias statement

Training uses the frozen finite sampling/quantisation population, so the training gradient is a biased estimate of the full-population objective; the implementation must state this and must never claim full-population or unbiased training. Evaluation keeps the independent full 28,391,948-triple, full-reference measurement under one fixed cross-model scale and metric.

## 7. Paper / implementation cross-check

| Module | Reference | Our choice |
|---|---|---|
| Sectional-curvature proxy | GraphMoRE A.1 discrete curvature form (the frozen #252/#253 `common_fixed3` evaluator) | same formula per level, equal-weight three experts, fixed scale; the layer weights never enter the supervision target |
| SID / RQ-VAE | TIGER (RecBole) semantic-ID protocol | unchanged: 768→32, three 256-word levels, fourth counter untouched |
| Assignment | #253's actual protocol: L1/L2 floor-constrained nearest, L3 Sinkhorn | unchanged; only the cost matrix changes, through the layer weights |
| Gating | layer-conditioned attention over one shared GNN readout | one shared GNN, four token types, per-level trainable queries; no hard-wired curvature per level |

## 8. Tolerances, edge cases and required assertions

- Prefix equality: bitwise (`np.array_equal`) between `Σ_{j≤l} e_{i,j}` and the SID's first `l` codes.
- Gate normalisation: `|Σ_k w − 1| ≤ 1e-6` in float32.
- Scale constants: frozen and recorded; a drift above `1e-9` relative is a failure.
- Empty neighbourhood, reciprocal pairs, duplicate edges, isolated items, single-item tokens: explicit masks, no fabricated values.
- Gradient paths: `L_curv^(l)` must reach the encoder, the level-`l` codebook and the GNN, and must **not** depend on future levels' parameters; cutting the cost path must remove exactly that path.
- DDP: identical initial parameters across the four ranks and identical losses for the same batch.
- Assets: the new direction-ego asset must reload bit-exactly from `assets/` with its registered sha256, and must not be rebuilt during training.

## 9. Performance contract

Four-card inclusive epoch timing over at least 30 consecutive steady-state epochs, including all structural work, loading, communication and amortised periodic evaluation; report mean/median/P95, initialisation and export separately, per-module and per-rank times, and memory. Target 0.4–0.5 s/epoch, previously authorised hard ceiling 0.8 s/epoch (#253 measured 0.6032 s/epoch). The ceiling may not be met by dropping 2-hop, freezing the GNN, reducing sampling or hiding work.
