"""The product manifold and the share map: the ONE place this mechanism computes geometry.

`ideas/AF3_curvature_factor_mass_idea/math.json` in code, formula by formula:

    share_map        p_i = u_i ; s_hi = sqrt(p_i), s_lo = sqrt(1 - p_i)   (behaviour_off: p_i = SHARE_REF)
    chart_head       x_i = (h_E, expmap0(h1, c_hi), expmap0(h2, c_lo)),  r_i^{E,0} = h_E
    energy_split     product side NOT scaled; the codeword side is:
                     g_k^{l,hi} = sqrt(p_i) tau_k^{l,h1}, g_k^{l,lo} = sqrt(1-p_i) tau_k^{l,h2}
                     v_k^{l,hi} = expmap0(g_k^{l,hi}, c_hi), v_k^{l,lo} = expmap0(g_k^{l,lo}, c_lo)
    product_cost     D_ik^l = ||r_i^{E,l} - tau_k^{E,l}||^2
                              + SCALE_HYP * ( d_{c_hi}(x_i^hi, v_k^{l,hi})^2
                                              + d_{c_lo}(x_i^lo, v_k^{l,lo})^2 )
    geometric_quantization  C (encoder side, codeword stopped) and Q (codeword side,
                     product stopped), both averaged over D_E+D_H1+D_H2 = 32
    euclidean_degeneracy    c -> 0 : d_c^2 -> 4|x-y|^2, so SCALE_HYP = 0.25 restores the
                     squared Euclidean term; `geometry_off` walks the baseline expansion
                     on the same 32-vector (drop p and both scalings, not a small c)

All operators come from `tools/research/geometry/poincare_contract.py`; nothing here
re-derives `expmap0`, `logmap0`, `squared_distance`, `project` or `BALL_EPS`. The module
holds no parameters and registers no trainable weight: `p` is per item, frozen before
training, and passed in by the caller.

This module sits at the candidate tree's top level on purpose. `model/__init__.py`
imports `model/layers.py`, which imports `distances/__init__.py`; both sides therefore
import THIS module and never each other's submodules, which is what keeps the package
graph acyclic (`distances/` does not import `model/`).

Curvature is a fixed ladder (`c_hi`, `c_lo` below): AF3 only moves `p_i`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

#: constants of the mechanism, copied from math.json `constants`
D_E = 16
D_H1 = 8
D_H2 = 8
D_H = D_H1 + D_H2
D_TOTAL = D_E + D_H1 + D_H2
C_HI = 4.0
C_LO = 0.25
SCALE_HYP = 0.25
SHARE_REF = 0.5

#: the two branches of `share`
SHARE_MODES = ("struct", "ref")
#: the three wiring switches (never a separate training arm)
COST_MODES = ("both_on", "behaviour_off", "geometry_off")
#: what this module needs from the geometry contract
CONTRACT_REQUIRED = ("expmap0", "logmap0", "squared_distance", "project", "BALL_EPS")
CONTRACT_RELATIVE = ("tools", "research", "geometry", "poincare_contract.py")

_CONTRACT = None


# --------------------------------------------------------------------------------------
# the geometry contract
# --------------------------------------------------------------------------------------

def repository_root() -> Path:
    """The repository root, found by the `stage2_RQ-VAE` marker (no parent counting)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


def contract_path() -> Path:
    """The full path of the ONE geometry implementation this module composes."""
    return repository_root().joinpath(*CONTRACT_RELATIVE)


def contract():
    """Import `poincare_contract` once (cached, fail loud with the path and the API)."""
    global _CONTRACT
    if _CONTRACT is not None:
        return _CONTRACT
    path = contract_path()
    if not path.is_file():
        raise RuntimeError("the geometry contract is missing: " + str(path))
    try:
        spec = importlib.util.spec_from_file_location("genrec_poincare_contract", path)
        if spec is None or spec.loader is None:
            raise ImportError("no import spec could be built for " + str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as error:
        raise RuntimeError("cannot import the geometry contract from %s (required API: %s): %s"
                           % (path, ", ".join(CONTRACT_REQUIRED), error)) from error
    missing = [name for name in CONTRACT_REQUIRED if not hasattr(module, name)]
    if missing:
        raise RuntimeError("%s does not expose %s; this module will not re-derive them"
                           % (path, ", ".join(missing)))
    _CONTRACT = module
    return module


# --------------------------------------------------------------------------------------
# validation (fail loud; a silent fallback would hide the mechanism)
# --------------------------------------------------------------------------------------

def _as_p(p, *, name: str = "p", size: int | None = None) -> torch.Tensor:
    """A 1-D share vector strictly inside (0, 1)."""
    values = p if torch.is_tensor(p) else torch.as_tensor(p, dtype=torch.float32)
    if values.dim() == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.dim() != 1:
        raise ValueError("%s must be a 1-D per-item share, got shape %s"
                         % (name, tuple(values.shape)))
    if size is not None and int(values.shape[0]) != int(size):
        raise ValueError("%s has %d entries but the batch has %d"
                         % (name, int(values.shape[0]), int(size)))
    if not torch.isfinite(values).all():
        raise ValueError("%s contains non-finite values" % (name,))
    if bool((values <= 0).any()) or bool((values >= 1).any()):
        raise ValueError("%s must be strictly inside (0, 1); a share of 0 or 1 would put a "
                         "whole factor's codeword mass on one side" % (name,))
    return values


def _check_batch(tensor: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if tensor.dim() != 2 or int(tensor.shape[1]) != int(dim):
        raise ValueError("%s must be (B, %d), got shape %s"
                         % (name, dim, tuple(tensor.shape)))
    return tensor


def _check_tau(tau: torch.Tensor) -> torch.Tensor:
    return _check_batch(tau, D_TOTAL, "tau (codebook)")


def _check_point(point: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    return _check_batch(point, dim, name)


def _check_centres(points: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    if points.dim() != 3 or int(points.shape[2]) != int(dim):
        raise ValueError("%s must be (B, K, %d), got shape %s"
                         % (name, dim, tuple(points.shape)))
    return points


# --------------------------------------------------------------------------------------
# share_map
# --------------------------------------------------------------------------------------

def share(u, mode: str = "struct") -> torch.Tensor:
    """`p_i = u_i` (mode 'struct') or the constant `SHARE_REF` (mode 'ref').

    Same signature on both branches, so `behaviour_off` changes nothing at the call
    site: only the value of the per-item share moves. `u` is the frozen full-table rank
    from `af3_item_potential`; it is not in any computation graph.
    """
    if mode not in SHARE_MODES:
        raise ValueError("unknown share mode %r; expected one of %s" % (mode, SHARE_MODES))
    values = _as_p(u, name="u")
    if mode == "ref":
        return torch.full_like(values, SHARE_REF)
    # p_i = u_i is the ONE mapping from the structure table to the geometry.
    return values.clone()


# --------------------------------------------------------------------------------------
# chart_head
# --------------------------------------------------------------------------------------

def split_tangent(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`h_i = (h_E | h1 | h2)` with the fixed dimension split of math.json constants."""
    _check_batch(h, D_TOTAL, "h (encoder tangent)")
    return h[:, :D_E], h[:, D_E:D_E + D_H1], h[:, D_E + D_H1:]


def item_chart_head(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(x_i^hi, x_i^lo) = (expmap0(h1, c_hi), expmap0(h2, c_lo))`.

    The product side is NOT scaled by `p`: this is the injective chart head that makes
    the encoder output a point of M. The Euclidean block `h_E` is carried straight
    through by `split_tangent` (it is `r_i^{E,0}`).
    """
    _, h1, h2 = split_tangent(h)
    geometry = contract()
    return geometry.expmap0(h1, C_HI), geometry.expmap0(h2, C_LO)


def log_at_origin(x_hi: torch.Tensor, x_lo: torch.Tensor) -> torch.Tensor:
    """The inverse of `item_chart_head`: `(B, 16) = (logmap0(x_hi) | logmap0(x_lo))`."""
    _check_point(x_hi, D_H1, "x_hi")
    _check_point(x_lo, D_H2, "x_lo")
    if int(x_hi.shape[0]) != int(x_lo.shape[0]):
        raise ValueError("x_hi and x_lo must share the batch dimension")
    geometry = contract()
    return torch.cat([geometry.logmap0(x_hi, C_HI), geometry.logmap0(x_lo, C_LO)], dim=-1)


# --------------------------------------------------------------------------------------
# energy_split (codeword side only)
# --------------------------------------------------------------------------------------

def codeword_chart(tau: torch.Tensor, p) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-item effective codeword cloud `(v_hi, v_lo)`, each `(B, K, 8)`.

    `g_k^{l,hi} = sqrt(p_i) tau_k^{l,h1}` and `g_k^{l,lo} = sqrt(1-p_i) tau_k^{l,h2}`
    are mapped into their own ball (`c_hi`, `c_lo`). The Euclidean block is untouched:
    it is handled by `cost`. Both factors share the codeword, the share decides how much
    of its tangent mass each factor receives -- that is the mechanism.
    """
    _check_tau(tau)
    if p is None:
        raise ValueError("codeword_chart needs the per-item share p; it is never defaulted")
    values = _as_p(p, name="p")
    geometry = contract()
    tau_h1 = tau[:, D_E:D_E + D_H1]
    tau_h2 = tau[:, D_E + D_H1:]
    s_hi = torch.sqrt(values).view(-1, 1, 1)
    s_lo = torch.sqrt(1.0 - values).view(-1, 1, 1)
    batch = int(values.shape[0])
    codes = int(tau.shape[0])
    g_hi = (s_hi * tau_h1.unsqueeze(0)).reshape(batch * codes, D_H1)
    g_lo = (s_lo * tau_h2.unsqueeze(0)).reshape(batch * codes, D_H2)
    v_hi = geometry.expmap0(g_hi, C_HI).view(batch, codes, D_H1)
    v_lo = geometry.expmap0(g_lo, C_LO).view(batch, codes, D_H2)
    return v_hi, v_lo


# --------------------------------------------------------------------------------------
# product_cost
# --------------------------------------------------------------------------------------

def cost(rE: torch.Tensor, x_hi: torch.Tensor, x_lo: torch.Tensor,
         v_hi: torch.Tensor, v_lo: torch.Tensor, tau_E: torch.Tensor) -> torch.Tensor:
    """`D_ik^l` of formula `product_cost`, `(B, K)`.

    `||r^E - tau_E||^2` on the Euclidean factor plus `SCALE_HYP * (d_{c_hi}^2 + d_{c_lo}^2)`
    on the two hyperbolic factors, each with its OWN curvature (`c_hi = 4.0`,
    `c_lo = 0.25`). `tau_E` is the codeword's Euclidean block and must be given: the
    first term of the formula is a distance to it, so it cannot be derived from
    `v_hi`/`v_lo`, which are points of the two balls.
    """
    _check_batch(rE, D_E, "rE (Euclidean residual)")
    _check_point(x_hi, D_H1, "x_hi")
    _check_point(x_lo, D_H2, "x_lo")
    _check_centres(v_hi, D_H1, "v_hi")
    _check_centres(v_lo, D_H2, "v_lo")
    _check_batch(tau_E, D_E, "tau_E (codeword Euclidean block)")
    batch = int(rE.shape[0])
    if int(x_hi.shape[0]) != batch or int(x_lo.shape[0]) != batch:
        raise ValueError("the three product-side blocks must share the batch dimension")
    if int(v_hi.shape[0]) != batch or int(v_lo.shape[0]) != batch:
        raise ValueError("the codeword clouds must have one row of points per item")
    if int(v_hi.shape[1]) != int(v_lo.shape[1]):
        raise ValueError("v_hi and v_lo must describe the same codewords")
    geometry = contract()
    euclidean = (rE.unsqueeze(1) - tau_E.unsqueeze(0)).pow(2).sum(-1)
    hyperbolic = (geometry.squared_distance(x_hi.unsqueeze(1), v_hi, C_HI)
                  + geometry.squared_distance(x_lo.unsqueeze(1), v_lo, C_LO))
    return euclidean + SCALE_HYP * hyperbolic


def euclidean_expansion(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The baseline expansion `|x|^2 - 2 x.c + |c|^2`, byte for byte (`geometry_off`)."""
    _check_batch(latent, D_TOTAL, "latent")
    _check_tau(codebook)
    return (latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t())


def product_cost_blocks(rE: torch.Tensor, x_hi: torch.Tensor, x_lo: torch.Tensor,
                        codebook: torch.Tensor, p,
                        mode: str = "both_on") -> torch.Tensor:
    """`D_ik` when the caller ALREADY applied the chart head (`x_hi`, `x_lo`).

    Same formula as `product_cost`, one code path: this entry exists so the quantiser
    can hand over the charted product point it got from the encoder instead of charting
    the same tangent twice. `geometry_off` is deliberately NOT available here -- that
    branch is defined on the tangent 32-vector (`h1`/`h2`), which `euclidean_expansion`
    consumes, so it walks `product_cost`.
    """
    if mode not in COST_MODES:
        raise ValueError("unknown cost mode %r; expected one of %s" % (mode, COST_MODES))
    if mode == "geometry_off":
        raise ValueError("geometry_off is the tangent branch: pass the 32-vector to "
                         "product_cost, not the charted product point")
    _check_batch(rE, D_E, "rE (Euclidean residual)")
    _check_tau(codebook)
    if p is None:
        raise ValueError("product_cost needs the per-item share p; it is never defaulted")
    _as_p(p, name="p", size=int(rE.shape[0]))
    values = share(p, "ref" if mode == "behaviour_off" else "struct")
    v_hi, v_lo = codeword_chart(codebook, values)
    return cost(rE, x_hi, x_lo, v_hi, v_lo, codebook[:, :D_E])


def product_cost(latent: torch.Tensor, codebook: torch.Tensor, p,
                 mode: str = "both_on") -> torch.Tensor:
    """`D_ik` for the whole 32-vector, dispatching on the wiring switch.

    `latent = (r_i^{E,l} | h1_i | h2_i)` -- the Euclidean residual first, then the two
    hyperbolic tangent blocks, the same layout as the baseline's 32-vector, so the
    caller never has to assemble anything. The chart head is applied here, in the one
    geometry implementation, when the caller hands over the tangent form.

    mode:
      'both_on'       p_i = u_i, product side unscaled, codeword side split by p (main arm)
      'behaviour_off' p_i == SHARE_REF, everything else identical
      'geometry_off'  ignore p and both scalings, walk `euclidean_expansion`
    """
    if mode not in COST_MODES:
        raise ValueError("unknown cost mode %r; expected one of %s" % (mode, COST_MODES))
    _check_batch(latent, D_TOTAL, "latent")
    _check_tau(codebook)
    batch = int(latent.shape[0])
    if p is None:
        raise ValueError("product_cost needs the per-item share p; it is never defaulted")
    _as_p(p, name="p", size=batch)
    if mode == "geometry_off":
        # The explicit Euclidean branch: c -> 0 AND the scalings dropped.
        return euclidean_expansion(latent, codebook)
    rE, _, _ = split_tangent(latent)
    x_hi, x_lo = item_chart_head(latent)
    return product_cost_blocks(rE, x_hi, x_lo, codebook, p, mode=mode)


# --------------------------------------------------------------------------------------
# geometric_quantization
# --------------------------------------------------------------------------------------

def _selected_cost(rE: torch.Tensor, x_hi: torch.Tensor, x_lo: torch.Tensor,
                   v_hi: torch.Tensor, v_lo: torch.Tensor, tau_E: torch.Tensor):
    """One row's `D_ik` summed over 32 dimensions, averaged over the batch."""
    geometry = contract()
    euclidean = (rE - tau_E).pow(2).sum(-1)
    hyperbolic = (geometry.squared_distance(x_hi, v_hi, C_HI)
                  + geometry.squared_distance(x_lo, v_lo, C_LO))
    return (euclidean + SCALE_HYP * hyperbolic).mean() / float(D_TOTAL)


def quant_parts(rE: torch.Tensor, x_hi: torch.Tensor, x_lo: torch.Tensor,
                v_hi: torch.Tensor, v_lo: torch.Tensor, tau_E: torch.Tensor,
                index: torch.Tensor):
    """`(C, Q)` for the assigned codewords, the pair the loss registry consumes.

    `index` is the `(B,)` assignment of the level. `C` stops the codeword side and
    reaches the encoder (like `losses/commitment.py`), `Q` stops the product side and
    reaches the codeword entries (like `losses/codebook.py`). Both are divided by
    `D_E + D_H1 + D_H2 = 32`, matching `F.mse_loss` over 32 dimensions.
    """
    _check_batch(rE, D_E, "rE (Euclidean residual)")
    _check_point(x_hi, D_H1, "x_hi")
    _check_point(x_lo, D_H2, "x_lo")
    _check_centres(v_hi, D_H1, "v_hi")
    _check_centres(v_lo, D_H2, "v_lo")
    _check_batch(tau_E, D_E, "tau_E (codeword Euclidean block)")
    batch = int(rE.shape[0])
    if index is None:
        raise ValueError("quant_parts needs the level's assignment index")
    if index.dim() != 1 or int(index.shape[0]) != batch:
        raise ValueError("index must be (B,) = (%d,), got shape %s"
                         % (batch, tuple(index.shape)))
    if not torch.is_floating_point(index) and int(index.min()) < 0:
        raise ValueError("index contains a negative codeword id")
    if int(index.max()) >= int(v_hi.shape[1]):
        raise ValueError("index %d is outside the codebook of %d entries"
                         % (int(index.max()), int(v_hi.shape[1])))
    gather_hi = index.view(batch, 1, 1).expand(batch, 1, D_H1)
    gather_lo = index.view(batch, 1, 1).expand(batch, 1, D_H2)
    v_hi_sel = v_hi.gather(1, gather_hi).squeeze(1)
    v_lo_sel = v_lo.gather(1, gather_lo).squeeze(1)
    tau_E_sel = tau_E.gather(0, index.view(batch, 1).expand(batch, D_E))
    commitment = _selected_cost(rE, x_hi, x_lo,
                                v_hi_sel.detach(), v_lo_sel.detach(), tau_E_sel.detach())
    codebook = _selected_cost(rE.detach(), x_hi.detach(), x_lo.detach(),
                              v_hi_sel, v_lo_sel, tau_E_sel)
    return commitment, codebook


__all__ = ["D_E", "D_H1", "D_H2", "D_H", "D_TOTAL", "C_HI", "C_LO", "SCALE_HYP",
           "SHARE_REF", "SHARE_MODES", "COST_MODES", "CONTRACT_REQUIRED",
           "repository_root", "contract_path", "contract", "share", "split_tangent",
           "item_chart_head", "log_at_origin", "codeword_chart", "cost",
           "euclidean_expansion", "product_cost", "product_cost_blocks", "quant_parts"]
