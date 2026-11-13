"""The mixed-curvature product manifold, one curvature per item row.

Mechanism (see `ideas/AF3_curvature_per_item_idea/math.json`):

    M_i   = R^{D_E} x H^{D_H}(c_i)                      (formula `manifold_point`)
    x_i   = (r_i^E, expmap0(r_i^H, c_i))                (base point = the ball centre)
    u_k^{(i)} = (tau_k^E, expmap0(tau_k^H, c_i))
    D_ik  = ||r_i^E - tau_k^E||^2
            + SCALE_HYP * d_{c_i}(expmap0(r_i^H,c_i), expmap0(tau_k^H,c_i))^2
                                                        (formula `product_cost`)

The identity the implementation uses (formula `curvature_identity`):

    expmap0(v, c)      = (1/sqrt c) expmap0(sqrt(c) v, 1)
    d_c(x, y)          = (1/sqrt c) d_1(sqrt(c) x, sqrt(c) y)
  => d_{c}(expmap0(r,c), expmap0(t,c)) = (1/sqrt c) d_1(expmap0(sqrt c r,1), expmap0(sqrt c t,1))

so (formula `row_radius_equivalence`)

    D_ik = ||r_i^E - tau_k^E||^2
           + (SCALE_HYP / c_i) * d_1(expmap0(sqrt(c_i) r_i^H, 1),
                                     expmap0(sqrt(c_i) tau_k^H, 1))^2

The contract only accepts a scalar curvature, so every operator below is called at
`c = 1` and the per-row factor `sqrt(c_i)` / `1/c_i` is broadcast. Nothing here is
trainable: the curvature table is frozen before training starts.

Two read-only decompositions are exposed for the probe (formula `block_weight_split`):

    main   E_ik + H1_ik(c_i) / c_i        (weight moves AND points move)
    W      E_ik + H1_ik(1)   / c_i        (only the block weight moves)
    P      E_ik + H1_ik(c_i)              (only the points move)

with `H1_ik(c) = SCALE_HYP * d_1(expmap0(sqrt(c) r_i^H, 1), expmap0(sqrt(c) tau_k^H, 1))^2`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .poincare_contract import expmap0, squared_distance

#: constants of the mechanism (math.json `constants`)
D = 32
D_E = 16
D_H = 16
SCALE_HYP = 0.25
C_MIN = 0.25
C_MAX = 4.0
C_EUC = 0.0
#: the curvature `behaviour_off` collapses to: `u_i = 0.5 -> c_i = 1`
C_OFF = 1.0

if D_E + D_H != D:  # pragma: no cover - a constant guard, fires at import if edited
    raise RuntimeError("the factor split must add up to the codebook dimension")


def squared_euclidean(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """The baseline expansion, `distances/distance.py:19-22`, character for character.

    `geometry_off` must agree with the baseline cost ELEMENT BY ELEMENT, so this is
    the same `|x|^2 - 2 x.c + |c|^2` write-out, not a broadcasted subtraction.
    """
    return (
        latent.pow(2).sum(1, keepdim=True)
        - 2 * latent @ codebook.t()
        + codebook.pow(2).sum(1, keepdim=True).t()
    )


def _validate_latent(latent: torch.Tensor) -> None:
    if latent.dim() != 2 or int(latent.shape[1]) != D:
        raise ValueError(
            "the per-item curvature geometry reads a (B, %d) tangent vector, got %s; "
            "the factor split is D_E=%d (curvature 0) + D_H=%d (curvature -c_i) and "
            "must not be changed by a caller" % (D, tuple(latent.shape), D_E, D_H)
        )


def _validate_rows(rows) -> None:
    if rows is None:
        raise ValueError(
            "per-item curvature needs the batch's item rows; `rows=None` has no "
            "fallback because any default (e.g. arange) would apply item A's "
            "curvature to item B without a word"
        )


class PerItemCurvatureGeometry:
    """`M_i = R^{D_E} x H^{D_H}(c_i)` read-out, cost matrix and the two quant parts.

    Frozen: no parameter, no buffer, nothing trains here. The table is a plain
    `float32` tensor and is moved to the caller's device on first use (so a
    `model.to(device)` call cannot leave it on the wrong device).
    """

    def __init__(self, curvature_table, *, behaviour_off: bool = False,
                 geometry_off: bool = False) -> None:
        table = torch.as_tensor(curvature_table, dtype=torch.float32).reshape(-1)
        if table.numel() == 0:
            raise ValueError("the per-item curvature table is empty")
        if not bool(torch.isfinite(table).all()):
            raise ValueError("the per-item curvature table has a non-finite entry")
        if bool((table <= 0).any()):
            raise ValueError("every per-item curvature must be > 0 (kappa = -c_i)")
        self._table = table
        self.behaviour_off = bool(behaviour_off)
        self.geometry_off = bool(geometry_off)

    # ---------------------------------------------------------------- switches
    @property
    def table(self) -> torch.Tensor:
        return self._table

    def _device_table(self, like: torch.Tensor) -> torch.Tensor:
        if self._table.device != like.device:
            self._table = self._table.to(like.device)
        return self._table

    def curvature_of(self, rows, like: torch.Tensor | None = None) -> torch.Tensor:
        """`c_i` for these rows (formula `curvature_map`); the two switches override it."""
        _validate_rows(rows)
        rows = torch.as_tensor(rows, dtype=torch.long)
        if self.geometry_off or self.behaviour_off:
            return torch.full((int(rows.numel()),), C_OFF,
                              dtype=torch.float32,
                              device=(like.device if like is not None else self._table.device))
        table = self._device_table(like if like is not None else self._table)
        rows = rows.to(table.device)
        if int(rows.numel()) and (int(rows.min()) < 0 or int(rows.max()) >= table.numel()):
            raise ValueError(
                "an item row is outside the frozen curvature table (%d rows); the "
                "table and the embedding row space must agree" % table.numel()
            )
        return table[rows]

    # ------------------------------------------------------------ row factors
    @staticmethod
    def _hyper_sq_rows(c: torch.Tensor, r_h: torch.Tensor, t_h: torch.Tensor,
                       *, detach_r: bool = False, detach_t: bool = False) -> torch.Tensor:
        """`H1(c) = d_1(expmap0(sqrt c r,1), expmap0(sqrt c t,1))^2` per row."""
        if detach_r:
            r_h = r_h.detach()
        if detach_t:
            t_h = t_h.detach()
        scale = torch.sqrt(c).unsqueeze(-1)
        x = expmap0(scale * r_h, 1.0)
        y = expmap0(scale * t_h, 1.0)
        return squared_distance(x, y, 1.0)

    @staticmethod
    def _hyper_sq_pairs(c: torch.Tensor, r_h: torch.Tensor, t_h: torch.Tensor,
                        *, detach_r: bool = False, detach_t: bool = False) -> torch.Tensor:
        """`H1_ik(c_i)` for every (row, codeword) pair -> `(B, K)`.

        Row `i` uses `sqrt(c_i)` on BOTH sides, i.e. the codeword is read in row i's
        own ball (`u_k^{(i)}`), which is where the per-item curvature shows up.
        """
        if detach_r:
            r_h = r_h.detach()
        if detach_t:
            t_h = t_h.detach()
        scale = torch.sqrt(c)                                  # (B,)
        x = expmap0(scale.unsqueeze(-1) * r_h, 1.0)             # (B, H)
        y = expmap0(scale.unsqueeze(-1).unsqueeze(-1) * t_h.unsqueeze(0), 1.0)  # (B, K, H)
        return squared_distance(x.unsqueeze(1), y, 1.0)         # (B, K)

    def _compose(self, latent: torch.Tensor, codebook: torch.Tensor, c: torch.Tensor,
                 *, weight_on: bool, point_on: bool) -> torch.Tensor:
        """`E_ik + SCALE_HYP * H1_ik(c_point) / c_weight` (formula `block_weight_split`)."""
        r_e, r_h = latent[:, :D_E], latent[:, D_E:]
        t_e, t_h = codebook[:, :D_E], codebook[:, D_E:]
        euclid = (r_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)
        c_point = c if point_on else torch.ones_like(c)
        hyper = self._hyper_sq_pairs(c_point, r_h, t_h)
        weight = c if weight_on else torch.ones_like(c)
        return euclid + SCALE_HYP * hyper / weight.unsqueeze(-1)

    # ------------------------------------------------------------------ public
    def cost(self, latent: torch.Tensor, codebook: torch.Tensor, rows) -> torch.Tensor:
        """`D_ik`, the main arm (formula `product_cost` / `row_radius_equivalence`)."""
        _validate_rows(rows)
        _validate_latent(latent)
        if self.geometry_off:
            return squared_euclidean(latent, codebook)
        c = self.curvature_of(rows, latent)
        return self._compose(latent, codebook, c, weight_on=True, point_on=True)

    def cost_weight_only(self, latent: torch.Tensor, codebook: torch.Tensor, rows) -> torch.Tensor:
        """The W arm: only the block weight `1/c_i` moves, the points stay at `c = 1`."""
        _validate_rows(rows)
        _validate_latent(latent)
        if self.geometry_off:
            return squared_euclidean(latent, codebook)
        c = self.curvature_of(rows, latent)
        return self._compose(latent, codebook, c, weight_on=True, point_on=False)

    def cost_point_only(self, latent: torch.Tensor, codebook: torch.Tensor, rows) -> torch.Tensor:
        """The P arm: only the points move with `sqrt(c_i)`, the weight stays at 1."""
        _validate_rows(rows)
        _validate_latent(latent)
        if self.geometry_off:
            return squared_euclidean(latent, codebook)
        c = self.curvature_of(rows, latent)
        return self._compose(latent, codebook, c, weight_on=False, point_on=True)

    def quant_parts(self, latent: torch.Tensor, assigned_codebook: torch.Tensor, rows):
        """`(commitment, codebook)` on the assigned codeword (formula `geometric_quantization`).

        `C_i` and `Q_i` are the same row-ball squared geodesic distance, divided by the
        dimension `d`, so the normalisation matches `F.mse_loss`. The stop-gradient side
        is the baseline's: `commitment` moves only the encoder output, `codebook` moves
        only the entry.
        """
        _validate_rows(rows)
        _validate_latent(latent)
        if assigned_codebook.shape != latent.shape:
            raise ValueError(
                "quant_parts compares the assigned codeword with the tangent vector; "
                "shapes %s and %s differ" % (tuple(latent.shape), tuple(assigned_codebook.shape))
            )
        if self.geometry_off:
            return (F.mse_loss(latent, assigned_codebook.detach()),
                    F.mse_loss(assigned_codebook, latent.detach()))
        c = self.curvature_of(rows, latent)
        r_e, r_h = latent[:, :D_E], latent[:, D_E:]
        q_e, q_h = assigned_codebook[:, :D_E], assigned_codebook[:, D_E:]
        # commitment: the encoder moves, the entry is stopped
        c_e = (r_e - q_e.detach()).pow(2).sum(-1)
        c_h = self._hyper_sq_rows(c, r_h, q_h, detach_t=True)
        commitment = (c_e + SCALE_HYP * c_h / c).mean() / D
        # codebook: the entry moves, the item point is stopped
        q_e_term = (q_e - r_e.detach()).pow(2).sum(-1)
        q_h_term = self._hyper_sq_rows(c, r_h, q_h, detach_r=True)
        codebook = (q_e_term + SCALE_HYP * q_h_term / c).mean() / D
        return commitment, codebook

    # ------------------------------------------------------------- diagnostics
    def ball_radius(self, rows, like: torch.Tensor | None = None) -> torch.Tensor:
        """`R_i = 1/sqrt(c_i)` (formula `curvature_map`)."""
        return 1.0 / torch.sqrt(self.curvature_of(rows, like))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return ("PerItemCurvatureGeometry(rows=%d, behaviour_off=%s, geometry_off=%s, "
                "D_E=%d, D_H=%d, SCALE_HYP=%s)"
                % (self._table.numel(), self.behaviour_off, self.geometry_off,
                   D_E, D_H, SCALE_HYP))
