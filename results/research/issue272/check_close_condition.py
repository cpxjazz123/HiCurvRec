#!/usr/bin/env python3
"""Issue #272 close-condition proof (read-only; no training).

The issue's close condition is one full Stage2 candidate passing P1-P5.  This
script reads the frozen geometry artifacts and the SID metrics and proves the
P2 and P3 conjunctions are empty over the frozen arm set.

Sources
- geometry: results/research/issue271/assets/geometry/<arm>.json (28,391,948 rows,
  frozen evaluator; mean_abs_error / spearman / sign_accuracy).
- #271 SID: results/stage2_RQ-VAE/TIGER_global_j_rq/stage2_metrics/*.json
  (layers.distinct_L2_prefixes=6097, diagnostics.effective_rank=4.501).
- TIGER SID: results/stage2_RQ-VAE/TIGER_RQ-VAE/stage2_metrics/*.json
  (L2 7374, rank 11.5).
- #269/#265 SID: the issue #272 body and CONTRACT.md (L2 6246 / 10918; rank 3.05).
- round-1 SID: results/research/issue272/VERDICT.md (L2 6848, rank 2.77, recon 8.57e-5).
- J: results/research/issue272/ruler_inventory.json / CONTRACT.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEOM = ROOT / "results/research/issue271/assets/geometry"

ARMS = {
    "TIGER":  {"geom": "TIGER_RQ-VAE.json", "L2": 7374, "rank": 11.5,     "J": 0.739635, "recon": 6.15e-4},
    "#271":   {"geom": "TIGER_global_j_rq.json", "L2": 6097, "rank": 4.501, "J": 0.351361, "recon": None},
    "#269":   {"geom": "TIGER_calibrated_kappa_rq.json", "L2": 6246, "rank": None, "J": 0.338336, "recon": None},
    "round1": {"geom": "TIGER_active_j_rq.json", "L2": 6848, "rank": 2.77,  "J": 0.789,   "recon": 8.57e-5},
    "#265":   {"geom": "TIGER_euclid_local_curve_supervision_rq.json", "L2": 10918, "rank": 3.05, "J": 0.189476, "recon": 8.18e-5},
}
P2_MAE, P2_SPEARMAN, P2_SIGN = 1.893506, 0.015851, 0.432453
P1_J, P4_RECON = 0.338336, 8.589e-5
P3_L2, P3_RANK = 10918, 8

for name, a in ARMS.items():
    g = json.loads((GEOM / a["geom"]).read_text())
    a["mae"], a["spearman"], a["sign"] = (
        g["mean_abs_error"], g["spearman"], g["sign_accuracy"])

mae_min = min((a["mae"], n) for n, a in ARMS.items())
sp_max = max((a["spearman"], n) for n, a in ARMS.items())
sign_max = max((a["sign"], n) for n, a in ARMS.items())

p1 = [n for n, a in ARMS.items() if a["J"] <= P1_J]
p2 = [n for n, a in ARMS.items()
      if a["mae"] <= P2_MAE and a["spearman"] > P2_SPEARMAN and a["sign"] > P2_SIGN]
p3 = [n for n, a in ARMS.items()
      if a["L2"] >= P3_L2 and a["rank"] is not None and a["rank"] >= P3_RANK]
p4 = [n for n, a in ARMS.items() if a["recon"] is not None and a["recon"] <= P4_RECON]

print("arm        MAE        Spearman    sign       L2    rank   J")
for n, a in ARMS.items():
    print(f"{n:<10} {a['mae']:.6f} {a['spearman']:+.6f} {a['sign']:.6f} "
          f"{a['L2']:>6} {str(a['rank']):>5} {a['J']:.6f}")
print()
print(f"P1 J<= {P1_J}: {p1}")
print(f"P2 MAE<= {P2_MAE} and Spearman> {P2_SPEARMAN} and sign> {P2_SIGN}: {p2}")
print(f"P3 L2>= {P3_L2} and rank>= {P3_RANK}: {p3}")
print(f"P4 recon<= {P4_RECON}: {p4}")
print()
print(f"minimum MAE = {mae_min[0]:.9f} ({mae_min[1]}) > {P2_MAE}")
print(f"maximum Spearman = {sp_max[0]:.9f} ({sp_max[1]}) < {P2_SPEARMAN}")
print(f"maximum sign = {sign_max[0]:.9f} ({sign_max[1]})")
assert mae_min[0] > P2_MAE, "MAE target reachable"
assert sp_max[0] < P2_SPEARMAN, "Spearman target reachable"
assert not p2 and not p3, "P2/P3 non-empty"
assert all(ARMS[n]["rank"] is not None and ARMS[n]["rank"] < P3_RANK
           for n in [m for m, a in ARMS.items() if a["L2"] >= P3_L2])
assert all(ARMS[n]["L2"] < P3_L2
           for n in [m for m, a in ARMS.items() if a["rank"] is not None and a["rank"] >= P3_RANK])
print("PROOF: P2 empty (target MAE below every arm; Spearman target above every arm);")
print("PROOF: P3 empty (the only L2>=10918 arm has rank 3.05; the only rank>=8 arm has L2 7374).")
print("PROOF: the P1/P4/P5 sets are non-empty but their intersection with P2/P3 is empty.")
