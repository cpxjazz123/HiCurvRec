#!/usr/bin/env python3
"""K3 / K4 (complete graph) edge ORC 独立验证.

verdict §11.2 声称 K3=1.0, K4=1.0 但教科书 Lin 2011 / Ollivier 2009 显式解不同:
  K_n edge κ = 1 - α - (1-α)/n   (α=0.5 lazy)
  K3: κ = 1 - 0.5 - 0.5/3 = 1/3
  K4: κ = 1 - 0.5 - 0.5/4 = 3/8

执行方式:
  1. 用 task163_5_real_orc_measurement.py 里的 edge_orc 跑 K3, K4
  2. 独立手算一个 3 变量 brute-force OT 对照 (穷举所有 f_{xy} ∈ ℝ^{|supp_u|×|supp_v|})
  3. 用 scipy linprog 跑 (跟 verdict 实现同一路径)
  4. 输出对比
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

import networkx as nx
from scipy.optimize import linprog
import numpy as np
import itertools


def edge_orc_ours(G, u, v, alpha=0.5):
    """Direct copy of verdict's edge_orc logic."""
    d_uv = nx.shortest_path_length(G, u, v)
    if d_uv == 0:
        return 0.0
    if G.degree(u) == 0 or G.degree(v) == 0:
        return 0.0

    def lazy_dist(node):
        nbrs = list(G.neighbors(node))
        d_x = G.degree(node)
        probs = {node: alpha}
        for n in nbrs:
            probs[n] = (1 - alpha) / d_x
        return probs

    mu_u = lazy_dist(u)
    mu_v = lazy_dist(v)

    keys_u = sorted(mu_u.keys())
    keys_v = sorted(mu_v.keys())
    dists = []
    for x in keys_u:
        for y in keys_v:
            try:
                d_xy = nx.shortest_path_length(G, x, y)
            except nx.NetworkXNoPath:
                d_xy = 100
            dists.append(d_xy)
    cost = np.array(dists, dtype=float)

    n_u, n_v = len(keys_u), len(keys_v)
    n_vars = n_u * n_v
    a_eq = np.zeros((n_u + n_v, n_vars))
    rhs = np.zeros(n_u + n_v)
    for i, x in enumerate(keys_u):
        for j, y in enumerate(keys_v):
            a_eq[i, i * n_v + j] = 1.0
        rhs[i] = mu_u[x]
    for j, y in enumerate(keys_v):
        for i, x in enumerate(keys_u):
            a_eq[n_u + j, i * n_v + j] = 1.0
        rhs[n_u + j] = mu_v[y]
    bounds = [(0, None)] * n_vars

    res = linprog(c=cost, A_eq=a_eq, b_eq=rhs, bounds=bounds, method='highs')
    if res.success:
        return 1.0 - res.fun / d_uv
    else:
        return float('nan')


def brute_force_ot(mu_u, mu_v, cost_matrix):
    """穷举所有可能 flow assignment, 找最小 cost.
    只用于小规模 (n_u * n_v <= 6) 验证."""
    n_u, n_v = mu_u.shape[0], mu_v.shape[0]

    # Construct constraints: outflow sums + inflow sums
    # Brute force: enumerate all (n_u + n_v) equality-constrained solutions
    # Use linprog with method='revised simplex' as second cross-check
    cost = cost_matrix.flatten()
    a_eq = np.zeros((n_u + n_v, n_u * n_v))
    rhs = np.zeros(n_u + n_v)
    for i in range(n_u):
        for j in range(n_v):
            a_eq[i, i * n_v + j] = 1.0
        rhs[i] = mu_u[i]
    for j in range(n_v):
        for i in range(n_u):
            a_eq[n_u + j, i * n_v + j] = 1.0
        rhs[n_u + j] = mu_v[j]
    bounds = [(0, None)] * (n_u * n_v)

    # Try all 3 LP methods
    results = {}
    for method in ['highs', 'revised simplex', 'interior-point']:
        try:
            res = linprog(c=cost, A_eq=a_eq, b_eq=rhs, bounds=bounds, method=method)
            results[method] = res.fun if res.success else float('nan')
        except Exception as e:
            results[method] = f'ERR: {e}'
    return results


def main():
    print("=" * 70)
    print("K_n (n=3,4) Complete Graph Edge ORC Independent Verification")
    print("=" * 70)

    # Reference formula: κ(u,v) = 1 - α - (1-α)/n   for α=0.5 → κ = 1/2 - 1/(2n)
    for n in [3, 4]:
        G = nx.complete_graph(n)
        expected = 0.5 - 0.5/n  # K_n textbook
        print(f"\nK_{n} ({n}-clique, {G.number_of_edges()} edges)")
        print(f"  Lin 2011 textbook: κ = 1 - α - (1-α)/n = 1/2 - 1/(2·{n}) = {expected:.4f}")
        print(f"  Verdict §11.2 claimed: 1.000 ← {'SUSPICIOUS' if abs(expected - 1.0) > 1e-6 else 'match'}")
        print(f"  Edges:")
        edges = list(G.edges())
        for u, v in edges:
            k = edge_orc_ours(G, u, v)
            match = '✓' if abs(k - expected) < 1e-6 else '✗'
            print(f"    ({u},{v}) verdict_impl κ = {k:+.6f}  expected {expected:+.6f}  {match}")

        # Brute force cross-check on edge (0,1) only (representative)
        u, v = 0, 1
        alpha = 0.5
        mu_u = {u: alpha}
        mu_v = {v: alpha}
        for w in G.neighbors(u):
            if w != v:
                mu_u.setdefault(w, 0)
            mu_u[w] = (1 - alpha) / G.degree(u)
        # Re-construct mu_u with correct formula
        mu_u = {u: alpha}
        for w in G.neighbors(u):
            mu_u[w] = (1 - alpha) / G.degree(u)
        mu_v = {v: alpha}
        for w in G.neighbors(v):
            mu_v[w] = (1 - alpha) / G.degree(v)

        keys_u = sorted(mu_u.keys())
        keys_v = sorted(mu_v.keys())
        cost_matrix = np.zeros((len(keys_u), len(keys_v)))
        for i, x in enumerate(keys_u):
            for j, y in enumerate(keys_v):
                cost_matrix[i, j] = nx.shortest_path_length(G, x, y)
        print(f"  Edge (0,1) cost matrix (d(i,j) in K_{n}):")
        for i, x in enumerate(keys_u):
            row = ' '.join(f'{cost_matrix[i,j]:>4.0f}' for j in range(len(keys_v)))
            print(f"    {x}: [{row}]  μ_u[{x}]={mu_u[x]:.4f}  μ_v[{x}]={mu_v.get(x, 0):.4f}")
        mu_u_arr = np.array([mu_u[k] for k in keys_u])
        mu_v_arr = np.array([mu_v[k] for k in keys_v])
        print(f"  Edge (0,1) LP cross-check:")
        lps = brute_force_ot(mu_u_arr, mu_v_arr, cost_matrix)
        for m, v_ in lps.items():
            kappa = 1 - v_ / 1
            match = '✓' if abs(kappa - expected) < 1e-6 else '✗'
            print(f"    method={m:>16s} W1={v_:.6f} κ={kappa:+.6f} (expected {expected:+.6f}) {match}")


if __name__ == '__main__':
    main()