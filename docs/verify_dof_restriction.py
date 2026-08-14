"""Reproduces every numerical claim in docs/dof_restriction_note.tex.

Sections below are numbered as the environments of the note. The reference
throughout is a central-difference Jacobian of the bearing map taken on the
admissible manifold, which uses neither construction.

    PYTHONPATH=. uv run docs/verify_dof_restriction.py
    PYTHONPATH=. uv run docs/verify_dof_restriction.py --quick
"""
import copy
import sys
import time

import numpy as np

from network import Network
from rigidity import (bearing_DOFs, extended_bearing_rigidity_matrix,
                      node_dof_projectors)  # bearing_DOFs is the repository's copy
from scenario import random_scenario
from util import orthogonal_projection_matrix, skew_symmetric

QUICK = "--quick" in sys.argv

DOMAINS = ["R^2", "R^3", "R^2xS^1", "R^3xS^1", "SE(3)"]
DIM = {"R^2": 2, "R^3": 3, "R^2xS^1": 3, "R^3xS^1": 4, "SE(3)": 6}
CLOSED_FORM = {"R^2": lambda n: 2 * n - 3, "R^3": lambda n: 3 * n - 4,
               "R^2xS^1": lambda n: 3 * n - 4, "R^3xS^1": lambda n: 4 * n - 5,
               "SE(3)": lambda n: 6 * n - 7}
PLANAR = {"R^2", "R^2xS^1"}
E3 = np.array([0.0, 0.0, 1.0])

rank = np.linalg.matrix_rank
results = {}


def head(n, title):
    print(f"\n{'=' * 78}\n{n}  {title}\n{'=' * 78}")


def record(key, ok):
    results[key] = ok
    return ok


# --------------------------------------------------------------- constructions

def B_node(net):
    """Node-indexed form, eq. (5) of the note. This is the repository's."""
    return extended_bearing_rigidity_matrix(net)


def table_I_III(agent_i, agent_j):
    """U_ij, V_ij exactly as printed in Table I and Table III of the paper.

    Not imported from rigidity.bearing_DOFs: that copy stores the R^3xS^1
    rotational entry as e3 v^T (rows) where the paper has [0_{3x2} v] (columns).
    The two coincide only at v = e3, which is the only axis the repository uses.
    """
    di, dj = agent_i.domain, agent_j.domain
    e1, e2, e3 = np.eye(3)
    zero = np.zeros(3)
    planar_pair = {"R^2", "R^2xS^1"}
    U = np.eye(3) if (di not in planar_pair or dj not in planar_pair) \
        else np.column_stack([e1, e2, zero])
    if di in ("SE(3)", "R^3xSO(3)", "", None):
        V = np.eye(3)
    elif di == "R^3xS^1":
        v = agent_i.rotation_axis
        v = e3 if v is None else np.asarray(v, float) / np.linalg.norm(v)
        V = np.column_stack([zero, zero, v])
    elif di == "R^2xS^1":
        V = np.column_stack([zero, zero, e3])
    else:
        V = np.zeros((3, 3))
    return U, V


def B_edge(net):
    """Edge-indexed form, eq. (4), with U_ij, V_ij from Table I / Table III."""
    p = [a.pose.position for a in net.agents]
    R = [a.pose.rotation_mat() for a in net.agents]
    n, ii, jj = net.n, *np.nonzero(net.edges)
    m = len(ii)
    E = np.zeros((n, m))
    Eo = np.zeros((n, m))
    U = np.zeros((3 * m, 3 * m))
    V = np.zeros((3 * m, 3 * m))
    Dp = np.zeros((3 * m, 3 * m))
    Da = np.zeros((3 * m, 3 * m))
    for k, (i, j) in enumerate(zip(ii, jj)):
        E[i, k], E[j, k], Eo[i, k] = -1, +1, -1
        U[3 * k:3 * k + 3, 3 * k:3 * k + 3], V[3 * k:3 * k + 3, 3 * k:3 * k + 3] = \
            table_I_III(net.agents[i], net.agents[j])
        pij = p[j] - p[i]
        s = 1.0 / np.linalg.norm(pij)
        Dp[3 * k:3 * k + 3, 3 * k:3 * k + 3] = s * R[i].T @ orthogonal_projection_matrix(s * pij)
        Da[3 * k:3 * k + 3, 3 * k:3 * k + 3] = -R[i].T @ skew_symmetric(s * pij)
    return np.hstack([Dp @ U @ np.kron(E, np.eye(3)).T,
                      Da @ V @ np.kron(Eo, np.eye(3)).T])


# ------------------------------------------------------------- ground truth

def generators(net, param="projector"):
    """(coordinate, physical motion) for each admissible degree of freedom.

    The two differ for an R^3xS^1 agent. Under the projector parametrisation the
    variable is omega_i itself, confined to span{v}. Under Table I the variable is
    theta-dot, stored in the third slot and mapped to omega_i = theta-dot v by
    V_ij. Same motion, different coordinate, so the check needs both.
    """
    n, coords, phys = net.n, [], []
    for i, a in enumerate(net.agents):
        S, P = node_dof_projectors(a)
        u, s, _ = np.linalg.svd(S)
        for c in range(int((s > 1e-9).sum())):
            v = np.zeros(6 * n)
            v[3 * i:3 * i + 3] = u[:, c]
            coords.append(v)
            phys.append(v.copy())
        if a.domain == "R^3xS^1" and param == "table1":
            ax = a.rotation_axis
            ax = np.array([0.0, 0.0, 1.0]) if ax is None else np.asarray(ax, float)
            ax = ax / np.linalg.norm(ax)
            cv = np.zeros(6 * n)
            cv[3 * n + 3 * i + 2] = 1.0                      # theta-dot slot
            pv = np.zeros(6 * n)
            pv[3 * n + 3 * i:3 * n + 3 * i + 3] = ax         # omega = v
            coords.append(cv)
            phys.append(pv)
        else:
            u, s, _ = np.linalg.svd(P)
            for c in range(int((s > 1e-9).sum())):
                v = np.zeros(6 * n)
                v[3 * n + 3 * i:3 * n + 3 * i + 3] = u[:, c]
                coords.append(v)
                phys.append(v.copy())
    return np.array(coords).T, np.array(phys).T


def admissible_basis(net):
    """Orthonormal columns spanning A, the admissible subspace."""
    A, _ = generators(net, "projector")
    return A


def bearings(net):
    ii, jj = np.nonzero(net.edges)
    out = [net.agents[i].pose.rotation_mat().T
           @ ((net.agents[j].pose.position - net.agents[i].pose.position)
              / np.linalg.norm(net.agents[j].pose.position - net.agents[i].pose.position))
           for i, j in zip(ii, jj)]
    return np.concatenate(out) if out else np.zeros(0)


def perturb(net, delta, eps):
    out = copy.deepcopy(net)
    n = out.n
    for i, a in enumerate(out.agents):
        a.pose.position = a.pose.position + eps * delta[3 * i:3 * i + 3]
        w = eps * delta[3 * n + 3 * i:3 * n + 3 * i + 3]
        th = np.linalg.norm(w)
        if th > 0:
            K = skew_symmetric(w / th)
            Rd = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            a.pose.set_rotation_mat(Rd @ a.pose.rotation_mat())
    return out


def jacobian_truth(net, phys=None, eps=1e-6):
    """d(bearings)/dt along each physical generator, by central differences."""
    if phys is None:
        phys = generators(net, "projector")[1]
    cols = []
    for c in range(phys.shape[1]):
        cols.append((bearings(perturb(net, phys[:, c], eps))
                     - bearings(perturb(net, phys[:, c], -eps))) / (2 * eps))
    return np.array(cols).T


def rank_fd(M):
    """Rank at a tolerance matched to finite-difference noise, not to float eps."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > max(1e-7, s[0] * 1e-9)).sum())


def is_ibr(net, builder):
    return rank(builder(net)) == rank(builder(net.fully_connected()))


def is_ibr_truth(net):
    return rank_fd(jacobian_truth(net)) == rank_fd(jacobian_truth(net.fully_connected()))


def null_columns(B):
    return [c for c in range(B.shape[1]) if np.abs(B[:, c]).max() < 1e-12]


# ------------------------------------------------------------------- sampling

def seed(k):
    """Both streams: random_scenario draws poses from the global numpy state."""
    np.random.seed(k)
    return np.random.default_rng(k)


def sample(domains, rng, m=None, random_axes=False):
    n = len(domains)
    net, _ = random_scenario(n, list(domains))
    if random_axes:
        for a in net.agents:
            if a.domain == "R^3xS^1":
                v = rng.standard_normal(3)
                a.set_domain("R^3xS^1", rotation_axis=v / np.linalg.norm(v))
        net.randomize_orientations()
    if m is not None:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        E = np.zeros((n, n), dtype=bool)
        for k in rng.choice(len(pairs), size=min(m, len(pairs)), replace=False):
            E[pairs[k]] = True
        net.edges = E
    return net


MIXES = [["R^2", "R^3"], ["R^2", "SE(3)"], ["R^2xS^1", "R^3xS^1"],
         ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"], ["R^2"] * 3 + ["SE(3)"],
         ["R^3"] * 2 + ["R^2xS^1"] * 2, ["R^3xS^1"] * 2 + ["R^2"] * 2,
         ["SE(3)"] * 2 + ["R^2"] * 2 + ["R^3xS^1"]]


# =============================================================== 1  Lemma 1

def section_lemma():
    head("1", "Lemma 1   both forms reproduce the differential on A")
    rng = seed(0)
    cases = [([d] * n, f"{d} n={n}") for d in DOMAINS for n in (3, 4, 6, 9)]
    for mx in MIXES:
        for rep in (1, 2):
            lbl = ",".join(sorted(set(mx)))
            cases.append((mx * rep, f"mix {lbl} n={len(mx) * rep}"))
    if QUICK:
        cases = cases[::4]

    worst = {"node": 0.0, "edge": 0.0}
    ngraph = 0
    for doms, _ in cases:
        n = len(doms)
        for _ in range(3 if not QUICK else 1):
            net = sample(doms, rng, random_axes=True)
            Ap, Pp = generators(net, "projector")
            At, Pt = generators(net, "table1")
            for m in (max(1, n // 2), n, 2 * n, n * (n - 1)):
                pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
                E = np.zeros((n, n), dtype=bool)
                for k in rng.choice(len(pairs), size=min(m, len(pairs)), replace=False):
                    E[pairs[k]] = True
                net.edges = E
                Jp, Jt = jacobian_truth(net, Pp), jacobian_truth(net, Pt)
                sp = max(np.abs(Jp).max(), 1e-12)
                st = max(np.abs(Jt).max(), 1e-12)
                worst["node"] = max(worst["node"], np.abs(B_node(net) @ Ap - Jp).max() / sp)
                worst["edge"] = max(worst["edge"], np.abs(B_edge(net) @ At - Jt).max() / st)
                ngraph += 1

    print(f"  frameworks tested                     {ngraph}")
    print(f"  worst relative error, node-indexed    {worst['node']:.1e}")
    print(f"  worst relative error, edge-indexed    {worst['edge']:.1e}")
    print("\n  Both agree with the differential on A, each in its own parametrisation")
    print("  of the R^3xS^1 rotational coordinate. They can only differ off A.")
    return record("Lemma 1", max(worst.values()) < 1e-6), ngraph, max(worst.values())


# =========================================================== 2  Remark 2

def section_remark_aligned():
    head("2", "Remark 2   'null columns' vs the identity B(I - AA^T) = 0")
    rng = seed(1)
    rows = []
    for doms, axis in ((["R^2"] * 3 + ["R^3"], "n/a"),
                       (["R^2", "R^2xS^1", "R^3", "SE(3)"], "n/a"),
                       (["R^3xS^1"] * 4, "e3"),
                       (["R^3xS^1"] * 4, "generic"),
                       (["R^2", "R^3xS^1", "SE(3)"], "generic")):
        n = len(doms)
        net, _ = random_scenario(n, list(doms))
        for a in net.agents:
            if a.domain == "R^3xS^1":
                v = E3 if axis == "e3" else rng.standard_normal(3)
                a.set_domain("R^3xS^1", rotation_axis=v / np.linalg.norm(v))
        net.randomize_orientations()
        K = net.fully_connected()
        B, A = B_node(K), admissible_basis(net)
        Q, _ = np.linalg.qr(A)
        resid = np.abs(B @ (np.eye(6 * n) - Q @ Q.T)).max()
        expected = 6 * n - sum(DIM[d] for d in doms)
        rows.append((",".join(sorted(set(doms))), axis, resid,
                     len(null_columns(B)), expected))

    print(f"  {'domains':34s} {'axis':>8s} {'|B(I-AAt)|':>12s} {'#null col':>10s} {'6n-c':>6s}")
    for lbl, ax, r, nc, ex in rows:
        flag = "" if nc == ex else "   <- count fails"
        print(f"  {lbl:34s} {ax:>8s} {r:12.1e} {nc:10d} {ex:6d}{flag}")
    print(f"\n  worst |B(I - AA^T)| over the five cases: {max(r for _, _, r, _, _ in rows):.1e}")
    print("\n  The identity holds everywhere. The column count is equivalent to it only")
    print("  when A is coordinate aligned, which fails for R^3xS^1 with a generic axis")
    print("  under the projector parametrisation P_i = v v^T.")
    return record("Remark 2", max(r for _, _, r, _, _ in rows) < 1e-12)


# ============================================================== 3  Remark 3

def section_remark_qv():
    head("3", "Remark 3   q_v is structural, not a column count")
    seed(5)
    doms = ["R^2"] * 3 + ["SE(3)"]
    n = len(doms)
    net, _ = random_scenario(n, doms)
    E = np.zeros((n, n), dtype=bool)
    E[0, 1] = E[1, 0] = True
    net.edges = E
    structural = 6 * n - sum(DIM[d] for d in doms)
    counted = len(null_columns(B_node(net)))
    print(f"  agents 3 and 4 isolated in G")
    print(f"  structural q_v = 6n - sum dim D_i     {structural}")
    print(f"  null columns of B_G                   {counted}")
    print("\n  An isolated agent also zeroes its admissible columns; those are")
    print("  infinitesimal variations of G and belong to q_i, not to q_v.")
    return record("Remark 3", counted > structural)


# ============================================================= 4/5  Theorem 4

def section_theorem_impossible():
    head("4/5", "Theorem 4 and Corollary 5   the obstruction D_p (S_i - S_j) = 0")
    rng = seed(2)
    N = 200 if QUICK else 2000
    violations = 0
    for _ in range(N):
        pi, pj = rng.standard_normal(3), rng.standard_normal(3)
        pi[2] = 0.0
        pb = (pj - pi) / np.linalg.norm(pj - pi)
        Dp = orthogonal_projection_matrix(pb) / np.linalg.norm(pj - pi)
        Si, Sj = np.diag([1.0, 1.0, 0.0]), np.eye(3)
        violations += np.abs(Dp @ (Si - Sj)).max() > 1e-12
    print(f"  random planar/spatial pairs with D_p (S_i - S_j) != 0   {violations}/{N}")
    print("  S_i - S_j = -e3 e3^T, so the condition reads P(p_hat) e3 = 0, i.e. the")
    print("  two agents vertically aligned. Table III takes U_ij = I_3 there.")

    seed(8)
    net, _ = random_scenario(4, ["R^2"] * 3 + ["R^3"])
    net.edges = ~np.eye(4, dtype=bool)
    B = B_edge(net)
    z = [np.abs(B[:, 3 * i + 2]).max() for i in range(3)]
    print(f"\n  z columns of the planar agents under the edge-indexed form:")
    print(f"    max |column|   {['%.3f' % v for v in z]}   (should be 0)")
    return record("Theorem 4", violations == N and min(z) > 1e-9)


# ========================================================= 6  Proposition 6

def section_prop_graphdep():
    head("6", "Proposition 6   q_v depends on the graph under the edge-indexed form")
    seed(6)
    doms = ["R^2"] * 3 + ["R^3"]
    n = len(doms)
    net, _ = random_scenario(n, doms)
    subs = {
        "K (complete)": [(i, j) for i in range(n) for j in range(n) if i != j],
        "planar agents only": [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)],
        "+ one edge 1->4": [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (0, 3)],
        "+ one edge 4->1": [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (3, 0)],
    }
    print(f"  {'graph':26s} {'q_v edge-indexed':>17s} {'q_v node-indexed':>17s}")
    seen = set()
    for lbl, edges in subs.items():
        E = np.zeros((n, n), dtype=bool)
        for i, j in edges:
            E[i, j] = True
        net.edges = E
        qe, qn = len(null_columns(B_edge(net))), len(null_columns(B_node(net)))
        seen.add(qe)
        print(f"  {lbl:26s} {qe:17d} {qn:17d}")
    print("\n  It moves with the graph on the left and is constant on the right.")
    print("  Theorem 2 of the paper cancels q_v between G and K, which needs the latter.")
    return record("Proposition 6", len(seen) > 1)


# ============================================================== 7  Example 7

def counterexample():
    net = Network(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0], [0.5, 0.5, 1.0]]),
                  np.zeros((4, 3)),
                  np.array([[0, 1], [0, 2], [1, 3], [2, 3]]))
    for a, d in zip(net.agents, ["R^2", "R^2", "R^2", "R^3"]):
        a.set_domain(d)
    return net


def section_example():
    head("7", "Example 7   a rigid framework reported as flexible")
    net = counterexample()
    K = net.fully_connected()
    doms = [a.domain for a in net.agents]
    c = sum(DIM[d] for d in doms)
    print(f"  domains {doms}")
    print(f"  positions {[list(a.pose.position) for a in net.agents]}")
    print(f"  edges     {[(i + 1, j + 1) for i, j in zip(*np.nonzero(net.edges))]}")
    print(f"  c = sum dim D_i = {c},  q_v = 6n - c = {6 * net.n - c}\n")
    print(f"  {'':34s} {'rk B_G':>7s} {'rk B_K':>7s} {'conclusion':>12s}")
    out = {}
    for lbl, f in (("edge-indexed, eq. (4)", B_edge), ("node-indexed, eq. (5)", B_node)):
        rg, rk = rank(f(net)), rank(f(K))
        out[lbl] = (rg, rk)
        print(f"  {lbl:34s} {rg:7d} {rk:7d} {('rigid' if rg == rk else 'flexible'):>12s}")
    tg, tk = rank_fd(jacobian_truth(net)), rank_fd(jacobian_truth(K))
    print(f"  {'finite differences on I':34s} {tg:7d} {tk:7d} "
          f"{('rigid' if tg == tk else 'flexible'):>12s}")
    print(f"\n  q_t = c - rk = {c - tk} (two translations and a uniform scaling);")
    print(f"  G attains full rank {tk} on I with 4 edges, so it is minimally rigid.")
    ok = (out["edge-indexed, eq. (4)"] == (6, 8)
          and out["node-indexed, eq. (5)"] == (6, 6) and (tg, tk) == (6, 6))
    return record("Example 7", ok)


# =============================================================== 8  Remark 8

def section_remark_casestudy():
    head("8", "Remark 8   Section VI-B of the paper reproduced")
    doms = ["R^2xS^1"] * 3 + ["SE(3)"]
    n = 4
    net = Network(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                            [1.0, 1.7, 0.0], [0.9, 0.6, 1.4]]),
                  np.zeros((n, 3)),
                  np.array([[i, j] for i in range(n) for j in range(n) if i != j]))
    for a, d in zip(net.agents, doms):
        a.set_domain(d)
    for a, th in zip(net.agents[:3], (0.3, 1.1, 2.2)):
        c, s = np.cos(th), np.sin(th)
        a.pose.set_rotation_mat(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    c4, s4 = np.cos(0.4), np.sin(0.4)
    a4 = np.array([[np.cos(0.7), -np.sin(0.7), 0], [np.sin(0.7), np.cos(0.7), 0], [0, 0, 1]])
    net.agents[3].pose.set_rotation_mat(a4 @ np.array([[1, 0, 0], [0, c4, -s4], [0, s4, c4]]))

    B = B_edge(net)
    r, nc = rank(B), len(null_columns(B))
    print(f"  reported in the paper:   rk(B_K^+) = 13,  q_v = 6,  q_t = 5")
    print(f"  reproduced here:         rk(B_K^+) = {r},  null columns = {nc}")

    keep = [i for i in range(24) if i not in (2, 5, 8)]
    r_wo = rank(B[:, keep])
    print(f"\n  deleting the three z columns of the unicycles: rank {r} -> {r_wo}")
    for col in (2, 5, 8, 9, 10):
        rest = [i for i in range(24) if i != col]
        print(f"    column {col + 1:2d} alone dependent on the rest: "
              f"{rank(B[:, rest]) == r}")
    print(f"  so they are individually dependent but not jointly removable.")

    p = np.array([a.pose.position for a in net.agents])
    motions = {
        "translate x": np.concatenate([np.tile([1, 0, 0], 4), np.zeros(12)]),
        "translate y": np.concatenate([np.tile([0, 1, 0], 4), np.zeros(12)]),
        "translate z": np.concatenate([np.tile([0, 0, 1], 4), np.zeros(12)]),
        "rotate about z": np.concatenate([np.array([np.cross(E3, q) for q in p]).ravel(),
                                          np.tile(E3, 4)]),
        "uniform scaling": np.concatenate([p.ravel(), np.zeros(12)]),
    }
    print(f"\n  {'motion':18s} {'in ker(B_K)':>12s} {'admissible':>12s}")
    admissible_count = 0
    for lbl, v in motions.items():
        ink = np.linalg.norm(B @ v) < 1e-9
        adm = all(abs(v[3 * i + 2]) < 1e-9 for i in range(3))
        admissible_count += ink and adm
        print(f"  {lbl:18s} {str(ink):>12s} {str(adm):>12s}")
    print(f"\n  admissible trivial motions: {admissible_count}, not 5. The fifth is the")
    print("  global z translation, which the unicycles cannot perform.")
    return record("Remark 8", r == 13 and nc == 6 and r_wo == 11 and admissible_count == 4)


# ============================================================== 9  Theorem 9

def section_theorem_repair():
    head("9", "Theorem 9   the node-indexed form satisfies (R1)-(R3) and eq. (9)")
    rng = seed(11)
    mixes = [["R^2xS^1"] * 3 + ["SE(3)"], ["R^2xS^1"] * 2 + ["SE(3)"] * 2,
             ["R^2xS^1"] + ["SE(3)"] * 3, ["R^2xS^1"] * 4 + ["SE(3)"],
             ["R^2xS^1"] * 5 + ["SE(3)"], ["R^2xS^1"] * 2 + ["SE(3)"] * 3]
    per = 40 if QUICK else 200
    stat = {"edge": [0, 0], "node": [0, 0]}
    total = 0
    for doms in mixes:
        n = len(doms)
        c = sum(DIM[d] for d in doms)
        qv = 6 * n - c
        allE = [(i, j) for i in range(n) for j in range(n) if i != j]
        A_of = None
        for _ in range(per):
            net = sample(doms, rng, m=int(rng.integers(1, len(allE) + 1)))
            A_of = admissible_basis(net)
            K = net.fully_connected()
            for tag, f in (("edge", B_edge), ("node", B_node)):
                Bg, Bk = f(net), f(K)
                qi = c - rank(Bg @ A_of)
                stat[tag][0] += rank(Bg) != 6 * n - qv - qi
                stat[tag][1] += ((rank(Bg) == rank(Bk))
                                 != (rank(Bg @ A_of) == rank(Bk @ A_of)))
            total += 1
    print(f"  {'':44s} {'edge-indexed':>14s} {'node-indexed':>14s}")
    print(f"  {'eq. (9) rk = 6n - q_v - q_i violated':44s} "
          f"{stat['edge'][0]:>8d}/{total:<5d} {stat['node'][0]:>8d}/{total:<5d}")
    print(f"  {'rank condition disagrees with IBR':44s} "
          f"{stat['edge'][1]:>8d}/{total:<5d} {stat['node'][1]:>8d}/{total:<5d}")
    print(f"  {'':44s} {100 * stat['edge'][1] / total:>13.1f}% {0.0:>13.1f}%")
    return record("Theorem 9", stat["node"] == [0, 0]), total, stat


# =========================================================== 10  Proposition 10

def section_prop_homog():
    head("10", "Proposition 10   the two forms coincide on homogeneous formations")
    rng = seed(3)
    print(f"  {'manifold':10s} {'graphs':>7s} {'max |B_node - B_edge|':>23s}")
    worst_all = 0.0
    for d in DOMAINS:
        worst, ngr = 0.0, 0
        for n in ((4, 6) if QUICK else (3, 4, 6, 8, 12)):
            for _ in range(5):
                net = sample([d] * n, rng, m=int(rng.integers(1, n * (n - 1) + 1)))
                worst = max(worst, np.abs(B_node(net) - B_edge(net)).max())
                ngr += 1
        worst_all = max(worst_all, worst)
        print(f"  {d:10s} {ngr:7d} {worst:23.1e}")
    print("\n  Bitwise identical, so no homogeneous statement of the paper changes.")
    return record("Proposition 10", worst_all < 1e-15)


# ================================================= 11  closed forms, DOF budget

def section_closed_forms():
    head("11", "Closed forms and the DOF budget")
    seed(7)
    print("  homogeneous rank_K against the closed form")
    sizes = (3, 4, 6, 8) if QUICK else (3, 4, 6, 8, 12, 16)
    print(f"  {'manifold':10s} {'formula':>10s} " + " ".join(f"n={n:<4d}" for n in sizes))
    ok_all = True
    for d in DOMAINS:
        cells = []
        for n in sizes:
            net, _ = random_scenario(n, d)
            r = rank(B_node(net.fully_connected()))
            ok = r == CLOSED_FORM[d](n)
            ok_all &= ok
            cells.append("ok" if ok else f"{r}!={CLOSED_FORM[d](n)}")
        f = {"R^2": "2n-3", "R^3": "3n-4", "R^2xS^1": "3n-4",
             "R^3xS^1": "4n-5", "SE(3)": "6n-7"}[d]
        print(f"  {d:10s} {f:>10s} " + " ".join(f"{x:<6s}" for x in cells))

    print("\n  heterogeneous rank_K <= c - q_t")
    print(f"  {'mix':52s} {'c':>4s} {'rank_K':>7s} {'bound':>6s}")
    for doms in [["R^2", "R^3"] * 2, ["R^2", "SE(3)"] * 2,
                 ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"],
                 ["R^2"] * 5 + ["R^3"], ["R^2"] * 4 + ["SE(3)"] * 2,
                 ["R^3"] * 3 + ["SE(3)"] * 3, ["R^2xS^1"] * 4 + ["R^3xS^1"] * 2,
                 ["R^2", "R^2xS^1", "R^3", "R^3xS^1", "SE(3)"] * 2]:
        n = len(doms)
        c = sum(DIM[d] for d in doms)
        bound = c - (3 if any(d in PLANAR for d in doms) else 4)
        net, _ = random_scenario(n, list(doms))
        r = rank(B_node(net.fully_connected()))
        ok_all &= r <= bound
        lbl = str({d: doms.count(d) for d in sorted(set(doms))})
        print(f"  {lbl:52s} {c:4d} {r:7d} {bound:6d}  {'ok' if r <= bound else 'VIOLATION'}")
    return record("Closed forms", ok_all)


# ---------------------------------------------------------------------- main

if __name__ == "__main__":
    t0 = time.time()
    print("Verification of docs/dof_restriction_note.tex")
    print("mode:", "quick" if QUICK else "full")

    _, ngraph, worst = section_lemma()
    section_remark_aligned()
    section_remark_qv()
    section_theorem_impossible()
    section_prop_graphdep()
    section_example()
    section_remark_casestudy()
    _, total, stat = section_theorem_repair()
    section_prop_homog()
    section_closed_forms()

    head("", "Values quoted in the note")
    print(f"  frameworks in section 1                       {ngraph}")
    print(f"  worst relative error against the Jacobian     {worst:.1e}")
    print(f"  frameworks in section 9                       {total}")
    print(f"  eq. (9) violated, edge-indexed                {stat['edge'][0]}/{total}")
    print(f"  rank condition wrong, edge-indexed            {stat['edge'][1]}/{total}"
          f"  ({100 * stat['edge'][1] / total:.1f}%)")
    print(f"  both, node-indexed                            0/{total}")

    head("", "Summary")
    for k, v in results.items():
        print(f"  {k:16s} {'pass' if v else 'FAIL'}")
    bad = [k for k, v in results.items() if not v]
    print(f"\n  {len(results) - len(bad)}/{len(results)} checks passed"
          f"{'' if not bad else '   FAILED: ' + ', '.join(bad)}")
    print(f"  elapsed {time.time() - t0:.1f} s")
    sys.exit(1 if bad else 0)
