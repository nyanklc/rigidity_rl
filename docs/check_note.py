"""Verify every numerical claim of `dof_restriction_note.pdf`, section by section.

Labels below follow the note's own numbering; it renumbers when environments are
added, so check them against the source if a row looks misfiled.

Three objects are compared:
  (a) edge-indexed   B^x = [Dp U Ebar^T | Da V Ebar_o^T]        eq. (4), Prop. 2 of [1]
  (b) node-indexed   B^. = [Dp Ebar^T Sbar | Da Ebar_o^T Pbar]  eq. (5) of the note
  (c) a central-difference Jacobian of the bearing map along admissible variations,
      which uses neither construction and is therefore the ground truth.

Every check prints the value the note states next to the value measured here, so a
disagreement is visible rather than implied. Deterministic: one seed, no sampling
outside it.

    python3 check_note.py            # everything (~3 min)
    python3 check_note.py --quick    # worked examples only, skip both sweeps
"""
import argparse
from itertools import permutations

import numpy as np
from scipy.linalg import expm

I3 = np.eye(3)
SEED = 1

# ------------------------------------------------------------------ primitives

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def P_(u):
    u = u / np.linalg.norm(u)
    return I3 - np.outer(u, u)


def rank(A, rtol=1e-9):
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > rtol * s[0]).sum())


def rank_fd(A):
    """finite differences carry ~1e-10 noise, so the cut has to be looser"""
    return rank(A, rtol=1e-5)


def dom(d, v=None):
    """(S_i, P_i): the translational and rotational projectors of Table I."""
    if d == 'R2':    return np.diag([1., 1, 0]), np.zeros((3, 3))
    if d == 'R3':    return I3.copy(), np.zeros((3, 3))
    if d == 'SE2':   return np.diag([1., 1, 0]), np.outer(I3[:, 2], I3[:, 2])
    if d == 'R3xS1': return I3.copy(), np.outer(v, v)
    if d == 'SE3':   return I3.copy(), I3.copy()
    raise ValueError(d)


CLOSED_FORM = {'R2': lambda n: 2*n - 3, 'R3': lambda n: 3*n - 4,
               'SE2': lambda n: 3*n - 4, 'R3xS1': lambda n: 4*n - 5,
               'SE3': lambda n: 6*n - 7}


def basis(M):
    if np.allclose(M, 0):
        return np.zeros((3, 0))
    u, s, _ = np.linalg.svd(M)
    return u[:, s > 1e-9]


def admissible_basis(doms, axes):
    """A in R^{6n x c} with orthonormal columns spanning A of eq. (3)."""
    n = len(doms)
    cols = []
    for i, d in enumerate(doms):
        S, Pr = dom(d, axes[i])
        for b in basis(S).T:
            e = np.zeros(6 * n); e[3*i:3*i+3] = b; cols.append(e)
        for b in basis(Pr).T:
            e = np.zeros(6 * n); e[3*n+3*i:3*n+3*i+3] = b; cols.append(e)
    return np.array(cols).T


def Dp_Da(p, R, i, j):
    pij = p[j] - p[i]
    s = 1.0 / np.linalg.norm(pij)
    ph = pij * s
    return s * R[i].T @ P_(ph), -R[i].T @ skew(ph)


def B_node(p, R, doms, edges, axes):
    """eq. (5): the factors indexed by node."""
    n, m = len(doms), len(edges)
    B = np.zeros((3 * m, 6 * n))
    for k, (i, j) in enumerate(edges):
        Dp, Da = Dp_Da(p, R, i, j)
        Si, Pi = dom(doms[i], axes[i])
        Sj, _ = dom(doms[j], axes[j])
        r = slice(3 * k, 3 * k + 3)
        B[r, 3*j:3*j+3] += Dp @ Sj
        B[r, 3*i:3*i+3] -= Dp @ Si
        B[r, 3*n+3*i:3*n+3*i+3] -= Da @ Pi
    return B


def U_edge(di, dj):
    """Table I / Table III: U_ij restricts to the manifold the measurement lives in,
    so it is I_3 as soon as either endpoint is spatial (Table III's choice)."""
    planar = {'R2', 'SE2'}
    return np.diag([1., 1, 0]) if (di in planar and dj in planar) else I3.copy()


def B_edge(p, R, doms, edges, axes):
    """eq. (4) = Proposition 2 of [1]: one factor U_ij for the whole edge."""
    n, m = len(doms), len(edges)
    B = np.zeros((3 * m, 6 * n))
    for k, (i, j) in enumerate(edges):
        Dp, Da = Dp_Da(p, R, i, j)
        U = U_edge(doms[i], doms[j])
        _, Pi = dom(doms[i], axes[i])
        r = slice(3 * k, 3 * k + 3)
        B[r, 3*j:3*j+3] += Dp @ U
        B[r, 3*i:3*i+3] -= Dp @ U
        B[r, 3*n+3*i:3*n+3*i+3] -= Da @ Pi
    return B


def bearings(p, R, edges):
    return np.concatenate([R[i].T @ ((p[j]-p[i]) / np.linalg.norm(p[j]-p[i]))
                           for (i, j) in edges])


def B_fd(p, R, doms, edges, axes, eps=1e-6):
    """Ground truth: central differences, attitudes moved as R_i -> exp([eps w]_x) R_i."""
    cols = []
    for i, d in enumerate(doms):
        S, Pr = dom(d, axes[i])
        for b in basis(S).T:
            pp, pm = p.copy(), p.copy()
            pp[i] += eps * b; pm[i] -= eps * b
            cols.append((bearings(pp, R, edges) - bearings(pm, R, edges)) / (2*eps))
        for b in basis(Pr).T:
            Rp, Rm = R.copy(), R.copy()
            Rp[i] = expm(eps * skew(b)) @ R[i]
            Rm[i] = expm(-eps * skew(b)) @ R[i]
            cols.append((bearings(p, Rp, edges) - bearings(p, Rm, edges)) / (2*eps))
    return np.array(cols).T


def complete(n):
    return list(permutations(range(n), 2))


def nullcols(B):
    return int(np.sum(np.all(np.abs(B) < 1e-12, axis=0)))


# ------------------------------------------------------------------- reporting

ROWS = []


def claim(where, what, stated, measured, ok):
    ROWS.append((where, what, str(stated), str(measured), "ok" if ok else "FAIL"))
    return ok


# ------------------------------------------------------------------ the checks

def three_agent_example():
    """Summary table and Example 4: the formation the note asks you to verify by hand."""
    print("\n" + "=" * 78)
    print("Example 4 / Summary table -- three agents, two planar and one spatial")
    print("=" * 78)
    p = np.array([[0., 0, 0], [2, 0, 0], [1, 1, 1]])
    doms, axes = ['R2', 'R2', 'R3'], [None] * 3
    R = np.tile(I3, (3, 1, 1))
    E, K = [(0, 2), (1, 2)], complete(3)
    A = admissible_basis(doms, axes)
    c = A.shape[1]
    print(f"  p1={p[0]}, p2={p[1]}, p3={p[2]},  E={{(1,3),(2,3)}}")
    claim("sec.4", "c (degrees of freedom)", 7, c, c == 7)

    for nm, f, sG, sK in [("edge-indexed, eq.(4)", B_edge, 4, 5),
                          ("node-indexed, eq.(5)", B_node, 4, 4)]:
        rG, rK = rank(f(p, R, doms, E, axes)), rank(f(p, R, doms, K, axes))
        verdict = 'rigid' if rG == rK else 'FLEXIBLE'
        print(f"  {nm:22s} rk B_G={rG}  rk B_K={rK}  -> {verdict}")
        claim("Summary", f"{nm} (rk B_G, rk B_K)", (sG, sK), (rG, rK), (rG, rK) == (sG, sK))

    rG = rank_fd(B_fd(p, R, doms, E, axes)); rK = rank_fd(B_fd(p, R, doms, K, axes))
    print(f"  {'finite differences':22s} rk B_G={rG}  rk B_K={rK}  -> "
          f"{'rigid' if rG == rK else 'FLEXIBLE'}")
    claim("Summary", "finite differences (rk B_G, rk B_K)", (4, 4), (rG, rK), (rG, rK) == (4, 4))

    # the hand argument: ker J = {u_i = w + t p_i, w horizontal}
    J = B_fd(p, R, doms, E, axes)
    _, _, vt = np.linalg.svd(J)
    ker = vt[rank_fd(J):]
    claim("sec.4", "dim ker J_G", 3, ker.shape[0], ker.shape[0] == 3)
    pos = (A @ ker.T).T[:, :9].reshape(-1, 3, 3)
    worst = 0.0
    for U in pos:
        t = U[2, 2]
        w = U[2] - t * p[2]
        worst = max(worst, np.abs(U - np.array([w + t*p[i] for i in range(3)])).max(),
                    abs(w[2]))
    print(f"  every kernel mode has the form w + t*p_i with w horizontal, to {worst:.1e}")
    claim("sec.4", "ker J = {w + t p_i, w horizontal}", "exact", f"{worst:.1e}", worst < 1e-6)

    # the note displays D_p^(1) explicitly for the reader to check by hand
    Dp1, _ = Dp_Da(p, R, 0, 2)
    stated = (np.sqrt(3) / 9) * np.array([[2., -1, -1], [-1, 2, -1], [-1, -1, 2]])
    d = float(np.abs(Dp1 - stated).max())
    claim("sec.4", "D_p^(1) = (sqrt3/9)[[2,-1,-1],[-1,2,-1],[-1,-1,2]]", "exact",
          f"{d:.1e}", d < 1e-12)

    # each of the two edges contributes rank 2, so G is minimally rigid
    per_edge = [rank(B_node(p, R, doms, [e], axes)) for e in E]
    claim("sec.4", "rank contributed per edge", [2, 2], per_edge, per_edge == [2, 2])


def second_mechanism():
    """Remark 5: the null-column COUNT itself becomes graph-dependent."""
    print("\n" + "=" * 78)
    print("Remark 5 -- a second mechanism, at n=4")
    print("=" * 78)
    p = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [.5, .5, 1]])
    doms, axes = ['R2', 'R2', 'R2', 'R3'], [None] * 4
    R = np.tile(I3, (4, 1, 1))
    E, K = [(0, 1), (0, 2), (1, 3), (2, 3)], complete(4)
    A = admissible_basis(doms, axes)
    claim("Rem.4", "c", 9, A.shape[1], A.shape[1] == 9)

    BxG, BxK = B_edge(p, R, doms, E, axes), B_edge(p, R, doms, K, axes)
    nG, nK = nullcols(BxG), nullcols(BxK)
    print(f"  null columns, edge-indexed:  G={nG}  K={nK}")
    claim("Rem.4", "null columns (G, K), edge-indexed", (13, 12), (nG, nK), (nG, nK) == (13, 12))
    rG, rK = rank(BxG), rank(BxK)
    print(f"  rk B^x_G = {rG}   rk B^x_K = {rK}")
    claim("Rem.4", "(rk B^x_G, rk B^x_K)", (6, 8), (rG, rK), (rG, rK) == (6, 8))

    keep = [k for k in range(24) if k not in [3*i + 2 for i in range(3)]]
    rdel = rank(BxK[:, keep])
    print(f"  after deleting the three planar z columns of B^x_K: {rdel}")
    claim("Rem.4", "rk B^x_K after deleting planar z cols", 6, rdel, rdel == 6)

    rn = rank(B_node(p, R, doms, E, axes))
    claim("Rem.4", "minimally rigid: c - q_t = 9 - 3", 6, rn, rn == 6)


def case_study_VIB(rng):
    """Remark 6: reproduce the numbers reported in Section VI-B of [1]."""
    print("\n" + "=" * 78)
    print("Remark 6 -- Section VI-B of [1]: three unicycles + one SE(3), complete graph")
    print("=" * 78)
    ranks, nulls, dels, cs = [], [], [], []
    for _ in range(3):
        q = np.zeros((4, 3))
        q[:3, :2] = rng.normal(size=(3, 2)); q[3] = rng.normal(size=3)
        dd, ax = ['SE2', 'SE2', 'SE2', 'SE3'], [None] * 4
        RR = np.stack([expm(skew([0, 0, t])) for t in rng.uniform(0, 6, 3)]
                      + [expm(skew(rng.normal(size=3)))])
        Bx = B_edge(q, RR, dd, complete(4), ax)
        keep = [k for k in range(24) if k not in (2, 5, 8)]
        ranks.append(rank(Bx)); nulls.append(nullcols(Bx)); dels.append(rank(Bx[:, keep]))
        cs.append(admissible_basis(dd, ax).shape[1])
    print(f"  rk B^+_K = {ranks}, null columns = {nulls}, after deleting unicycle z = {dels}")
    claim("Rem.5", "rk B^+_K (reported: 13)", 13, ranks, all(r == 13 for r in ranks))
    claim("Rem.5", "null columns (reported: 6)", 6, nulls, all(x == 6 for x in nulls))
    claim("Rem.5", "rk after deleting unicycle z = c - q_t = 15-4", 11, dels,
          all(x == 11 for x in dels))
    claim("Rem.5", "c", 15, cs, all(x == 15 for x in cs))


# 16 heterogeneous mixes, fixed so the sweep is reproducible
MIXES = [
    ['R2','R3'], ['R2','SE2'], ['R2','R3xS1'], ['R2','SE3'],
    ['R3','SE2'], ['R3','R3xS1'], ['R3','SE3'], ['SE2','R3xS1'],
    ['SE2','SE3'], ['R3xS1','SE3'], ['R2','R3','SE2'], ['R2','R3','SE3'],
    ['R2','SE2','R3xS1'], ['R3','R3xS1','SE3'], ['R2','R3','SE2','SE3'],
    ['R2','R3','SE2','R3xS1','SE3'],
]
DENSITIES = (0.35, 0.55, 0.75, 1.0)


def _frame(rng, doms, density):
    n = len(doms)
    q = rng.normal(size=(n, 3))
    for i, d in enumerate(doms):
        if d in ('R2', 'SE2'):
            q[i, 2] = 0.0
    # a common rotation axis: the closed form 4n-5 for R^3xS^1 presupposes one,
    # since without it there is no coordinated-rotation trivial motion
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    ax = [v] * n
    RR = np.stack([expm(skew(rng.normal(size=3))) for _ in range(n)])
    Kn = complete(n)
    E = Kn if density >= 1.0 else ([e for e in Kn if rng.random() < density] or Kn[:2])
    return q, RR, ax, E, Kn


def sweep_F_Z(rng):
    """The (F)/(Z) sweep: 5 manifolds at n=3,4,6,9 plus 16 mixes, 4 densities, 3 reps."""
    print("\n" + "=" * 78)
    print("Numerical checks -- (F) and (Z) over homogeneous and heterogeneous frameworks")
    print("=" * 78)
    configs = [[d] * n for d in ['R2','R3','SE2','R3xS1','SE3'] for n in (3,4,6,9)]
    configs += [m * 2 if len(m) < 4 else m for m in MIXES]
    wF = wZn = wZx = wU = 0.0
    count = 0
    for doms in configs:
        for density in DENSITIES:
            for _ in range(3):
                q, RR, ax, E, Kn = _frame(rng, doms, density)
                n = len(doms)
                A = admissible_basis(doms, ax)
                J = B_fd(q, RR, doms, E, ax)
                Bn, Bx = B_node(q, RR, doms, E, ax), B_edge(q, RR, doms, E, ax)
                nJ = max(np.linalg.norm(J), 1e-12)
                wF = max(wF, np.linalg.norm(Bn @ A - J)/nJ, np.linalg.norm(Bx @ A - J)/nJ)
                Pperp = np.eye(6*n) - A @ A.T
                wZn = max(wZn, np.linalg.norm(Bn @ Pperp))
                wZx = max(wZx, np.linalg.norm(Bx @ Pperp))
                # exact identity, no finite differences involved
                wU = max(wU, np.abs(Bn - Bx @ A @ A.T).max())
                count += 1
    print(f"  {count} frameworks   (the note states 432)")
    claim("Num.", "sweep size", 432, count, count == 432)
    claim("Num.", "(F) worst rel. error, both forms", "4.8e-10",
          f"{wF:.1e}", f"{wF:.1e}" == "4.8e-10")
    claim("Num.", "(Z) ||B^.(I-AA^T)|| is machine zero", "3.2e-15",
          f"{wZn:.1e}", f"{wZn:.1e}" == "3.2e-15")
    claim("Num.", "(Z) same quantity for eq.(4) is order one", ">0.1",
          f"{wZx:.1e}", wZx > 0.1)
    claim("Prop.6", "B^. = B^x A A^T exactly", "0", f"{wU:.1e}", wU < 1e-12)


def homogeneous_agreement(rng):
    """Proposition 9: the two constructions coincide, and the ranks are the closed forms."""
    print("\n" + "=" * 78)
    print("Proposition 9 -- homogeneous formations")
    print("=" * 78)
    worst, cf_bad, graphs = 0.0, 0, 0
    for d in ['R2', 'R3', 'SE2', 'R3xS1', 'SE3']:
        for n in (3, 4, 6, 8, 12):
            for _ in range(5):
                q, RR, ax, _, Kn = _frame(rng, [d] * n, 1.0)
                worst = max(worst, np.abs(B_edge(q, RR, [d]*n, Kn, ax)
                                          - B_node(q, RR, [d]*n, Kn, ax)).max())
                cf_bad += rank(B_node(q, RR, [d]*n, Kn, ax)) != CLOSED_FORM[d](n)
                graphs += 1
    print(f"  {graphs} complete graphs, five manifolds at n=3,4,6,8,12")
    claim("Prop.8", "the two forms agree bitwise", "0", f"{worst:.1e}", worst == 0.0)
    claim("Prop.8", "rk B_K equals the closed form (failures)", 0, cf_bad, cf_bad == 0)


def randomised(rng, N=1200):
    """Numerical checks: does each construction's rank test match the ground truth?"""
    print("\n" + "=" * 78)
    print(f"Numerical checks -- {N} random heterogeneous frameworks, n in {{4,5,6}}")
    print("=" * 78)
    choices = ['R2', 'R3', 'SE2', 'R3xS1', 'SE3']
    bad_x = bad_n = viol_x = viol_n = used = 0
    for _ in range(N):
        n = int(rng.integers(4, 7))
        dd = [choices[int(rng.integers(0, 5))] for _ in range(n)]
        if len(set(dd)) == 1:
            dd[0] = 'R2' if dd[0] != 'R2' else 'SE3'
        ax = [a / np.linalg.norm(a) for a in rng.normal(size=(n, 3))]
        q = rng.normal(size=(n, 3))
        for i, d in enumerate(dd):
            if d in ('R2', 'SE2'):
                q[i, 2] = 0.0
        RR = np.stack([expm(skew(rng.normal(size=3))) for _ in range(n)])
        Kn = complete(n)
        E = [e for e in Kn if rng.random() < 0.45]
        if len(E) < 2:
            continue
        used += 1
        A = admissible_basis(dd, ax)
        c, qv = A.shape[1], 6*n - A.shape[1]
        truth = rank_fd(B_fd(q, RR, dd, E, ax)) == rank_fd(B_fd(q, RR, dd, Kn, ax))
        bad_x += (rank(B_edge(q, RR, dd, E, ax)) == rank(B_edge(q, RR, dd, Kn, ax))) != truth
        bad_n += (rank(B_node(q, RR, dd, E, ax)) == rank(B_node(q, RR, dd, Kn, ax))) != truth
        for f, acc in ((B_edge, 'x'), (B_node, 'n')):
            B = f(q, RR, dd, E, ax)
            qi = c - rank(B @ A)
            bad = rank(B) != 6*n - qv - qi
            if acc == 'x': viol_x += bad
            else:          viol_n += bad
    print(f"  {used} of {N} draws had at least two edges")
    claim("Num.", "rank formula fails, edge-indexed", "504/1198",
          f"{viol_x}/{used}", f"{viol_x}/{used}" == "504/1198")
    claim("Num.", "rank formula fails, node-indexed", "0/1198", f"{viol_n}/{used}", viol_n == 0)
    claim("Num.", "rank test disagrees, edge-indexed", "67/1198",
          f"{bad_x}/{used}", f"{bad_x}/{used}" == "67/1198")
    claim("Num.", "rank test disagrees, node-indexed", "0/1198", f"{bad_n}/{used}", bad_n == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="only the worked examples; skip both sweeps")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    three_agent_example()
    second_mechanism()
    case_study_VIB(rng)
    if not args.quick:
        sweep_F_Z(rng)
        homogeneous_agreement(rng)
        randomised(rng)

    print("\n" + "=" * 78)
    print(f"  {'where':10s} {'claim':52s} {'note':>12s} {'measured':>14s}  ")
    print("-" * 78)
    for where, what, stated, got, status in ROWS:
        print(f"  {where:10s} {what:52.52s} {stated:>12.12s} {got:>14.14s}  {status}")
    fails = [r for r in ROWS if r[4] == "FAIL"]
    print("-" * 78)
    print(f"  {len(ROWS) - len(fails)}/{len(ROWS)} checks reproduce"
          + (f"   FAILURES: {len(fails)}" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
