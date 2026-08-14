"""
Independent verification of the claims in `rigidity_correction.pdf` against
Michieletto, Cenedese & Zelazo, "A Unified Dissertation on Bearing Rigidity
Theory", IEEE TCNS 8(4):1624-1636, 2021.

Conventions (all taken from [1]):
    b_ij   = R_i^T phat_ij ,  phat_ij = (p_j-p_i)/||p_j-p_i||, s_ij = 1/||p_j-p_i||
    [E]_ik = -1 if e_k=(v_i,v_j), +1 if e_k=(v_j,v_i), 0 otherwise
    [Eo]_ik= -1 if e_k=(v_i,v_j), 0 otherwise
    Ebar   = E (x) I3,  Eobar = Eo (x) I3
    Dp     = blkdiag(s_ij R_i^T P(phat_ij)),  Da = blkdiag(-R_i^T [phat_ij]_x)
    per-edge (Prop. 2 of [1]):  B^x = [ Dp U Ebar^T ,  Da V Eobar^T ]
    per-node (Def. 3 of note):  B^. = [ Dp Ebar^T Sbar , Da Eobar^T Pbar ]
    delta^+ = [delta_p ; delta_a] in R^{6n}, world-frame angular velocities.
"""
import itertools
import numpy as np

I3 = np.eye(3)
E1, E2, E3 = np.eye(3)

# ----------------------------------------------------------------------------- utils
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def expSO3(w):
    th = np.linalg.norm(w)
    if th < 1e-14:
        return I3 + skew(w)
    K = skew(w / th)
    return I3 + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def Rz(a): return expSO3(np.array([0, 0, a]))
def Rx(a): return expSO3(np.array([a, 0, 0]))
def Pperp(x): return I3 - np.outer(x, x)          # x unit

def rank(M, tol=1e-9):
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())

def nullspace(M, tol=1e-9):
    u, s, vt = np.linalg.svd(M)
    s_full = np.zeros(vt.shape[0]); s_full[:len(s)] = s
    return vt[s_full <= tol * max(1.0, s[0] if len(s) else 1.0)].T

# ------------------------------------------------------------------- agent domains
# S = projector on admissible translations, P = projector on admissible rotations,
# TableI_U / TableI_V = the entries of Table I of [1] for the homogeneous case.
def domain(name, v=None):
    if name == 'R2':
        return dict(c=2, S=np.diag([1., 1., 0.]), P=np.zeros((3, 3)),
                    U=np.diag([1., 1., 0.]), V=np.zeros((3, 3)))
    if name == 'R3':
        return dict(c=3, S=I3, P=np.zeros((3, 3)), U=I3, V=np.zeros((3, 3)))
    if name == 'R2S1':
        return dict(c=3, S=np.diag([1., 1., 0.]), P=np.outer(E3, E3),
                    U=np.diag([1., 1., 0.]), V=np.column_stack([np.zeros(3), np.zeros(3), E3]))
    if name == 'R3S1':
        v = E3 if v is None else v / np.linalg.norm(v)
        return dict(c=4, S=I3, P=np.outer(v, v), U=I3,
                    V=np.column_stack([np.zeros(3), np.zeros(3), v]), v=v)
    if name == 'SE3':
        return dict(c=6, S=I3, P=I3, U=I3, V=I3)
    raise ValueError(name)

# ------------------------------------------------------------------- framework core
def incidence(n, edges):
    m = len(edges)
    E = np.zeros((n, m)); Eo = np.zeros((n, m))
    for k, (i, j) in enumerate(edges):
        E[i, k] = -1.0; E[j, k] = +1.0; Eo[i, k] = -1.0
    return np.kron(E, I3), np.kron(Eo, I3)

def DpDa(p, R, edges):
    m = len(edges)
    Dp = np.zeros((3 * m, 3 * m)); Da = np.zeros((3 * m, 3 * m))
    for k, (i, j) in enumerate(edges):
        d = p[j] - p[i]; nrm = np.linalg.norm(d); ph = d / nrm; s = 1.0 / nrm
        Dp[3*k:3*k+3, 3*k:3*k+3] = s * R[i].T @ Pperp(ph)
        Da[3*k:3*k+3, 3*k:3*k+3] = -R[i].T @ skew(ph)
    return Dp, Da

def B_pernode(p, R, edges, doms):
    n = len(p)
    Eb, Eob = incidence(n, edges)
    Dp, Da = DpDa(p, R, edges)
    Sb = np.zeros((3*n, 3*n)); Pb = np.zeros((3*n, 3*n))
    for i, d in enumerate(doms):
        Sb[3*i:3*i+3, 3*i:3*i+3] = d['S']; Pb[3*i:3*i+3, 3*i:3*i+3] = d['P']
    return np.hstack([Dp @ Eb.T @ Sb, Da @ Eob.T @ Pb])

def B_peredge(p, R, edges, doms, Ufun, Vfun):
    """Ufun(i,j)->U_ij, Vfun(i,j)->V_ij : the Prop.2/Table I/Table III form."""
    n, m = len(p), len(edges)
    Eb, Eob = incidence(n, edges)
    Dp, Da = DpDa(p, R, edges)
    U = np.zeros((3*m, 3*m)); V = np.zeros((3*m, 3*m))
    for k, (i, j) in enumerate(edges):
        U[3*k:3*k+3, 3*k:3*k+3] = Ufun(i, j)
        V[3*k:3*k+3, 3*k:3*k+3] = Vfun(i, j)
    return np.hstack([Dp @ U @ Eb.T, Da @ V @ Eob.T])

def admissible_basis(doms):
    """orthonormal A in R^{6n x c} spanning the admissible subspace of R^{6n}."""
    n = len(doms)
    cols = []
    for blk, key in ((0, 'S'), (1, 'P')):
        for i, d in enumerate(doms):
            M = d[key]
            r = rank(M)
            if r == 0:
                continue
            u, s, _ = np.linalg.svd(M)           # orthonormal basis of range(M)
            for t in range(r):
                c = np.zeros(6*n); c[blk*3*n + 3*i: blk*3*n + 3*i + 3] = u[:, t]
                cols.append(c)
    return np.column_stack(cols)

def bearings(p, R, edges):
    out = []
    for (i, j) in edges:
        d = p[j] - p[i]
        out.append(R[i].T @ (d / np.linalg.norm(d)))
    return np.concatenate(out)

def fd_jacobian(p, R, edges, doms, eps=1e-6):
    """Ground truth: central-difference differential of the bearing map,
       restricted to admissible variations (columns ordered as in admissible_basis)."""
    n = len(p)
    A = admissible_basis(doms)
    J = np.zeros((3*len(edges), A.shape[1]))
    for t in range(A.shape[1]):
        d = A[:, t]
        cols = []
        for sgn in (+1, -1):
            pp = [p[i] + sgn*eps*d[3*i:3*i+3] for i in range(n)]
            RR = [expSO3(sgn*eps*d[3*n+3*i: 3*n+3*i+3]) @ R[i] for i in range(n)]
            cols.append(bearings(pp, RR, edges))
        J[:, t] = (cols[0] - cols[1]) / (2*eps)
    return J, A

def complete_edges(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]

# ============================================================================ TESTS
print("="*78)
print("A. Example 9 of the note  (3 unicycles in R^2xS^1 + 1 fully actuated SE(3))")
print("="*78)
p = [np.array([0., 0., 0.]), np.array([2., 0., 0.]),
     np.array([1., 1.7, 0.]), np.array([0.9, 0.6, 1.4])]
R = [Rz(0.3), Rz(1.1), Rz(2.2), Rz(0.7) @ Rx(0.4)]
doms = [domain('R2S1'), domain('R2S1'), domain('R2S1'), domain('SE3')]
n = 4
G = [(0, 1), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)]   # 1-indexed in the note
K = complete_edges(n)
c = sum(d['c'] for d in doms); qv = 6*n - c
print(f"n={n}, c={c}, q_v (structural) = 6n-c = {qv}")

# Table III of [1]: U=I3 as soon as one endpoint is the aerial agent, else diag(1,1,0)
planar = [0, 1, 2]
Utab = lambda i, j: (np.diag([1., 1., 0.]) if (i in planar and j in planar) else I3)
Vtab = lambda i, j: (np.outer(E3, E3) if i in planar else I3)

Bx_G = B_peredge(p, R, G, doms, Utab, Vtab); Bx_K = B_peredge(p, R, K, doms, Utab, Vtab)
Bn_G = B_pernode(p, R, G, doms);             Bn_K = B_pernode(p, R, K, doms)
J_G, A = fd_jacobian(p, R, G, doms);         J_K, _ = fd_jacobian(p, R, K, doms)

print(f"  max |B_pernode*A - FD Jacobian| (G) = {np.abs(Bn_G@A - J_G).max():.2e}")
print(f"  max |B_peredge*A - FD Jacobian| (G) = {np.abs(Bx_G@A - J_G).max():.2e}")
print(f"  max |B_pernode*A - FD Jacobian| (K) = {np.abs(Bn_K@A - J_K).max():.2e}")
print(f"  max |B_peredge*A - FD Jacobian| (K) = {np.abs(Bx_K@A - J_K).max():.2e}")
print()
print(f"  per-edge   rk B_G = {rank(Bx_G):2d}   rk B_K = {rank(Bx_K):2d}"
      f"   -> verdict {'IBR' if rank(Bx_G)==rank(Bx_K) else 'NOT IBR'}")
print(f"  per-node   rk B_G = {rank(Bn_G):2d}   rk B_K = {rank(Bn_K):2d}"
      f"   -> verdict {'IBR' if rank(Bn_G)==rank(Bn_K) else 'NOT IBR'}")
print(f"  ground trth rk J_G = {rank(J_G):2d}   rk J_K = {rank(J_K):2d}"
      f"   -> verdict {'IBR' if rank(J_G)==rank(J_K) else 'NOT IBR'}")
qi = J_G.shape[1] - rank(J_G); qt = J_K.shape[1] - rank(J_K)
print(f"  q_i = dim ker(B_G|A) = {qi} , q_t = dim ker(B_K|A) = {qt}")
NG, NK = nullspace(J_G), nullspace(J_K)
# check ker(B_K|A) subset ker(B_G|A) and equality
proj = NG @ NG.T
print(f"  ker(B_K|A) contained in ker(B_G|A): residual "
      f"{np.abs(proj@NK - NK).max():.2e}  -> spaces equal: {qi==qt}")
print("  admissible trivial variations (in terms of the 15 admissible coords):")
print("    basis of ker(B_K|A), rounded:")
print(np.round(NK, 3).T)

print()
print("-"*78)
print("B. Remarks 10 & 11 (the phantom rank and the '5 dependent columns')")
print("-"*78)
z_cols = [2, 5, 8]                      # 0-indexed z-columns of agents 1,2,3
keep = [i for i in range(6*n) if i not in z_cols]
print(f"  rk B^x_K (per-edge, full R^24)            = {rank(Bx_K)}")
print(f"  rk B^x_K after deleting cols 3,6,9 (1-idx)= {rank(Bx_K[:, keep])}")
s = Bx_K[:, 2] + Bx_K[:, 5] + Bx_K[:, 8] + Bx_K[:, 11]
print(f"  ||c3+c6+c9+c12|| (global z-translation)   = {np.linalg.norm(s):.2e}")
def dep_residual(M, col):
    others = [i for i in range(M.shape[1]) if i != col]
    x = np.linalg.lstsq(M[:, others], M[:, col], rcond=None)[0]
    return np.linalg.norm(M[:, others] @ x - M[:, col])

for col in [2, 5, 8, 9, 10]:
    print(f"  column {col+1:2d} (paper's list) individually dependent on the others? "
          f" residual={dep_residual(Bx_K, col):.2e}")
print("  but so is EVERY translational column (3 global-translation relations):")
print("   ", " ".join(f"c{c+1}:{dep_residual(Bx_K, c):.0e}" for c in range(12)))
print(f"  null columns of B^x_K = {sum(np.linalg.norm(Bx_K[:,i])<1e-12 for i in range(24))}"
      f"   (paper reports 6; structural q_v = {qv})")
print(f"  null columns of B^._K = {sum(np.linalg.norm(Bn_K[:,i])<1e-12 for i in range(24))}"
      f"   (structural q_v = {qv})")
print(f"  identity rk = 6n-q_v-q_i :  per-edge G: {rank(Bx_G)} vs {6*n-qv-qi}"
      f" | per-node G: {rank(Bn_G)} vs {6*n-qv-qi}")
print(f"                              per-edge K: {rank(Bx_K)} vs {6*n-qv-qt}"
      f" | per-node K: {rank(Bn_K)} vs {6*n-qv-qt}")

print()
print("-"*78)
print("C. Theorem 6: no per-edge U_ij can satisfy (R1) fidelity and (R2) annihilation")
print("-"*78)
# mixed edge 1->4 (planar i, spatial j): solve for U in the 18 linear equations
i, j = 0, 3
d = p[j] - p[i]; ph = d/np.linalg.norm(d); s = 1/np.linalg.norm(d)
Dpk = s * R[i].T @ Pperp(ph)
Si, Sj = doms[i]['S'], doms[j]['S']
# unknown U (9 entries). Equations: Dpk U Si = Dpk Si ; Dpk U Sj = Dpk Sj ;
#                                   Dpk U (I-Si)=0 ; Dpk U (I-Sj)=0
rows, rhs = [], []
for Mright, target in ((Si, Dpk@Si), (Sj, Dpk@Sj),
                       (I3-Si, np.zeros((3, 3))), (I3-Sj, np.zeros((3, 3)))):
    rows.append(np.kron(Mright.T, Dpk)); rhs.append(target.flatten(order='F'))
Amat = np.vstack(rows); bvec = np.concatenate(rhs)
sol, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
print(f"  edge (1,4): least-squares residual over all 9 entries of U = "
      f"{np.linalg.norm(Amat@sol - bvec):.4f}   (0 would mean a valid U exists)")
print(f"  Theorem 6 obstruction ||Dp^(k)(S_i-S_j)|| = "
      f"{np.linalg.norm(Dpk@(Si-Sj)):.4f}   (vanishes iff phat_ij = +-e3)")
print(f"  phat_14 = {np.round(ph,3)}  -> not +-e3, as Corollary 7 requires")
# and the special aligned configuration where it DOES work
p_al = [np.array([0., 0., 0.]), np.array([2., 0., 0.]),
        np.array([1., 1.7, 0.]), np.array([0., 0., 1.4])]
d = p_al[3]-p_al[0]; ph2 = d/np.linalg.norm(d)
print(f"  vertically aligned pair: ||P(phat)(S_i-S_j)|| = "
      f"{np.linalg.norm(Pperp(ph2)@(Si-Sj)):.2e}  (Corollary 7 boundary case)")

print()
print("-"*78)
print("D. Proposition 13: homogeneous formations -> the two forms coincide")
print("-"*78)
rng = np.random.default_rng(0)
for name in ['R2', 'R3', 'R2S1', 'R3S1', 'SE3']:
    worst = 0.0
    for trial in range(60):
        nn = rng.integers(3, 7)
        d = domain(name)
        dd = [d]*nn
        pp = [rng.normal(size=3) for _ in range(nn)]
        if name in ('R2', 'R2S1'):
            pp = [np.array([q[0], q[1], 0.]) for q in pp]
        RR = []
        for _ in range(nn):
            if name in ('R2', 'R3'):
                RR.append(I3)
            elif name == 'R2S1':
                RR.append(Rz(rng.uniform(0, 6.28)))
            elif name == 'R3S1':
                RR.append(Rz(rng.uniform(0, 6.28)))          # v = e3
            else:
                RR.append(expSO3(rng.normal(size=3)))
        ed = [e for e in complete_edges(nn) if rng.random() < 0.6]
        if not ed:
            continue
        b1 = B_peredge(pp, RR, ed, dd, lambda a, b: d['U'], lambda a, b: d['V'])
        b2 = B_pernode(pp, RR, ed, dd)
        worst = max(worst, np.abs(b1-b2).max())
    print(f"  D = {name:5s}: max |B_peredge - B_pernode| over 60 random graphs = {worst:.1e}")
# the v != e3 caveat of Remark 14
dv = domain('R3S1', v=np.array([1., 1., 1.]))
pp = [rng.normal(size=3) for _ in range(4)]
RR = [expSO3(dv['v']*rng.uniform(0, 3)) for _ in range(4)]
dd = [dv]*4
ed = complete_edges(4)
b1 = B_peredge(pp, RR, ed, dd, lambda a, b: dv['U'], lambda a, b: dv['V'])
b2 = B_pernode(pp, RR, ed, dd)
print(f"  R3S1 with v=(1,1,1)/sqrt3: max|B^x - B^.| = {np.abs(b1-b2).max():.3f} "
      f"(differ), ranks {rank(b1)} vs {rank(b2)}, "
      f"null cols {sum(np.linalg.norm(b1[:,i])<1e-12 for i in range(24))} vs "
      f"{sum(np.linalg.norm(b2[:,i])<1e-12 for i in range(24))}")

print()
print("-"*78)
print("E. Randomised study: heterogeneous formations, random subgraphs")
print("-"*78)
rng = np.random.default_rng(12345)
N = 1200
bad_id_x = bad_id_n = used = 0
disagree_x = disagree_n = 0
nullcol_x = nullcol_n = 0
for trial in range(N):
    nn = int(rng.integers(4, 7))
    kinds = [('R2S1' if rng.random() < 0.6 else 'SE3') for _ in range(nn)]
    if all(k == kinds[0] for k in kinds):          # force heterogeneity
        kinds[0] = 'SE3' if kinds[0] == 'R2S1' else 'R2S1'
    dd = [domain(k) for k in kinds]
    pl = [i for i, k in enumerate(kinds) if k == 'R2S1']
    pp = []
    for i, k in enumerate(kinds):
        q = rng.normal(size=3)
        pp.append(np.array([q[0], q[1], 0.]) if k == 'R2S1' else q)
    RR = [Rz(rng.uniform(0, 6.28)) if k == 'R2S1' else expSO3(rng.normal(size=3))
          for k in kinds]
    Kall = complete_edges(nn)
    ed = [e for e in Kall if rng.random() < 0.45]
    if len(ed) < 2:
        continue
    used += 1
    Uf = lambda i, j: (np.diag([1., 1., 0.]) if (i in pl and j in pl) else I3)
    Vf = lambda i, j: (np.outer(E3, E3) if i in pl else I3)
    cc = sum(d['c'] for d in dd); qvv = 6*nn - cc
    Ax = admissible_basis(dd)
    bxG = B_peredge(pp, RR, ed, dd, Uf, Vf); bxK = B_peredge(pp, RR, Kall, dd, Uf, Vf)
    bnG = B_pernode(pp, RR, ed, dd);         bnK = B_pernode(pp, RR, Kall, dd)
    qiT = Ax.shape[1] - rank(bnG @ Ax)       # ground truth (== FD Jacobian)
    qtT = Ax.shape[1] - rank(bnK @ Ax)
    true_ibr = (qiT == qtT)
    # identity (14) with structural q_v
    if rank(bxG) != 6*nn - qvv - (Ax.shape[1] - rank(bxG @ Ax)):
        bad_id_x += 1
    if rank(bnG) != 6*nn - qvv - (Ax.shape[1] - rank(bnG @ Ax)):
        bad_id_n += 1
    if (rank(bxG) == rank(bxK)) != true_ibr:
        disagree_x += 1
    if (rank(bnG) == rank(bnK)) != true_ibr:
        disagree_n += 1
    nullcol_x += (sum(np.linalg.norm(bxK[:, i]) < 1e-12 for i in range(6*nn)) == qvv)
    nullcol_n += (sum(np.linalg.norm(bnK[:, i]) < 1e-12 for i in range(6*nn)) == qvv)
print(f"  trials generated = {N}, usable (>=2 edges) = {used}")
print(f"  identity rk = 6n-q_v-q_i violated : per-edge {bad_id_x:4d}/{used}   "
      f"per-node {bad_id_n:4d}/{used}")
print(f"  rank criterion disagrees with truth: per-edge {disagree_x:4d}/{used} "
      f"({100*disagree_x/used:.1f}%)   per-node {disagree_n:4d}/{used}")
print(f"  null columns of B_K equal q_v      : per-edge {nullcol_x:4d}/{used}   "
      f"per-node {nullcol_n:4d}/{used}")

print()
print("-"*78)
print("F. Remark 5: graph-dependence of 'number of null columns' (homogeneous SE(3))")
print("-"*78)
nn = 4
dd = [domain('SE3')]*nn
pp = [rng.normal(size=3) for _ in range(nn)]
RR = [expSO3(rng.normal(size=3)) for _ in range(nn)]
ed = [(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0), (3, 1), (3, 2), (2, 1)]  # agent 0 measures nothing
bG = B_pernode(pp, RR, ed, dd); bK = B_pernode(pp, RR, complete_edges(nn), dd)
print(f"  agent 1 has no outgoing edge: null columns of B_G = "
      f"{sum(np.linalg.norm(bG[:,i])<1e-12 for i in range(24))}, of B_K = "
      f"{sum(np.linalg.norm(bK[:,i])<1e-12 for i in range(24))}; structural q_v = 0")
Ax = admissible_basis(dd)
print(f"  rk B_G = {rank(bG)}, rk B_K = {rank(bK)}  -> criterion says "
      f"{'IBR' if rank(bG)==rank(bK) else 'NOT IBR'} (correct: NOT IBR, "
      f"rotating agent 1 is a non-trivial infinitesimal variation)")
print(f"  homogeneous absolute test of Thm 3 (rk = cn-c-1 = {6*nn-7}): rk B_G = {rank(bG)}")


print("="*78)
print("G. (R2) as an operator statement:  B (I - A A^T) = 0 ?")
print("="*78)
p = [np.array([0.,0.,0.]), np.array([2.,0.,0.]), np.array([1.,1.7,0.]), np.array([0.9,0.6,1.4])]
R = [Rz(0.3), Rz(1.1), Rz(2.2), Rz(0.7)@Rx(0.4)]
doms = [domain('R2S1')]*3 + [domain('SE3')]
K = complete_edges(4); planar=[0,1,2]
Utab = lambda i,j: (np.diag([1.,1.,0.]) if (i in planar and j in planar) else I3)
Vtab = lambda i,j: (np.outer(E3,E3) if i in planar else I3)
A = admissible_basis(doms); Pi_perp = np.eye(24) - A@A.T
Bx = B_peredge(p,R,K,doms,Utab,Vtab); Bn = B_pernode(p,R,K,doms)
print(f"  per-edge  ||B(I-AA^T)|| = {np.linalg.norm(Bx@Pi_perp):.4f}   (R2 FAILS)")
print(f"  per-node  ||B(I-AA^T)|| = {np.linalg.norm(Bn@Pi_perp):.2e}   (R2 holds)")

# R^3 x S^1 with v not a coordinate axis: null-column count vs the operator statement
v = np.array([1.,1.,1.])/np.sqrt(3); dv = domain('R3S1', v=v)
pp = [np.array([0.,0.,0.]), np.array([1.3,.2,-.4]), np.array([.1,1.1,.7]), np.array([-.6,.4,1.2])]
RR = [expSO3(v*t) for t in (0.2,1.0,2.1,0.6)]
dd=[dv]*4; Av = admissible_basis(dd); Pv = np.eye(24)-Av@Av.T
b1 = B_peredge(pp,RR,K,dd, lambda a,b: dv['U'], lambda a,b: dv['V'])
b2 = B_pernode(pp,RR,K,dd)
print(f"\n  R^3xS^1, v=(1,1,1)/sqrt(3)  [Remark 14 caveat]")
print(f"    Table-I form  : null cols = {sum(np.linalg.norm(b1[:,i])<1e-12 for i in range(24))},"
      f" rank = {rank(b1)}, ||B(I-AA^T)|| = {np.linalg.norm(b1@Pv):.3f}")
print(f"    projector form: null cols = {sum(np.linalg.norm(b2[:,i])<1e-12 for i in range(24))},"
      f" rank = {rank(b2)}, ||B(I-AA^T)|| = {np.linalg.norm(b2@Pv):.2e}")
print("    -> same rank & same q_v, but 'null column' is coordinate-dependent;")
print("       only the operator form of (R2) is invariant.")

print()
print("="*78)
print("H. Theorem 6 when rk(S_i - S_j) = 2: obstruction cannot be met at ANY configuration")
print("="*78)
Sxy = np.diag([1.,1.,0.])   # agent confined to the xy-plane
Sxz = np.diag([1.,0.,1.])   # agent confined to the xz-plane
D = Sxy - Sxz
print(f"  S_i - S_j = diag{np.diag(D)}, rank = {rank(D)} > 1")
worst = np.inf
for _ in range(20000):
    ph = np.random.normal(size=3); ph/=np.linalg.norm(ph)
    worst = min(worst, np.linalg.norm(Pperp(ph)@D))
print(f"  min over 20000 random directions of ||P(phat)(S_i-S_j)|| = {worst:.4f}  (never 0)")

print()

print()
print("="*78)
print("I. Reproduction of the numbers reported in Section VI-B of [1]")
print("="*78)
print("  [1] reports for the heterogeneous case study on the complete graph:")
print("      rk(B_K^+) = 13, six null columns, null(B_K^+) = 11, q_v = 6, q_t = 5.")
print(f"  per-edge form (Table III): rk = {rank(Bx)}, null cols = "
      f"{sum(np.linalg.norm(Bx[:,i])<1e-12 for i in range(24))}, nullity = {24-rank(Bx)}"
      "   <-- reproduces [1] exactly")
print(f"  admissible truth        : c = 15, q_t = dim ker(B|A) = "
      f"{15-rank(Bn@A)}, rk(B|A) = {rank(Bn@A)}")
print("  -> [1]'s q_t = 5 counts the global z-translation, which is NOT in Ibar")
print("     (a unicycle cannot leave its plane), contradicting Definition 11 of [1].")
