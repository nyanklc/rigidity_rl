from util import *
import numpy as np
import copy

from cost import counted


# (S_i, P_i): the translational and rotational coordinates agent i can vary.
# Per node, not per edge.
def node_dof_projectors(agent):
    domain = agent.domain

    S = np.eye(3)
    if domain in ("R^2", "R^2xS^1"):
        S = np.diag([1.0, 1.0, 0.0])

    P = np.zeros((3, 3))
    if domain in ("SE(3)", "R^3xSO(3)", "", None):
        P = np.eye(3)
    elif domain == "R^3xS^1":
        # v v^T, not a row placement: the two agree only at v = e3
        v = agent.rotation_axis
        v = np.array([0.0, 0.0, 1.0]) if v is None else np.asarray(v, dtype=float)
        v = v / np.linalg.norm(v)
        P = np.outer(v, v)
    elif domain == "R^2xS^1":
        e3 = np.array([0.0, 0.0, 1.0])
        P = np.outer(e3, e3)

    return S, P


# Michieletto Table I / III. Superseded by node_dof_projectors and unused by the
# matrix; kept as the reference the homogeneous-equivalence test compares against.
def bearing_DOFs(agent_i, agent_j):
    domain_i = agent_i.domain
    domain_j = agent_j.domain

    Uij = np.zeros((3, 3))
    Vij = np.zeros((3, 3))

    zero = np.zeros(3)
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    if domain_i == "SE(3)" or domain_i == "R^3xSO(3)" or domain_i == "" or domain_i == None:
        Uij = np.eye(3)
        Vij = np.eye(3)
    elif domain_i == "R^3xS^1":
        rax = None
        if agent_i.rotation_axis is not None:
            rax = agent_i.rotation_axis
        Vij = np.asarray([zero, zero, rax])
        Uij = np.eye(3)
    elif domain_i == "R^2xS^1":
        Vij = np.asarray([zero, zero, e3])
        if domain_j in ["SE(3)", "R^3xS^1", "R^3"]:
            Uij = np.eye(3)
        elif domain_j in ["R^2xS^1", "R^2"]:
            Uij = np.array([e1, e2, zero])
    elif domain_i == "R^3":
        Uij = np.eye(3)
        Vij = np.zeros((3, 3))
    elif domain_i == "R^2":
        Vij = np.zeros((3, 3))
        if domain_j in ["SE(3)", "R^3xS^1", "R^3"]:
            Uij = np.eye(3)
        elif domain_j in ["R^2xS^1", "R^2"]:
            Uij = np.array([e1, e2, zero])

    return Uij, Vij

# # OLD
# def extended_bearing_rigidity_matrix(network):
#     p = [agent.pose.position for agent in network.agents]
#     R = [agent.pose.rotation_mat() for agent in network.agents]
#     edges = network.edges

#     n = len(p)
#     m = int(edges.sum())

#     E = np.zeros((n, m))
#     Eo = np.zeros((n, m))
#     U = np.zeros((3*m, 3*m))
#     V = np.zeros((3*m, 3*m))

#     i_indices, j_indices = np.nonzero(edges)
#     # TODO: there should be a more efficient implementation using the adjacency mat, i was lazy
#     for k, (i, j) in enumerate(zip(i_indices, j_indices)):
#         E[i, k] = -1
#         E[j, k] = +1
#         Eo[i, k] = -1
#         # Uij, Vij
#         U[3*k:3*(k+1), 3*k:3*(k+1)], V[3*k:3*(k+1), 3*k:3*(k+1)] = bearing_DOFs(
#             network.agents[i], network.agents[j]
#             )

#     E_bar = np.kron(E, np.eye(3))
#     Eo_bar = np.kron(Eo, np.eye(3))

#     Dp = np.zeros((3*m, 3*m))
#     Da = np.zeros((3*m, 3*m))
#     for k, (i, j) in enumerate(zip(i_indices, j_indices)):

#         # TODO: not sure if we should do this
#         if i == j:
#             continue

#         pij = p[j] - p[i]
#         s = 1.0 / np.linalg.norm(pij)
#         p_bar = s * pij

#         Ri = R[i]

#         P = orthogonal_projection_matrix(p_bar)

#         Dp_k = s * Ri.T @ P
#         Da_k = -Ri.T @ skew_symmetric(p_bar)

#         Dp[3*k:3*(k+1), 3*k:3*(k+1)] = Dp_k

#         Da[3*k:3*(k+1), 3*k:3*(k+1)] = Da_k

#     Bp = Dp @ U @ E_bar.T
#     Ba = Da @ V @ Eo_bar.T
#     B = np.hstack([Bp, Ba]) # (3m, 6n)

#     return B

# B = [ Dp E_bar^T S_bar | Da Eo_bar^T P_bar ], (3m, 6n). The DOF restriction is
# applied per node on the column side, so an infeasible coordinate is a zero
# column.
@counted
def extended_bearing_rigidity_matrix(network):
    p = [agent.pose.position for agent in network.agents]
    R = [agent.pose.rotation_mat() for agent in network.agents]
    edges = network.edges

    n = len(p)
    m = int(edges.sum())

    E = np.zeros((n, m))
    Eo = np.zeros((n, m))

    i_indices, j_indices = np.nonzero(edges)
    for k, (i, j) in enumerate(zip(i_indices, j_indices)):
        E[i, k] = -1
        E[j, k] = +1
        Eo[i, k] = -1   # only the measurer's attitude enters its own bearing

    E_bar = np.kron(E, np.eye(3))
    Eo_bar = np.kron(Eo, np.eye(3))

    S_bar = np.zeros((3*n, 3*n))
    P_bar = np.zeros((3*n, 3*n))
    for i, agent in enumerate(network.agents):
        S_bar[3*i:3*(i+1), 3*i:3*(i+1)], P_bar[3*i:3*(i+1), 3*i:3*(i+1)] = (
            node_dof_projectors(agent)
        )

    Dp = np.zeros((3*m, 3*m))
    Da = np.zeros((3*m, 3*m))
    for k, (i, j) in enumerate(zip(i_indices, j_indices)):
        if i == j:   # no self bearings; keeps fully_connected() honest
            continue

        pij = p[j] - p[i]
        s = 1.0 / np.linalg.norm(pij)
        p_bar = s * pij

        Ri = R[i]

        Proj = orthogonal_projection_matrix(p_bar)

        Dp[3*k:3*(k+1), 3*k:3*(k+1)] = s * Ri.T @ Proj
        Da[3*k:3*(k+1), 3*k:3*(k+1)] = -Ri.T @ skew_symmetric(p_bar)

    Bp = Dp @ E_bar.T @ S_bar
    Ba = Da @ Eo_bar.T @ P_bar
    B = np.hstack([Bp, Ba]) # (3m, 6n)

    return B

@counted
def is_IBR_explicit(brmat, rank_K=None):
    if rank_K is None:
        raise Exception("HEY WHAT")
    rank = np.linalg.matrix_rank(brmat)
    return rank == rank_K, rank

def is_IBR(network, brmat=None, rank_K=None):
    if int(network.edges.sum()) == 0:
        return False

    # rigidity matrix
    if brmat is None:
        brmat = extended_bearing_rigidity_matrix(network)

    if rank_K is None:
        # rigidity matrix of the fully connected graph
        network_K = network.fully_connected()
        brmat_K = extended_bearing_rigidity_matrix(network_K)
        rank_K = np.linalg.matrix_rank(brmat_K)

    # print(f"IBR check: {np.linalg.matrix_rank(brmat)} =? {rank_K}")
    return is_IBR_explicit(brmat, rank_K=rank_K)

def rigidity_eigenvalue(network, eps=1e-10, rank_K=None):
    eigs = network.eigenvalues()

    if rank_K is None:
        network_K = network.fully_connected()
        # TODO: we can/should get this from IBR check during training
        brmat_K = extended_bearing_rigidity_matrix(network_K)
        rank_K = np.linalg.matrix_rank(brmat_K)

    n = len(network.agents)
    zero_count = 6*n - rank_K

    # print(f"hello BRM: {network.extended_bearing_rigidity_matrix().shape}")
    # print(f"hello edges: {np.sum(network.edges)} IBR: {network.is_IBR()} eigs: {eigs}")
    # print(f"hello returning: {eigs[zero_count]} from index: {zero_count}")
    return eigs[zero_count]

# M. H. Trinh, Q. Van Tran, and H.-S. Ahn, “Minimal and Redundant Bearing Rigidity: Conditions and Applications,” IEEE Transactions on Automatic Control, vol. 65, no. 10, pp. 4186-4200, Oct. 2020, doi: 10.1109/TAC.2019.2958563.
# NOTE: ONLY FOR R^d
def is_MBR_Rd(network, rank_K=None, brmat=None):
    if brmat:
        isIBR, rank = is_IBR_explicit()
    isIBR, rank = is_IBR(network, rank_K=rank_K)

    if len(network.agents) == 0:
        return False, isIBR

    if not isIBR:
        return False, isIBR

    n = len(network.agents)
    d = 2 if network.agents[0].domain in ["R^2", "R^2xS^1"] else 3
    m = int(network.edges.sum())

    if d < 2 or n < 3:
        return False, isIBR

    # cycle graph
    if 3 <= n <= d + 1:
        return m == n

    k = (n - 2) // (d - 1)
    r = (n - 2) % (d - 1)
    sgn = 1 if r > 0 else 0

    m_required = 1 + k * d + r + sgn

    return m == m_required, isIBR

def MBR_required_Rd(n ,d):
    k = (n - 2) // (d - 1)
    r = (n - 2) % (d - 1)
    sgn = 1 if r > 0 else 0

    m_required = 1 + k * d + r + sgn

    return m_required

# Most rank one edge could contribute at these poses. EXACT -- makes no claim
# about what is jointly achievable, which is why the state score normalizes with
# it rather than with an edge count.
@counted
def max_edge_rank(network, brmat_K=None):
    n = len(network.agents)
    if n < 2:
        return 1

    domains = {agent.domain for agent in network.agents}
    if len(domains) == 1:
        domain = next(iter(domains))
        if domain in ["R^2", "R^3"]:
            return 2 if domain == "R^3" else 1

    if brmat_K is None:
        brmat_K = extended_bearing_rigidity_matrix(network.fully_connected())

    m_K = brmat_K.shape[0] // 3
    c_max = max(
        (np.linalg.matrix_rank(brmat_K[3*k:3*(k+1), :]) for k in range(m_K)),
        default=1,
    )
    return max(int(c_max), 1)

# Fewest edges that could make these poses rigid.
# LOWER BOUND, not a ground truth: keep it out of the reward, use it for
# reporting and the MBR metric only. Costs n(n-1) rank computations -- cache it.
@counted
def required_edge_count(network, rank_K=None, brmat_K=None, block_ranks=None):
    n = len(network.agents)
    if n < 2:
        return 0

    domains = {agent.domain for agent in network.agents}
    if len(domains) == 1:
        domain = next(iter(domains))
        if domain in ["R^2", "R^3"]:
            return MBR_required_Rd(n, 2 if domain == "R^2" else 3)

    if brmat_K is None or rank_K is None:
        network_K = network.fully_connected()
        brmat_K = extended_bearing_rigidity_matrix(network_K)
        if rank_K is None:
            rank_K = np.linalg.matrix_rank(brmat_K)

    if block_ranks is None:
        block_ranks = edge_block_ranks(brmat_K)
    block_ranks = sorted(block_ranks, reverse=True)

    sum_c = 0
    m_req = 0
    for c in block_ranks:
        if c == 0:  # a zero block constrains nothing; the rest are zero too
            break
        sum_c += c
        m_req += 1
        if sum_c >= rank_K:
            break

    return max(m_req, 1)

# # idk if this is reliable
# def is_MBR_general(network, rank_K=None):
#     raise Exception("MBR (general) doesn't quite work i think. Abort.")

#     isIBR = is_IBR(network, rank_K=rank_K)

#     if len(network.agents) < 2 or not isIBR:
#         return False, isIBR

#     edges_list = network.get_edge_list()
#     for edge in edges_list:
#         network.remove_edge(*edge)
#         still_rigid = is_IBR(network, rank_K=rank_K)
#         network.add_edge(*edge)
#         if still_rigid:
#             return False, True

#     return True, True

# Rank each edge's own 3-row block. Constant (d-1) in homogeneous R^d, so it only
# carries information on heterogeneous networks.
# Fewest edges that could make a BROKEN graph rigid again, which is the question
# after an agent leaves or a link fails. required_edge_count cannot answer it: it
# starts from the empty graph and ignores what the survivors still have.
# Karimian and Tron Theorem 4 is the c_max = 1 case of this, exact in homogeneous
# R^2; the 3-D and heterogeneous extension is what makes this a bound instead.
@counted
def repair_edge_count(network, rank_K=None, brmat=None, rank_brm=None,
                      length_scale=None):
    """Lower bound on the edges needed to restore rigidity. 0 if already rigid.

        deficit = rank_K - rank(B)
        m_rep   = smallest k with sum of the k largest marginals >= deficit   (19.2)

    The marginals are the exact per-pair gains rank(b_ij Z) over the absent
    pairs, not the complete graph's block ranks: an edge's block may have rank 2
    while contributing 1 here, and using the block rank would undercount.
    """
    n = network.n
    if n < 2:
        return 0

    if brmat is None:
        brmat = extended_bearing_rigidity_matrix(network)
    if rank_K is None:
        rank_K = int(np.linalg.matrix_rank(
            extended_bearing_rigidity_matrix(network.fully_connected())))
    if rank_brm is None:
        rank_brm = int(np.linalg.matrix_rank(brmat)) if brmat.size else 0

    deficit = int(rank_K) - int(rank_brm)
    if deficit <= 0:
        return 0

    if length_scale is None:
        length_scale = characteristic_length(network)
    Z = nullspace_in_scaled_units(nullspace(brmat, int(rank_brm)), n, length_scale)
    gains = candidate_gain(network, Z, length_scale=length_scale)[1]

    absent = np.logical_and(~network.edges.astype(bool), ~np.eye(n, dtype=bool))
    available = sorted((g for g in gains[absent].tolist() if g > 0), reverse=True)

    total = 0
    for k, g in enumerate(available, start=1):
        total += g
        if total >= deficit:
            return k

    # only a rank threshold can land here; report the whole budget rather than lie
    return len(available)


@counted
def edge_block_ranks(brmat):
    return [np.linalg.matrix_rank(brmat[3*k:3*(k+1), :]) for k in range(brmat.shape[0] // 3)]


# Per-node flex tensor: the (n, 3, 3) diagonal blocks of the projector onto the
# framework's non-trivial infinitesimal flex space -- the directions it cannot
# resist (under-constrained) or resists least (rigid).
#
# Returns the projector block rather than a single eigenvector on purpose. The
# flex space is usually multi-dimensional, and any individual eigenvector inside
# a degenerate eigenspace is an arbitrary basis choice, so a per-mode feature is
# not reproducible. G_i = sum_c v_i^(c) v_i^(c)^T is basis-independent, and it
# transforms as a tensor, so scalars read off it are rotation-invariant.
def trivial_modes(positions):
    """Motions every bearing framework admits: translation and uniform scaling.

    These span part of B's null space, and eigh returns an *arbitrary* basis of
    that whole space, so they have to be projected out explicitly -- taking "the
    columns after the first few" mixes them in and gives a basis-dependent,
    rotation-dependent answer.
    """
    n = len(positions)
    T = np.zeros((3*n, 4))
    for k in range(3):
        T[k::3, k] = 1.0                                   # translations
    T[:, 3] = (positions - positions.mean(axis=0)).reshape(-1)  # uniform scaling
    q, _ = np.linalg.qr(T)
    return q


def flex_tensor(brmat, n, positions, tol=1e-9):
    Bp = brmat[:, :3*n]
    if Bp.size == 0:
        return np.zeros((n, n, 3, 3))

    w, V = np.linalg.eigh(Bp.T @ Bp)
    scale = max(w.max(), 1.0)
    Z = V[:, w <= scale * tol]                 # the whole null space

    T = trivial_modes(positions)
    if Z.shape[1]:
        Z = Z - T @ (T.T @ Z)                  # strip translation and scaling
        u, s, _ = np.linalg.svd(Z, full_matrices=False)
        Z = u[:, s > 1e-7]

    if Z.shape[1] == 0:
        # rigid: no flex left, so use the weakest resisted direction instead
        keep = w > scale * tol
        if not keep.any():
            return np.zeros((n, n, 3, 3))
        Z = V[:, np.argmax(keep)][:, None]

    # full projector, as (n, n, 3, 3) blocks: Pi[i, j] = sum_c v_ci v_cj^T.
    # The cross blocks are needed because a bearing constrains the *relative*
    # motion of i and j, not i alone.
    blocks = Z.reshape(n, 3, -1)               # (n, 3, k)
    return np.einsum("idk,jek->ijde", blocks, blocks)


def flex_constraint_power(Pi, bearings):
    """For every ordered pair, how much of the current flex the edge would remove.

    A bearing constrains P(p_hat_ij) (v_j - v_i) = 0, i.e. the components of the
    relative flex *perpendicular* to the bearing; the parallel component is the
    scale freedom it cannot see. So the useful quantity is

        A[i,j]^2 = sum_c || P(p_hat_ij) (v_cj - v_ci) ||^2
                 = sum_c ||D_c||^2  -  sum_c (p_hat_ij . D_c)^2 ,   D_c = v_cj - v_ci

    both terms of which come straight out of the projector blocks. Basis-
    independent and rotation-invariant.
    """
    n = Pi.shape[0]
    Gd = np.einsum("iidd->id", Pi)                     # trace(G_i) per axis
    tr = Gd.sum(axis=1)                                # trace(G_i)
    cross = np.einsum("ijdd->ij", Pi)                  # trace(Pi[i,j])
    sq_norm = tr[:, None] + tr[None, :] - 2.0 * cross  # sum_c ||D_c||^2

    # M[i,j] = G_i + G_j - Pi[i,j] - Pi[j,i]  is the quadratic form of D_c
    M = Pi[np.arange(n), np.arange(n)][:, None] + Pi[np.arange(n), np.arange(n)][None, :]
    M = M - Pi - np.swapaxes(Pi, 0, 1)
    parallel = np.einsum("ijd,ijde,ije->ij", bearings, M, bearings, optimize=True)

    return np.sqrt(np.maximum(sq_norm - parallel, 0.0))


# Rank and margin from one thin SVD. The null space costs extra and is only
# needed by the rigidity features, so it is a separate call.
@counted
def rigidity_decomposition(brmat, rank_K):
    """(rank, singular values, lam). lam is 0 unless the framework is rigid.

    rank uses numpy's matrix_rank tolerance, so the IBR verdict is unchanged.
    """
    if brmat.size == 0:
        return 0, np.zeros(0), 0.0
    s = np.linalg.svd(brmat, compute_uv=False)
    tol = s.max() * max(brmat.shape) * np.finfo(s.dtype).eps
    rank = int((s > tol).sum())
    lam = float(s[rank_K - 1] ** 2) if rank >= rank_K and rank_K - 1 < len(s) else 0.0
    return rank, s, lam


def estimation_error(s, rank_K, rank=None):
    """(a_opt, e_opt, d_opt) from B's singular values, descending.        (18.1)

    tr((B^T B)^+), 1/lam and -sum log w over the rank_K nonzero eigenvalues
    w_k = s_k^2. inf for all three on a flexible framework, where the shape is
    not identifiable.
    """
    if rank is None:
        rank = int(len(s))
    if rank_K < 1 or len(s) < rank_K or rank < rank_K:
        return np.inf, np.inf, np.inf

    w = np.asarray(s[:rank_K], dtype=float) ** 2
    if w[-1] <= 0.0:
        return np.inf, np.inf, np.inf

    return float((1.0 / w).sum()), float(1.0 / w[-1]), float(-np.log(w).sum())


# B's position columns carry 1/length while its attitude columns do not, so a
# spectrum read off the raw matrix mixes units.
def scaled_rigidity_matrix(network, brmat=None, length_scale=None):
    """B with its position columns in units of the formation's RMS radius."""
    if brmat is None:
        brmat = extended_bearing_rigidity_matrix(network)
    if brmat.size == 0:
        return brmat
    if length_scale is None:
        length_scale = characteristic_length(network)
    out = brmat.copy()
    out[:, :3 * network.n] *= length_scale
    return out


@counted
def estimation_error_blocks(brmat, rank_K, n):
    """(a_pos, a_att): tr((B^T B)^+) over the position and attitude blocks.

    What the position and attitude RMS errors are separately predicted by;
    a_opt alone predicts a quantity mixing lengths and radians. The cut is by
    index rather than tolerance, so squaring B costs no accuracy in the rank.
    """
    if brmat.size == 0:
        return np.inf, np.inf
    cols = brmat.shape[1]
    k = cols - int(rank_K)
    if k < 0:
        return np.inf, np.inf

    w, V = np.linalg.eigh(brmat.T @ brmat)     # ascending
    w, V = w[k:], V[:, k:]                     # drop ker(B)
    if w.min() <= 0:
        return np.inf, np.inf

    inv = 1.0 / w
    return (float((inv * (V[:3 * n] ** 2).sum(axis=0)).sum()),
            float((inv * (V[3 * n:] ** 2).sum(axis=0)).sum()))


@counted
def error_covariance(brmat, rank_K):
    """(B^T B)^+, the shape-error covariance per unit bearing-noise variance.

    (6n, 6n). Its per-agent 3x3 position blocks are what an uncertainty ellipse
    is drawn from.
    """
    cols = brmat.shape[1]
    k = cols - int(rank_K)
    if brmat.size == 0 or k < 0:
        return np.full((cols, cols), np.inf)
    w, V = np.linalg.eigh(brmat.T @ brmat)
    w, V = w[k:], V[:, k:]
    if w.min() <= 0:
        return np.full((cols, cols), np.inf)
    return (V * (1.0 / w)) @ V.T


# Noise on one measurement propagates through B^+, so that measurement's share of
# the total squared shape error is ||B^+ restricted to its columns||_F^2. The shares
# sum to tr((B^T B)^+) exactly, which is what makes them readable as fractions.
def measurement_sensitivity(network, rank_K, brmat=None, length_scale=None):
    """(per_edge, per_node): each measurement's share of the total shape error.

    per_edge is in np.nonzero(edges) order; per_node groups by the agent that
    takes the measurement, i.e. over its outgoing edges. Both sum to a_opt.
    """
    n = network.n
    Bs = scaled_rigidity_matrix(network, brmat, length_scale)
    m = Bs.shape[0] // 3
    if m == 0:
        return np.zeros(0), np.zeros(n)

    M = error_covariance(Bs, rank_K)
    if not np.isfinite(M).all():
        return np.full(m, np.inf), np.full(n, np.inf)

    per_edge = np.array([np.linalg.norm(M @ Bs[3 * k:3 * k + 3, :].T) ** 2
                         for k in range(m)])

    per_node = np.zeros(n)
    for k, i in enumerate(np.nonzero(network.edges)[0]):
        per_node[i] += per_edge[k]
    return per_edge, per_node


def estimation_error_of(network, rank_K, brmat=None, length_scale=None):
    """estimation_error on the length-normalised B. One SVD."""
    Bs = scaled_rigidity_matrix(network, brmat, length_scale)
    if Bs.size == 0:
        return np.inf, np.inf, np.inf
    rank, s, _ = rigidity_decomposition(Bs, rank_K)
    return estimation_error(s, rank_K, rank=rank)


@counted
def greedy_rigid_construction(network, rank_K, rng):
    """From the empty graph, keep any edge that raises rank(B). (edges, added, rank).

    `rng` must be private to the caller -- the global stream is the one instances
    are drawn from.
    """
    n = network.n
    order = [(i, j) for i in range(n) for j in range(n) if i != j]
    E = np.zeros((n, n), dtype=bool)
    added, rank, progress = [], 0, True

    while rank < rank_K and progress:
        progress = False
        for k in rng.permutation(len(order)):
            i, j = order[k]
            if E[i, j]:
                continue
            E[i, j] = True
            network.edges = E
            new_rank = np.linalg.matrix_rank(extended_bearing_rigidity_matrix(network))
            if new_rank > rank:
                rank, progress = new_rank, True
                added.append((i, j))
            else:
                E[i, j] = False

    network.edges = E
    return E, added, rank


# Repair by marginal gain: add the absent pair that raises rank(B) most, until
# rigid. In the c_max = 1 domains the independent sets are a matroid, so this is
# minimum-edge there by the same argument that makes greedy optimal in 1.4.
@counted
def greedy_rigid_repair(network, rank_K, rng=None, brmat=None, length_scale=None):
    """Restore rigidity by adding edges. (edges, added), mutating network.edges.

    Unlike greedy_rigid_construction this keeps the edges already present, which
    is the difference between rebuilding a formation and repairing one.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = network.n
    if length_scale is None:
        length_scale = characteristic_length(network)

    added = []
    brm = extended_bearing_rigidity_matrix(network) if brmat is None else brmat
    rank = int(np.linalg.matrix_rank(brm)) if brm.size else 0

    while rank < rank_K:
        Z = nullspace_in_scaled_units(nullspace(brm, rank), n, length_scale)
        gains = candidate_gain(network, Z, length_scale=length_scale)[1]
        gains[network.edges.astype(bool)] = -1
        np.fill_diagonal(gains, -1)
        if gains.max() <= 0:
            break                                  # nothing left that would help

        best = np.argwhere(gains == gains.max())
        i, j = best[rng.integers(len(best))]
        network.edges[i, j] = True
        added.append((int(i), int(j)))

        brm = extended_bearing_rigidity_matrix(network)
        rank = int(np.linalg.matrix_rank(brm))

    return network.edges, added


# Higher is better for all three, so a state score can use any of them the same
# way. Widths are the eigenvalue's 0.75 decades scaled by each functional's
# measured p10-p90 spread (tools/spectral_criteria.py); logdet is in nats.
SPECTRAL_FUNCTIONALS = ("eigenvalue", "trace", "logdet")
SPECTRAL_SIGMOID_WIDTH = {"eigenvalue": 0.75, "trace": 0.60, "logdet": 6.7}

# A fixed centre for the conditioning term, so it needs no per-episode reference.
# lambda's centre moves 2.5 decades over n = 8..20 and the five domains, which is why
# it needed one; shape_err's moves 1.1, because the formation's size is already divided
# out by the length normalisation and the agent count by the /n.
# The exponent would divide out what is left of n, and is 0 because it was measured not
# to buy anything: a sigmoid wide enough for the within-instance spread is already wide
# enough for that drift, so at 1.9 the median and p10 instance keep exactly the same
# gradient. Setting it near 1.9 makes phi's ceiling n-invariant, worth about 1% of phi
# at kappa = 2. tools/shape_error_scale.py re-derives all three and prices the exponent.
SHAPE_ERR_EXPONENT = 0.0
SHAPE_ERR_CENTRE = 1.35
SHAPE_ERR_SIGMOID_DECADES = 0.70


def shape_error_quality(shape_err, n):
    """q in (0, 1): how well conditioned this graph is, on a scale fixed in advance.

    Lower shape_err is better, so q rises as it falls, and q = 0.5 at the centre --
    a graph as good as a typical minimal one. None when the shape is not
    identifiable, i.e. on a flexible framework.
    """
    if shape_err is None or not np.isfinite(shape_err) or shape_err <= 0:
        return None
    g = np.log10(shape_err) - SHAPE_ERR_CENTRE
    if SHAPE_ERR_EXPONENT:
        g -= SHAPE_ERR_EXPONENT * np.log10(max(int(n), 2))
    return float(1.0 / (1.0 + np.exp(g / SHAPE_ERR_SIGMOID_DECADES)))


def spectral_value(functional, lam, a_opt=None, d_opt=None):
    """The scalar a spectral state score reads. None when it is undefined."""
    if functional == "eigenvalue":
        return float(np.log10(lam)) if lam and lam > 0 else None
    if functional == "trace":
        return -float(np.log10(a_opt)) if a_opt and np.isfinite(a_opt) else None
    if functional == "logdet":
        return -float(d_opt) if d_opt is not None and np.isfinite(d_opt) else None
    raise ValueError(f"unknown spectral functional {functional!r}")


def reference_spectral(network, rank_K, rng, samples=3, functional="eigenvalue"):
    """Median spectral_value over `samples` greedy graphs on these poses.

    None if none reached rank_K. At functional="eigenvalue" this is
    log10(reference_stiffness), so the two agree by construction.
    """
    vals = []
    for _ in range(max(1, int(samples))):
        work = copy.deepcopy(network)
        _, _, rank = greedy_rigid_construction(work, rank_K, rng)
        if rank < rank_K:
            continue
        brm = extended_bearing_rigidity_matrix(work)
        lam = rigidity_decomposition(brm, rank_K)[2]
        a_opt, _, d_opt = estimation_error_of(work, rank_K, brmat=brm)
        v = spectral_value(functional, lam, a_opt, d_opt)
        if v is not None:
            vals.append(v)
    return float(np.median(vals)) if vals else None


def reference_stiffness(network, rank_K, rng, samples=3):
    """stiffness_ref: log-median lambda over `samples` greedy graphs on these poses.

    0 if none reached rank_K.
    """
    lams = []
    for _ in range(max(1, int(samples))):
        work = copy.deepcopy(network)
        _, _, rank = greedy_rigid_construction(work, rank_K, rng)
        if rank < rank_K:
            continue
        lams.append(rigidity_decomposition(
            extended_bearing_rigidity_matrix(work), rank_K)[2])
    lams = [x for x in lams if x > 0]
    return float(10.0 ** np.median(np.log10(lams))) if lams else 0.0


@counted
def nullspace(brmat, rank):
    """Orthonormal basis of ker(B), (6n, 6n - rank), given the rank.

    From eigh(B^T B), which is (6n, 6n) and so much cheaper than an SVD of B
    whose left factor is (3m, 3m) and never used. Squaring costs precision in the
    eigen*values*, but the rank is passed in rather than thresholded here, and
    the span of the smallest 6n - rank eigenvectors is what the features use.
    """
    cols = brmat.shape[1]
    if brmat.size == 0 or rank == 0:
        return np.eye(cols)
    if rank >= cols:
        return np.zeros((cols, 0))
    _, V = np.linalg.eigh(brmat.T @ brmat)          # eigenvalues ascending
    return V[:, :cols - rank]


@counted
def nullspace_and_softest(brmat, rank):
    """(ker(B), softest non-trivial mode, eigenvalues, eigenvectors) from one eigh.

    The kernel is the smallest 6n - rank eigenvectors, and v is the very next one:
    the eigenvector at the smallest NONZERO eigenvalue, which is the rigidity
    eigenvalue. v is (6n, 1), or (6n, 0) when the framework is flexible. w and V
    come back so a caller needing (B^T B)^+ does not decompose a second time.
    """
    cols = brmat.shape[1]
    if brmat.size == 0 or rank == 0:
        return np.eye(cols), np.zeros((cols, 0)), None, None
    w, V = np.linalg.eigh(brmat.T @ brmat)          # eigenvalues ascending
    if rank >= cols:
        return np.zeros((cols, 0)), np.zeros((cols, 0)), w, V
    return V[:, :cols - rank], V[:, cols - rank:cols - rank + 1], w, V


# What an existing edge costs to delete. Both are exact: the leverage block
# H = b (B^T B)^+ b^T has eigenvalues in [0, 1] and one per rank the edge alone
# carries, and dropping its rows is the downdate B^T B - b^T b.
@counted
def removal_costs(brmat, network, rank_K, lam=0.0, w=None, V=None, c_max=1,
                  need_stiffness=True):
    """(rank_lost, stiffness_lost) over all pairs, nonzero only on existing edges.

    rank_lost is in units of c_max, stiffness_lost the fraction of lambda given up,
    1 when removal breaks rigidity. Pass w, V from nullspace_and_softest.

    need_stiffness=False skips the downdate, which is one eigvalsh(6n) per redundant
    edge and the whole cost of this function; stiffness_lost then holds only the
    1 that marks a removal breaking rigidity.
    """
    n = network.n
    rank_lost = np.zeros((n, n))
    stiffness_lost = np.zeros((n, n))
    ii, jj = np.nonzero(network.edges)
    if brmat.size == 0 or len(ii) == 0:
        return rank_lost, stiffness_lost

    G = brmat.T @ brmat
    if w is None or V is None:
        w, V = np.linalg.eigh(G)
    cols = brmat.shape[1]
    tol = max(float(w.max()), 1e-30) * 1e-10
    Minv = (V * np.where(w > tol, 1.0 / np.maximum(w, 1e-300), 0.0)) @ V.T
    cm = max(int(c_max), 1)

    for k, (i, j) in enumerate(zip(ii, jj)):
        # B carries one 3-row block per directed edge, in np.nonzero(edges) order,
        # so the edge's own block is a slice rather than something to rebuild
        b = brmat[3 * k:3 * k + 3, :]
        # separation between "spanned by the others" and "uniquely carried" is
        # eight orders of magnitude, so 1e-6 sits far from either side
        c = int((np.linalg.eigvalsh(b @ Minv @ b.T) > 1.0 - 1e-6).sum())
        rank_lost[i, j] = c / cm
        if c > 0:
            stiffness_lost[i, j] = 1.0          # removal breaks rigidity
        elif lam > 0 and need_stiffness:
            w2 = np.linalg.eigvalsh(G - b.T @ b)
            stiffness_lost[i, j] = min(max(1.0 - w2[cols - rank_K] / lam, 0.0), 1.0)
    return rank_lost, stiffness_lost


def _node_projectors(network):
    n = network.n
    S = np.zeros((n, 3, 3))
    P = np.zeros((n, 3, 3))
    for i, agent in enumerate(network.agents):
        S[i], P[i] = node_dof_projectors(agent)
    return S, P


# The exact addition criterion: edge i->j raises rank(B) iff its row block has a
# component outside the row space, i.e. iff b_ij Z != 0.
def characteristic_length(network):
    """RMS radius about the centroid: the formation's own length unit."""
    p = np.array([a.pose.position for a in network.agents], dtype=float)
    p = p - p.mean(axis=0)
    return float(max(np.sqrt(np.mean((p ** 2).sum(axis=-1))), 1e-9))


# B's position columns carry units of 1/length while its attitude columns are
# dimensionless, so ker(B) moves under a uniform scaling of the formation. Fixing
# the length unit to the formation's own size makes it invariant, which is the
# same normalisation coord_features already applies.
@counted
def nullspace_in_scaled_units(Z, n, length_scale):
    if Z.shape[1] == 0:
        return Z
    W = Z.copy()
    W[:3 * n] /= length_scale
    q, _ = np.linalg.qr(W)
    return q[:, :Z.shape[1]]


def candidate_block(network, i, j, length_scale=1.0):
    """The 3 x 6n block b_ij that edge i->j would append to B.

    Built by the matrix routine itself on a network carrying only this edge, so it
    cannot drift from the construction it describes. length_scale rescales the
    position columns, matching nullspace_in_scaled_units.
    """
    single = copy.copy(network)
    single.edges = np.zeros_like(network.edges)
    single.edges[i, j] = True
    b = extended_bearing_rigidity_matrix(single)
    b[:, :3 * network.n] *= length_scale
    return b


def candidate_gain_reference(network, Z, length_scale=1.0):
    """candidate_gain written as the formula it implements. The test oracle.

    rank(B with i->j) - rank(B) = rank(b_ij Z), because the row space and the null
    space are orthogonal complements. One pair at a time, forming
    b_ij explicitly. candidate_gain fuses these steps and is ~3x faster; this is
    what tests/test_flex.py holds it to.
    """
    n = network.n
    gain = np.zeros((n, n))
    rank = np.zeros((n, n))
    if Z.shape[1] == 0:
        return gain, rank

    for i in range(n):
        for j in range(n):
            if i == j:                              # no self bearings
                continue
            b = candidate_block(network, i, j, length_scale)
            bZ = b @ Z                              # (3, dim ker B)
            norm_b = np.linalg.norm(b)
            gain[i, j] = np.linalg.norm(bZ) / max(norm_b, 1e-12)
            # threshold measured, not guessed: gains split 1.6e-10 vs 1.4e-02
            s = np.linalg.svd(bZ, compute_uv=False)
            rank[i, j] = int((s > 1e-6 * norm_b).sum())
    return gain, rank


@counted
def candidate_gain(network, Z, length_scale=1.0):
    """(gain, rank) over all ordered pairs, for the edge each pair would add.

    gain[i,j] = ||b_ij Z||_F / ||b_ij||_F in [0, 1], the fraction of the row block
    edge i->j would contribute that lies outside the current row space, and
    rank[i,j] = rank(b_ij Z), the rank it would add. gain is zero exactly on the
    pairs that would add nothing.

    Vectorized restatement of candidate_gain_reference, which is the readable form
    and the oracle the tests hold this to. b_ij is never built: expanding its three
    nonzero blocks gives

        b_ij Z = Dp (S_j Z_j - S_i Z_i) - Da P_i Z_i

    and each term becomes one batched product over all pairs.

    Normalised per pair rather than against the spread: on a rigid framework every
    raw gain is at machine zero, and dividing those by their own RMS turns noise
    into an O(1) feature. Pass length_scale with a Z from
    nullspace_in_scaled_units to make gain scale invariant.
    """
    n = network.n
    p = np.array([a.pose.position for a in network.agents], dtype=float)
    R = np.array([a.pose.rotation_mat() for a in network.agents], dtype=float)
    S, P = _node_projectors(network)
    k = Z.shape[1]
    if k == 0:
        return np.zeros((n, n)), np.zeros((n, n))

    # Z split into its position and attitude halves, per node
    Zp = Z[:3 * n].reshape(n, 3, k)
    Za = Z[3 * n:].reshape(n, 3, k)
    SZ = np.einsum("iab,ibk->iak", S, Zp)              # S_i Z_p,i
    PZ = np.einsum("iab,ibk->iak", P, Za)              # P_i Z_a,i

    d = p[None, :, :] - p[:, None, :]                  # p_j - p_i
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, 1.0)
    pb = d / dist[..., None]
    # Dp = (L / d_ij) R_i^T P(p_hat_ij),  Da = -R_i^T [p_hat_ij]_x, as in
    # extended_bearing_rigidity_matrix. "iba" transposes R_i.
    Proj = np.eye(3) - np.einsum("ija,ijb->ijab", pb, pb)
    Dp = np.einsum("ij,iba,ijbc->ijac", length_scale / dist, R, Proj)
    Sk = np.zeros((n, n, 3, 3))
    Sk[..., 0, 1], Sk[..., 0, 2] = -pb[..., 2], pb[..., 1]
    Sk[..., 1, 0], Sk[..., 1, 2] = pb[..., 2], -pb[..., 0]
    Sk[..., 2, 0], Sk[..., 2, 1] = -pb[..., 1], pb[..., 0]
    Da = -np.einsum("iba,ijbc->ijac", R, Sk)

    # b_ij Z = Dp (S_j Z_j - S_i Z_i) - Da P_i Z_i. The minus on the attitude term
    # is E_o's -1 at the measuring node; with a plus this looks plausible and is
    # wrong.
    rel = SZ[None, :, :, :] - SZ[:, None, :, :]        # S_j Z_j - S_i Z_i
    blk = np.einsum("ijab,ijbk->ijak", Dp, rel) \
        - np.einsum("ijab,ibk->ijak", Da, PZ)
    np.einsum("iiak->iak", blk)[...] = 0.0             # no self bearings

    # the block has 3 rows, so its Gram matrix is 3x3: one small eigendecomposition
    # per pair gives both the norm and the rank, where a batched SVD of (3, k)
    # would cost far more
    G = np.einsum("ijak,ijbk->ijab", blk, blk)
    gain = np.sqrt(np.maximum(np.einsum("ijaa->ij", G), 0.0))

    # ||b_ij||_F, from its three nonzero blocks, which sit in disjoint columns
    Bi = np.einsum("ijab,ibc->ijac", Dp, S)
    Bj = np.einsum("ijab,jbc->ijac", Dp, S)
    Ba = np.einsum("ijab,ibc->ijac", Da, P)
    row = np.sqrt((Bi ** 2).sum((2, 3)) + (Bj ** 2).sum((2, 3)) + (Ba ** 2).sum((2, 3)))

    # measured separation between "adds nothing" and "adds rank" is ~8 orders of
    # magnitude in gain/||b_ij||, so 1e-6 sits far from either side
    w = np.linalg.eigvalsh(G)
    ref = np.maximum(row ** 2, 1e-30)
    rk = (w > ref[..., None] * 1e-12).sum(axis=-1).astype(float)

    gain = gain / np.maximum(row, 1e-12)
    np.fill_diagonal(gain, 0.0)
    np.fill_diagonal(rk, 0.0)
    return gain, rk


# The non-trivial flex: ker(B_G) with ker(B_K) removed. By Michieletto Theorem 1
# the latter IS the trivial variation set, exactly, in every domain and mix, so
# nothing has to be enumerated by hand.
@counted
def flex_space(Z, Z_K, tol=1e-7):
    if Z.shape[1] == 0 or Z_K.shape[1] == 0:
        return Z
    W = Z - Z_K @ (Z_K.T @ Z)
    u, s, _ = np.linalg.svd(W, full_matrices=False)
    return u[:, s > tol]


def node_flex_magnitude(F, n):
    """How free each node is: the norm of its own rows of the flex basis."""
    if F.shape[1] == 0:
        return np.zeros((n, 1))
    Fp = F[:3 * n].reshape(n, 3, -1)
    Fa = F[3 * n:].reshape(n, 3, -1)
    mag = np.sqrt((Fp ** 2).sum(axis=(1, 2)) + (Fa ** 2).sum(axis=(1, 2)))
    return mag[:, None]


@counted
def is_MBR(network, rank_K=None, brmat=None, block_ranks=None, rank_brm=None):
    if int(network.edges.sum()) == 0:
        return False, False, 0

    if brmat is None:
        brmat = extended_bearing_rigidity_matrix(network)

    if rank_K is None:
        network_K = network.fully_connected()
        brmat_K = extended_bearing_rigidity_matrix(network_K)
        rank_K = np.linalg.matrix_rank(brmat_K)

    if rank_brm is None:
        isIBR, rank_brmat = is_IBR_explicit(brmat, rank_K=rank_K)
    else:
        rank_brmat = int(rank_brm)
        isIBR = rank_brmat == rank_K

    if not isIBR:
        return False, isIBR, rank_brmat

    m = int(network.edges.sum())
    c_e = edge_block_ranks(brmat) if block_ranks is None else list(block_ranks)

    c_e_sorted = sorted(c_e, reverse=True)

    sum_c = 0
    m_req = 0
    for c in c_e_sorted:
        sum_c += c
        m_req += 1
        if sum_c >= rank_K:
            break

    return m == m_req, isIBR, rank_brmat

# J. F. Presenza, L. J. Colombo, J. I. Giribet, and I. Mas, “Angle-based Localization and Rigidity Maintenance Control for Multi-Robot Networks,” Apr. 17, 2026, arXiv: arXiv:2604.11754. doi: 10.48550/arXiv.2604.11754.
def isIAR(network):
    print(f"IAR not implemented.")
    quit()
