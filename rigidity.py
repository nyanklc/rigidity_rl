from util import *
import numpy as np
import copy


# (S_i, P_i): the translational and rotational coordinates agent i can vary.
# Per node, not per edge -- see THEORY.md §12.
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
# column. See THEORY.md §12 and DESIGN_NOTES.md#per-node-dof.
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

# M. H. Trinh, Q. Van Tran, and H.-S. Ahn, “Minimal and Redundant Bearing Rigidity: Conditions and Applications,” IEEE Transactions on Automatic Control, vol. 65, no. 10, pp. 4186–4200, Oct. 2020, doi: 10.1109/TAC.2019.2958563.
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
# it rather than with an edge count. See DESIGN_NOTES.md#max-edge-rank
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
# See DESIGN_NOTES.md#required-edge-count
def required_edge_count(network, rank_K=None, brmat_K=None):
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

    m_K = brmat_K.shape[0] // 3
    block_ranks = sorted(
        (np.linalg.matrix_rank(brmat_K[3*k:3*(k+1), :]) for k in range(m_K)),
        reverse=True,
    )

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
# carries information on heterogeneous networks. See DESIGN_NOTES.md#rigidity-features
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
# See DESIGN_NOTES.md#rigidity-features
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
    independent and rotation-invariant. See THEORY.md.
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


def is_MBR(network, rank_K=None, brmat=None, block_ranks=None):
    if int(network.edges.sum()) == 0:
        return False, False, 0

    if brmat is None:
        brmat = extended_bearing_rigidity_matrix(network)

    if rank_K is None:
        network_K = network.fully_connected()
        brmat_K = extended_bearing_rigidity_matrix(network_K)
        rank_K = np.linalg.matrix_rank(brmat_K)

    isIBR, rank_brmat = is_IBR_explicit(brmat, rank_K=rank_K)

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
