from util import *
import numpy as np
import copy


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

def extended_bearing_rigidity_matrix(network):
    p = [agent.pose.position for agent in network.agents]
    R = [agent.pose.rotation_mat() for agent in network.agents]
    edges = network.edges

    n = len(p)
    m = int(edges.sum())

    E = np.zeros((n, m))
    Eo = np.zeros((n, m))
    U = np.zeros((3*m, 3*m))
    V = np.zeros((3*m, 3*m))

    i_indices, j_indices = np.nonzero(edges)
    # TODO: there should be a more efficient implementation using the adjacency mat, i was lazy
    for k, (i, j) in enumerate(zip(i_indices, j_indices)):
        E[i, k] = -1
        E[j, k] = +1
        Eo[i, k] = -1
        # Uij, Vij
        U[3*k:3*(k+1), 3*k:3*(k+1)], V[3*k:3*(k+1), 3*k:3*(k+1)] = bearing_DOFs(
            network.agents[i], network.agents[j]
            )

    E_bar = np.kron(E, np.eye(3))
    Eo_bar = np.kron(Eo, np.eye(3))

    Dp = np.zeros((3*m, 3*m))
    Da = np.zeros((3*m, 3*m))
    for k, (i, j) in enumerate(zip(i_indices, j_indices)):

        # TODO: not sure if we should do this
        if i == j:
            continue

        pij = p[j] - p[i]
        s = 1.0 / np.linalg.norm(pij)
        p_bar = s * pij

        Ri = R[i]

        P = orthogonal_projection_matrix(p_bar)

        Dp_k = s * Ri.T @ P
        Da_k = -Ri.T @ skew_symmetric(p_bar)

        Dp[3*k:3*(k+1), 3*k:3*(k+1)] = Dp_k

        Da[3*k:3*(k+1), 3*k:3*(k+1)] = Da_k

    Bp = Dp @ U @ E_bar.T
    Ba = Da @ V @ Eo_bar.T
    B = np.hstack([Bp, Ba]) # (3m, 6n)

    return B

def is_IBR_explicit(brmat, brmat_K=None, rank_K=None):
    if rank_K is None:
        rank_K = np.linalg.matrix_rank(brmat_K)
    return np.linalg.matrix_rank(brmat) == rank_K

def is_IBR(network, rank_K=None):
    if int(network.edges.sum()) == 0:
        return False

    # rigidity matrix
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
def is_MBR(network, rank_K=None):
    isIBR = is_IBR(network, rank_K=rank_K)

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

# J. F. Presenza, L. J. Colombo, J. I. Giribet, and I. Mas, “Angle-based Localization and Rigidity Maintenance Control for Multi-Robot Networks,” Apr. 17, 2026, arXiv: arXiv:2604.11754. doi: 10.48550/arXiv.2604.11754.
def isIAR(network):
    print(f"IAR not implemented.")
    quit()
