from util import *
import numpy as np
import copy


def old_extended_bearing_rigidity_matrix(network):
    positions = [agent.pose.position for agent in network.agents]
    rotations = [agent.pose.rotation_mat() for agent in network.agents]
    edges = network.edges

    n = len(positions)
    m = len(edges)

    B = np.zeros((3*m, 6*n))

    for k, (i, j) in enumerate(edges):

        p_ij = positions[j] - positions[i]
        dist = np.linalg.norm(p_ij)
        p_bar_ij = p_ij / dist
        R_i = rotations[i]
        P = orthogonal_projection_matrix(p_bar_ij)

        Q = (R_i.T @ P) / dist
        A = -R_i.T @ skew_symmetric(p_bar_ij)

        rows = slice(3*k, 3*(k+1))

        B[rows, 3*i : 3*i+3] = -Q # agent i vel
        B[rows, 3*j : 3*j+3] = Q # agent j vel

        B[rows, 3*n+3*i : 3*n+3*i+3] = -A # agent i ang vel

    return B

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
    m = len(edges)

    E = np.zeros((n, m))
    Eo = np.zeros((n, m))
    U = np.zeros((3*m, 3*m))
    V = np.zeros((3*m, 3*m))
    for k, (i, j) in enumerate(edges):
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
    for k, (i, j) in enumerate(edges):
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
    B = np.hstack([Bp, Ba])

    return B

def old_is_IBR(network):
    brmat = old_extended_bearing_rigidity_matrix(network)

    print(f"IBR check: {np.linalg.matrix_rank(brmat)} =? {brmat.shape[1] - (6+1)}")
    return np.linalg.matrix_rank(brmat) == brmat.shape[1] - (6+1)

def is_IBR(network):
    # rigidity matrix
    brmat = extended_bearing_rigidity_matrix(network)

    # rigidity matrix of the fully connected graph
    network_K = copy.copy(network)
    n = len(network_K.agents)
    network_K.edges = np.asarray([(i, j) for i in range(n) for j in range(n) if i != j])
    brmat_K = extended_bearing_rigidity_matrix(network_K)

    # print(f"IBR check: {np.linalg.matrix_rank(brmat)} =? {np.linalg.matrix_rank(brmat_K)}")
    return np.linalg.matrix_rank(brmat) == np.linalg.matrix_rank(brmat_K)
