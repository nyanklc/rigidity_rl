from util import *
import numpy as np


def extended_bearing_rigidity_matrix(positions, rotations, edges):
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

        # print(f"k, i, j: {k, i, j}")
        # print(f"p_ij: {p_ij}")
        # print(f"dist: {dist}")
        # print(f"g_ij: {g_ij}")
        # print(f"R_i: {R_i}")
        # print(f"P: {P}")
        # print(f"Q: {Q}")
        # print(f"A: {A}")

        rows = slice(3*k, 3*(k+1))

        B[rows, 6*i : 6*i+3] = -Q # agent i vel
        B[rows, 6*i+3 : 6*i+6] = -A # agent i ang vel

        B[rows, 6*j : 6*j+3] = Q # agent j vel

    return B


def is_IBR(brmat, d):
    print(f"IBR check: {np.linalg.matrix_rank(brmat)} =? {brmat.shape[1] - (d + 1)}")
    return np.linalg.matrix_rank(brmat) == brmat.shape[1] - (d + 1)
