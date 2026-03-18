from util import *

import numpy as np


# TODO: implement the extended bearing rigidity matrix instead, this is the
# simple case


# closed form jacobian of the bearing function
def bearing_rigidity_matrix(positions, edges):
    n = len(positions)
    d = positions.shape[1]
    m = len(edges)

    B = np.zeros((m * d, n * d))

    for k, (i, j) in enumerate(edges):

        pi = positions[i, :]
        pj = positions[j, :]

        diff = pj - pi
        dist = np.linalg.norm(diff)

        bearing = diff / dist

        projection_mat = np.eye(d) - np.outer(bearing, bearing)

        Q = 1 / dist * projection_mat

        B[k * d : (k + 1) * d, i * d : (i + 1) * d] = -Q
        B[k * d : (k + 1) * d, j * d : (j + 1) * d] = Q

    return B


def is_IBR(brmat, d):
    return np.linalg.matrix_rank(brmat) == brmat.shape[1] - (d + 1)


n = 4
m = 5
d = 2
p = np.random.rand(n, d)
edges = np.zeros((m, 2), dtype=np.int32)
edges[0][0] = 0
edges[0][1] = 1
edges[1][0] = 1
edges[1][1] = 2
edges[2][0] = 2
edges[2][1] = 3
edges[3][0] = 3
edges[3][1] = 0
edges[4][0] = 0
edges[4][1] = 2

print(f"hey p: {p}")
print(f"hey edges: {edges}")

B = bearing_rigidity_matrix(p, edges)

rank = np.linalg.matrix_rank(B)

is_rigid = False
if is_IBR(B, d):
    print("infinitesimally bearing rigid")
    is_rigid = True

print(f"B: {B} B shape: {B.shape}")
print(f"rank(B): {rank}")
print(f"rank == {d*len(p) - (d+1)}? {is_rigid}")

u, s, v = np.linalg.svd(B)
# print(f"singular values left: {u}")
print(f"singular values: {s}")
# print(f"singular values right: {v}")

# plot_graph(p, edges)
