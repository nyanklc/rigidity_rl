import numpy as np
from network import Network
from rigidity import *
import copy


# graph/network
positions = (
    np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            # [0, 0, 1],
            # [1, 0, 1],
            # [1, 1, 1],
            # [0, 1, 1],
        ],
        dtype=float,
    )
    * 50
)
n = len(positions)
orientations_euler = np.zeros((n, 3))
# orientations_euler = np.random.rand(n, 3)
# fully connected (no self loops)
edges = np.asarray([(i, j) for i in range(n) for j in range(n) if i != j])
# edges = np.asarray([
#     (0, 1),
#     (1, 2),
#     (2, 3),
#     (3, 0),
#     (4, 5),
#     (5, 6),
#     (6, 7),
#     (7, 4),
#     (0, 4),
#     (1, 5),
#     (2, 6),
#     (3, 7),
#     (0, 6),
# ])
print(f"----------------network----------------")
network = Network(positions, orientations_euler, edges)
bearings = network.get_bearings()
network.print()
print(f"bearings: {bearings}")
print(f"rigid: {network.is_IBR()}")


a = old_extended_bearing_rigidity_matrix(network)
print(f"HELLO: old: {a}")
b = extended_bearing_rigidity_matrix(network)
print(f"HELLO: new: {b}")

print(f"diff sum: {np.sum(a - b)}")

print(f"is IBR old: {old_is_IBR(network)}")
print(f"is IBR new: {is_IBR(network)}")

print(f"n: {len(positions)}")
print(f"m: {len(edges)}")
print(f"dim a: {a.shape}")
print(f"dim b: {b.shape}")

aq = a.flatten()
bq = b.flatten()
for idx, (ai, bi) in enumerate(zip(aq, bq)):
    if ai != bi:
        print(f"Index {idx}: a = {ai}, b = {bi}")
