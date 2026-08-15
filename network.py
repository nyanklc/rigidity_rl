from pyexpat import features

import numpy as np
from util import circle_polygon, move_polygon, Pose, invert_color, discretize_array
import quaternion
import rigidity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import viser
from util import skew_symmetric
from enum import Enum
import copy


class Agent:
    def __init__(self, pose=None):
        self.pose = pose if pose is not None else Pose()
        self.velocity = np.zeros(len(self.pose.position))
        self.angular_velocity = np.zeros(3)
        self.domain = "SE(3)"
        self.rotation_axis = None

    def step(self, dt):
        self.pose.step(self.velocity, self.angular_velocity, dt)

    def set_domain(self, domain, rotation_axis=None):
        self.domain = domain
        if domain == "R^3xS^1":
            # TODO: is this in the world frame?
            rax = (
                np.array([0, 0, 1])
                if rotation_axis is None
                else (rotation_axis / np.linalg.norm(rotation_axis))
            )
            self.rotation_axis = rax
        else:
            self.rotation_axis = None

    """ TODO: we manually clip the velocities here but i'm not sure if the gradient based controller should
    inherently handle this, since the domain information is in the bearing rigidity matrix. """
    def set_velocity(self, vel):
        self.velocity = vel
        if self.domain in ["R^2", "R^2xS^1"]:
            self.velocity[2] = 0.0

    def set_angular_velocity(self, ang_vel):
        if self.domain in ["R^2", "R^3"]:
            self.angular_velocity = np.zeros(3)
        elif self.domain == "R^2xS^1":
            self.angular_velocity = np.zeros(3)
            self.angular_velocity[2] = ang_vel[2]
        else:
            self.angular_velocity = ang_vel

    def get_node_features(self):
        return np.hstack([self.pose.position, self.pose.euler_angles()])

    def get_footprint(self):
        x, y = self.pose.position[:2]
        yaw = quaternion.as_euler_angles(self.pose.orientation)[2]
        polygon = circle_polygon()
        footprint = move_polygon(polygon, x, y, yaw)
        return footprint

    def get_bearing(self, other: "Agent"):
        p = other.pose.position - self.pose.position
        p = p / np.linalg.vector_norm(p)
        bearing = np.zeros(3)
        if self.domain not in ["R^3", "R^2"]:
            # bearing in body frame
            bearing = self.pose.rotation_mat().T @ p
        else:
            bearing = p
        return bearing

    # the azimuth and elevation of the bearing vector
    def get_bearing_angles(self, other: "Agent"):
        bearing = self.get_bearing(other)
        theta = np.atan2(bearing[1], bearing[0])
        phi = np.arccos(bearing[2])
        return np.asarray([theta, phi])

    def randomize_position(self, low=[-100, -100, -100], high=[100, 100, 100]):
        if self.domain in ["R^2", "R^2xS^1"]:
            pos = np.random.uniform(low[:2], high[:2], size=2)
            self.pose.position = np.hstack([pos, [0.0]])
        else:
            self.pose.position = np.random.uniform(low[:3], high[:3], size=3)

    def randomize_orientation(self):
        if self.domain == "SE(3)":
            euler = np.random.uniform(0, 2*np.pi, size=3)
            self.pose.orientation = quaternion.from_euler_angles(euler)
        elif self.domain in ["R^3xS^1", "R^2xS^1"]:
            axis = self.rotation_axis if self.rotation_axis is not None else np.array([0,0,1])
            angle = np.random.uniform(0, 2*np.pi)
            self.pose.orientation = quaternion.from_rotation_vector(axis * angle)
        else:
            self.pose.orientation = quaternion.one


class Network:
    def __init__(self, positions, orientations_euler, edges):
        self.n = len(positions)
        self.edges = np.zeros((self.n, self.n), dtype=bool)
        if edges is not None:
            if edges.shape[1] == self.n: # adj matrix provided
                self.edges = edges
                np.fill_diagonal(self.edges, False) # ignore self-loops
            else: # edge list provided
                for k, (i, j) in enumerate(edges):
                    if i != j:
                        self.edges[i, j] = True
        self.agents: list[Agent] = []
        for i in range(len(positions)):
            self.agents.append(Agent(Pose(positions[i], orientations_euler[i])))
        self.nr_max_edges = self.n**2

    def step(self, dt):
        for agent in self.agents:
            agent.step(dt)

    def set_inputs(self, u):
        n = len(self.agents)
        for i in range(n):
            self.agents[i].set_velocity(u[3 * i : 3 * i + 3])
            self.agents[i].set_angular_velocity(u[3 * n + 3 * i : 3 * n + 3 * i + 3])

    def translate_network(self, dp):
        for agent in self.agents:
            agent.pose.position += np.array(dp)

    def rotate_network(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        positions = np.array([agent.pose.position for agent in self.agents])
        center = np.mean(positions, axis=0)

        K = skew_symmetric(axis)
        # rodrigues
        R = (
            np.eye(3)
            + np.sin(angle) * K
            + (1 - np.cos(angle)) * (K @ K)
        )
        # rotate around center
        positions = (positions - center) @ R.T + center

        for i, agent in enumerate(self.agents):
            agent.pose.position = positions[i]

            if agent.domain not in ["R^3", "R^2"]:
                # rotate the orientations also
                R_agent = agent.pose.rotation_mat()
                agent.pose.set_rotation_mat(R @ R_agent)

            # P_i = v v^T is in world coordinates, so a world rotation carries the
            # controllable axis with it; otherwise this is not a symmetry
            if agent.rotation_axis is not None:
                agent.rotation_axis = R @ np.asarray(agent.rotation_axis, dtype=float)

    # positions are numpy arrays; the old .x/.y/.z form raised AttributeError.
    # Scale about the centroid, so a uniform scale is the trivial motion of
    # THEORY.md section 3 and leaves every bearing unchanged.
    def scale_network(self, scale):
        s = np.asarray(scale, dtype=float)
        if s.ndim == 0:
            s = np.repeat(s, 3)
        positions = np.array([agent.pose.position for agent in self.agents])
        centre = positions.mean(axis=0)
        for agent in self.agents:
            agent.pose.position = centre + (agent.pose.position - centre) * s

    def randomize_positions(self, low, high):
        for agent in self.agents:
            agent.randomize_position(low, high)

    def randomize_orientations(self):
        for agent in self.agents:
            agent.randomize_orientation()

    def set_edges(self, edges):
        if edges is not None:
            self.edges = edges

    def set_edges_indices(self, i_indices, j_indices):
        n = len(self.agents)
        self.edges = np.zeros((n, n), dtype=bool)
        m = len(i_indices)
        if m == 0:
            return
        self.edges[i_indices, j_indices] = True

    def set_edges_list(self, lst):
        n = len(self.agents)
        self.edges = np.zeros((n, n), dtype=bool)
        m = len(lst)
        if m == 0:
            return

        for i, j in lst:
            self.edges[i, j] = True

    def add_edge(self, i_idx, j_idx):
        if i_idx != j_idx:
            self.edges[i_idx, j_idx] = True

    def remove_edge(self, i_idx, j_idx):
        self.edges[i_idx, j_idx] = False

    def edge_exists(self, i_idx, j_idx):
        return self.edges[i_idx, j_idx]

    def get_edge_list(self):
        lists = np.nonzero(self.edges)
        return [(int(lists[0][i]), int(lists[1][i])) for i in range(len(lists[0]))]

    def set_agents_domain_homogeneous(self, domain: str, rotation_axis=None):
        # print(f"agents' domain: {domain}")
        # default values are for SE(3)
        if domain == "R^3xSO(3)" or domain == "" or domain == None:
            domain = "SE(3)"

        if (
            domain != "SE(3)"
            and domain != "R^3xS^1" # TODO: old todo what is this?: TODO: rotation axis is cumbersome to support (e.g. in velocity clipping)
            and domain != "R^2xS^1"
            and domain != "R^3"
            and domain != "R^2"
        ):
            print(f"given agents' domain {domain} is not valid.")
            quit()

        for agent in self.agents:
            agent.set_domain(domain, rotation_axis)

    def extended_bearing_rigidity_matrix(self):
        return rigidity.extended_bearing_rigidity_matrix(self)

    def is_IBR(self, rank_K=None):
        return rigidity.is_IBR(self, rank_K=rank_K)

    # also returns is IBR
    def is_MBR(self, rank_K=None, brm=None, block_ranks=None, rank_brm=None):
        # for agent in self.agents:
        #     if agent.domain not in ["R^2", "R^3"]:
        #         raise Exception("Minimally Bearing Rigidity is not defined for domains other than R^d.")

        return rigidity.is_MBR(self, rank_K=rank_K, brmat=brm, block_ranks=block_ranks,
                               rank_brm=rank_brm)

    def eigenvalues(self, eps=1e-10):
        brm = self.extended_bearing_rigidity_matrix()
        information_mat = brm.T @ brm
        # symmetric
        eigenvalues = np.linalg.eigvalsh(information_mat)
        eigenvalues[np.abs(eigenvalues) < eps] = 0.0
        eigenvalues.sort()
        return eigenvalues

    # 3M
    def get_bearings(self):
        i_indices, j_indices = np.nonzero(self.edges)
        m = len(i_indices)
        bearings = np.zeros(3 * m)
        for k, (i, j) in enumerate(zip(i_indices, j_indices)):
            bearings[3*k:3*k+3] = self.agents[i].get_bearing(self.agents[j])
        return bearings

    # N, N, 3
    def get_bearings_explicit(self):
        n = len(self.agents)
        i_indices, j_indices = np.arange(n), np.arange(n)
        b = np.zeros((n, n, 3))
        for i in i_indices:
            for j in j_indices:
                if i != j and self.edges[i, j]:
                    b[i, j] = self.agents[i].get_bearing(self.agents[j])
        return b

    # N, N, 2
    def get_bearing_angles_explicit(self):
        n = len(self.agents)
        i_indices, j_indices = np.arange(n), np.arange(n)
        b = np.zeros((n, n, 2))
        for i in i_indices:
            for j in j_indices:
                if i != j and self.edges[i, j]:
                    b[i, j] = self.agents[i].get_bearing_angles(self.agents[j])
        return b

    def get_pose_features(self):
        features = [agent.get_node_features() for agent in self.agents] # 6 each, pos+ori
        return features

    # N, 3
    def get_domain_features(self):
        agent_domain_feature = {
            "R^2":     np.array([0, 0, 0, 0, 1]),
            "R^2xS^1": np.array([0, 0, 0, 1, 0]),
            "R^3":     np.array([0, 0, 1, 0, 0]),
            "R^3xS^1": np.array([0, 1, 0, 0, 0]),
            "SE(3)":   np.array([1, 0, 0, 0, 0]),
        }
        feats = np.asarray([agent_domain_feature[agent.domain] for agent in self.agents])
        return feats

    # N, 3
    def get_orientation_features(self):
        orientations = np.asarray([agent.pose.euler_angles() for agent in self.agents])
        return orientations

    # N, 3
    def get_position_features(self):
        # features = np.asarray([agent.get_node_features() for agent in self.agents]) # 6 each, pos+ori
        features = np.asarray([agent.pose.position for agent in self.agents])
        return features

    # N, 3 -- centred on the centroid and scaled to unit RMS radius.
    # See DESIGN_NOTES.md#pose-normalization
    def get_normalized_position_features(self, eps=1e-9):
        p = self.get_position_features().astype(float)
        p = p - p.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(p**2, axis=-1)))
        return p / max(rms, eps)

    # N, N, 3
    def get_bearing_features(self):
        existing_bearing_features = self.get_bearings_explicit()
        return existing_bearing_features

    # N, N, 3 -- every ordered pair, in the WORLD frame regardless of domain.
    # The rigidity algebra (flex tensor, projectors) lives in world coordinates,
    # so anything contracted against it must too. get_all_pairs_bearings() returns
    # the body-frame *measurement* instead, which differs by R_i for oriented
    # domains. See THEORY.md section 9.
    def get_all_pairs_bearings_world(self):
        n = len(self.agents)
        p = np.array([a.pose.position for a in self.agents])
        d = p[None, :, :] - p[:, None, :]
        norm = np.linalg.norm(d, axis=-1, keepdims=True)
        np.fill_diagonal(norm[:, :, 0], 1.0)
        return d / norm

    # N, N, 3 -- every ordered pair, whether or not the edge exists.
    # Candidate-edge geometry; see DESIGN_NOTES.md#all-pairs-bearings
    def get_all_pairs_bearings(self):
        n = len(self.agents)
        b = np.zeros((n, n, 3))
        for i in range(n):
            for j in range(n):
                if i != j:
                    b[i, j] = self.agents[i].get_bearing(self.agents[j])
        return b

    # N, N, 1
    def get_edge_exists_features(self):
        return self.edges.astype(float)[:, :, np.newaxis]

    # N, N, 3
    def get_bearing_features_discrete(self):
        existing_bearing_features = self.get_bearings_explicit()
        return discretize_array(existing_bearing_features)

    # N, N, 2
    def get_bearing_angle_features(self):
        existing_bearing_angle_features = self.get_bearing_angles_explicit()
        return existing_bearing_angle_features

    # N, N, 2
    def get_bearing_angle_features_discrete(self):
        existing_bearing_angle_features = self.get_bearing_angles_explicit()
        return discretize_array(existing_bearing_angle_features)

    # N, N, 3
    def get_simplified_bearing_features(self):
        existing_bearing_features = self.get_bearings_explicit()
        return np.sign(existing_bearing_features)

    # M, 3
    def get_edge_bearing_features(self):
        existing_bearing_features = self.get_bearings_explicit() # N, N, 3
        return existing_bearing_features[self.edges]

    # N, 2
    def get_degree_features(self):
        # in-degree and out-degree
        out_degree = np.sum(self.edges, axis=1)
        in_degree = np.sum(self.edges, axis=0)
        return np.column_stack((in_degree, out_degree))

    # N, 2 -- degree relative to the mean degree of a MINIMAL graph, m_req/n.
    # Dividing by (n-1) instead would over-correct: required edges grow linearly in
    # n while the pair count grows quadratically, so that introduces a 1/n trend
    # where there was none. Here a node at the target density reads ~1.
    # See DESIGN_NOTES.md#aggregation-and-scale
    def get_degree_features_normalized(self, m_req=None):
        if m_req is None:
            m_req = rigidity.required_edge_count(self)
        return self.get_degree_features() / max(m_req / max(self.n, 1), 1e-6)

    # N, N, 1 -- common neighbours per unit of mean degree. Dividing by (n-2)
    # over-corrects: the raw count grows only mildly with n (0.9 -> 1.9 from n=8 to
    # 32), so the pair count is the wrong yardstick.
    def get_common_neighbors_features_normalized(self, m_req=None):
        if m_req is None:
            m_req = rigidity.required_edge_count(self)
        return self.get_common_neighbors_features() / max(m_req / max(self.n, 1), 1e-6)

    # N, 1
    def get_closeness_centrality_features(self):
        n = self.n
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        dist[self.edges] = 1

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        closeness = np.zeros(n)
        if n > 1:
            for i in range(n):
                valid_dists = dist[i, dist[i, :] < np.inf]
                reachable = len(valid_dists) - 1
                if reachable > 0:
                    closeness[i] = (reachable / np.sum(valid_dists)) * (reachable / (n - 1))
        return closeness[:, np.newaxis]

    # N, 1
    def get_eigenvector_centrality_features(self, max_iter=100, tol=1e-6):
        n = self.n
        x = np.ones(n) / n
        A = self.edges.astype(float)

        for _ in range(max_iter):
            x_next = A.T @ x
            norm = np.linalg.norm(x_next)
            if norm == 0:
                return x[:, np.newaxis]
            x_next = x_next / norm
            if np.linalg.norm(x_next - x) < tol:
                return x_next[:, np.newaxis]
            x = x_next
        return x[:, np.newaxis]

    def _brandes_betweenness(self):
        n = self.n
        node_betweenness = np.zeros(n)
        edge_betweenness = np.zeros((n, n))

        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if self.edges[i, j]:
                    adj[i].append(j)

        for s in range(n):
            S = []
            P = {w: [] for w in range(n)}
            sigma = np.zeros(n)
            sigma[s] = 1.0
            d = np.full(n, -1.0)
            d[s] = 0.0

            Q = [s]
            while Q:
                v = Q.pop(0)
                S.append(v)
                for w in adj[v]:
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1.0
                    if d[w] == d[v] + 1.0:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            delta = np.zeros(n)
            while S:
                w = S.pop()
                for v in P[w]:
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    edge_betweenness[v, w] += c
                    delta[v] += c
                if w != s:
                    node_betweenness[w] += delta[w]

        return node_betweenness, edge_betweenness

    # N, 1
    def get_node_betweenness_features(self):
        nb, _ = self._brandes_betweenness()
        return nb[:, np.newaxis]

    # N, N, 1
    def get_edge_betweenness_features(self):
        _, eb = self._brandes_betweenness()
        return eb[:, :, np.newaxis]

    # N, N, 1
    def get_edge_reciprocity_features(self):
        reciprocal = (self.edges & self.edges.T).astype(float)
        return reciprocal[:, :, np.newaxis]

    # N, N, 1
    def get_common_neighbors_features(self):
        A = self.edges.astype(float)
        common = A @ A
        return common[:, :, np.newaxis]

    def fully_connected(self):
        network_K = copy.copy(self)
        n = len(network_K.agents)
        # no self loops: a diagonal entry produced a 3-row block of zeros, which
        # left rank_K correct only because extended_bearing_rigidity_matrix skips
        # i == j. Anything that iterates the blocks (required_edge_count) would
        # otherwise have to know about the padding.
        network_K.edges = ~np.eye(n, dtype=bool)
        return network_K # TODO: return does copy?

    def print(self):
        print(f"NETWORK")
        for i, agent in enumerate(self.agents):
            print(
                f"agent {i} in {agent.domain} with rotation axis {agent.rotation_axis}"
            )
            agent.pose.print()
        print("edges:")
        n = len(self.agents)
        edge_list = self.get_edge_list()
        for i, j in edge_list:
            print(f"{i} -> {j}")

    def __str__(self):
        lines = []
        lines.append("NETWORK")

        for i, agent in enumerate(self.agents):
            lines.append(
                f"agent {i} in {agent.domain} with rotation axis {agent.rotation_axis}"
            )
            # Assuming agent.pose.print() also needs to be a string:
            lines.append(str(agent.pose))

        lines.append("edges:")
        edge_list = self.get_edge_list()
        for i, j in edge_list:
            lines.append(f"{i} -> {j}")

        return "\n".join(lines)
