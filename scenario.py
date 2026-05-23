import numpy as np
from network import Network
from util import Pose
from control import *
import quaternion
import json
import copy
import os


# TODO: collinearity check
def random_scenario(
    n,
    domains: str | list[str] = "SE(3)",
    pos_limits=([-100, -100, -100], [100, 100, 100]),
):
    low, high = np.array(pos_limits[0]), np.array(pos_limits[1])
    positions = np.zeros((n, 3))
    orientations_euler = np.random.uniform(0, 2 * np.pi, size=(n, 3))

    network = Network(positions, orientations_euler, edges=np.zeros((0, 2), dtype=int))
    if isinstance(domains, str):
        network.set_agents_domain_homogeneous(domains)
    else:
        for agent, domain in zip(network.agents, domains):
            agent.set_domain(domain)
    network.randomize_positions(low, high)
    network.randomize_orientations()

    edge_set = set()
    max_possible_edges = n**2 - n # no self loops
    m = np.random.randint(0, max_possible_edges + 1)
    while len(edge_set) < m:
        i, j = np.random.choice(n, size=2, replace=False)
        if ((i, j) not in edge_set):
            edge_set.add((i, j))
    edges = np.array(list(edge_set))
    if len(edge_set) == 0:
        network.set_edges(None)
    else:
        network.set_edges_indices(edges[:, 0], edges[:, 1])

    orientations_euler = np.random.uniform(0, 2 * np.pi, size=(n, 3))
    goal_network = Network(
        positions, orientations_euler, edges=np.zeros((0, 2), dtype=int)
    )
    if isinstance(domains, str):
        goal_network.set_agents_domain_homogeneous(domains)
    else:
        for agent, domain in zip(goal_network.agents, domains):
            agent.set_domain(domain)
    goal_network.randomize_positions(low, high)
    goal_network.randomize_orientations()
    # same edges
    goal_network.set_edges(network.edges)

    return network, goal_network


def save_scenario(filename, network, goal_network):
    data = {
        "positions": [agent.pose.position.tolist() for agent in network.agents],
        "orientations_euler": [
            quaternion.as_euler_angles(agent.pose.orientation).tolist()
            for agent in network.agents
        ],
        "edges": network.edges.tolist(),
        "domains": [agent.domain for agent in network.agents],
        "rotation_axes": [
            agent.rotation_axis.tolist() if agent.rotation_axis is not None else None
            for agent in network.agents
        ],

        "goal_positions": [agent.pose.position.tolist() for agent in goal_network.agents],
        "goal_orientations_euler": [
            quaternion.as_euler_angles(agent.pose.orientation).tolist()
            for agent in goal_network.agents
        ],
        "goal_edges": goal_network.edges.tolist(),
    }

    with open(filename, "w") as f:
        os.makedirs("./scenarios/", exist_ok=True)
        json.dump(data, f, indent=2)


def load_scenario(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    network = Network(
        np.array(data["positions"]),
        np.array(data["orientations_euler"]),
        np.array(data["edges"]),
    )
    for agent, domain in zip(network.agents, data["domains"]):
        agent.domain = domain
    for agent, rax in zip(network.agents, data["rotation_axes"]):
        agent.rotation_axis = np.array(rax)

    goal_network = Network(
        np.array(data["goal_positions"]),
        np.array(data["goal_orientations_euler"]),
        np.array(data["edges"]),
    )
    for agent, domain in zip(goal_network.agents, data["domains"]):
        agent.domain = domain
    for agent, rax in zip(goal_network.agents, data["rotation_axes"]):
        agent.rotation_axis = np.array(rax)

    return network, goal_network


####################################################
####################################################
####################################################
####################################################
####################################################

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"input filename as argument")
    filename = sys.argv[1]

    # graph/network
    positions = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0.5, 0.5, 1],
                # [1, 0, 1],
                # [1, 1, 1],
                # [0, 1, 1],
                # [0.5, 0.5, 1],
            ],
            dtype=float,
        )
        * 50
    )
    n = len(positions)
    orientations_euler = np.zeros((n, 3))
    # edges = None
    edges = np.asarray([(i, j) for i in range(n) for j in range(n) if i != j])
    # edges = np.asarray(
    #     [
    #         (0, 1),
    #         (1, 2),
    #         (2, 3),
    #         (3, 0),
    #         # (4, 5),
    #         # (5, 6),
    #         # (6, 7),
    #         # (7, 4),
    #         # (0, 4),
    #         # (1, 5),
    #         # (2, 6),
    #         # (3, 7),
    #         # (0, 6),
    #         (0, 2),
    #     ]
    # )
    network = Network(positions, orientations_euler, edges)
    network.set_agents_domain_homogeneous("R^2")
    network.agents[4].set_domain("SE(3)")
    bearings = network.get_bearings()

    # goal
    goal_network = copy.deepcopy(network)
    # goal_network.agents[4].pose.position = np.array([-50, -50, 50])
    # goal_network.agents[5].pose.position = np.array([100, -50, 50])
    # goal_network.agents[6].pose.position = np.array([100, 100, 50])
    # goal_network.agents[7].pose.position = np.array([-50, 100, 50])
    goal_network.agents[0].pose.position = np.array([-50, 25, 0])
    # goal_network.translate_network([150, 100, 0])
    # goal_network.rotate_network([0, 0, 1], np.pi/4)
    # goal_network.agents[4].pose.position[0] += 50
    goal_bearings = goal_network.get_bearings()



    print(f"----------------network----------------")
    network.print()
    print(f"bearings: {bearings}")
    print(f"rigid: {network.is_IBR()}")
    print(f"----------------goal network----------------")
    goal_network.print()
    print(f"goal bearings: {goal_bearings}")
    print(f"goal rigid: {goal_network.is_IBR()}")

    print("#####################################")
    print("#####################################")
    print("#####################################")


    save_scenario("./scenarios/" + filename + ".json", network, goal_network)
