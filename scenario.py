import numpy as np
from network import Network
from util import Pose
from control import *
import quaternion
import json


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
                # [0.5, 0.5, 2],
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
    # edges = np.asarray(
    #     [
    #         (0, 1),
    #         (1, 2),
    #         (2, 3),
    #         (3, 0),
    #         (0, 2),
    #     ]
    # )
    print(f"----------------network----------------")
    network = Network(positions, orientations_euler, edges)
    network.set_agents_domain_homogeneous("R^2xS^1")
    # network.agents[4].set_domain("SE(3)")
    bearings = network.get_bearings()
    network.print()
    print(f"bearings: {bearings}")
    print(f"rigid: {network.is_IBR()}")

    # goal
    goal_positions = (
        np.array(
            [
                [2, -2, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0],
                # [2.5, 3.5, 2],
            ],
            dtype=float,
        )
        * 50
    )
    center = np.mean(goal_positions, axis=0)
    rotate = Pose(orientation_euler=(0, 0, np.pi/4)).rotation_mat()
    goal_positions = (goal_positions - center) @ rotate.T + center
    goal_network = Network(goal_positions, orientations_euler, edges)
    goal_network.set_agents_domain_homogeneous("R^2xS^1")
    # goal_network.agents[4].set_domain("SE(3)")
    goal_bearings = goal_network.get_bearings()
    print(f"----------------goal network----------------")
    goal_network.print()
    print(f"goal bearings: {goal_bearings}")
    print(f"goal rigid: {goal_network.is_IBR()}")

    print("#####################################")
    print("#####################################")
    print("#####################################")


    save_scenario("./scenarios/" + filename + ".json", network, goal_network)
