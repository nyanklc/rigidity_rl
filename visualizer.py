from network import Network
from util import invert_color
import numpy as np
import viser


class Visualizer:
    def __init__(self):
        self.server = viser.ViserServer()

    def draw_viser(
        self,
        network: Network,
        node_color=(0, 255, 0),
        edge_color=(0, 128, 0),
        node_size=1,
        label_prefix="",
    ):
        positions = np.array([agent.pose.position for agent in network.agents])

        # nodes
        for i, p in enumerate(positions):
            self.server.scene.add_icosphere(
                name=f"/node_{label_prefix}{i}",
                radius=node_size,
                color=node_color,
                position=p,
            )

            self.server.scene.add_label(
                name=f"/label_{label_prefix}{i}",
                text=f"{label_prefix}{i}",
                position=p,
            )

        # edges
        edge_line_segments = np.zeros((len(network.edges), 2, 3))
        for k, (i, j) in enumerate(network.edges):
            edge_line_segments[k] = np.array([positions[i], positions[j]])
        self.server.scene.add_line_segments(
            name=f"/edge_{label_prefix}{k}:({i}, {j})",
            points=edge_line_segments,
            colors=np.array(edge_color),
        )

        orientation_positions = positions
        orientation_positions += [
            node_size * agent.pose.rotation_mat() @ np.array([1, 0, 0])
            for agent in network.agents
        ]

        # orientations
        for i, p in enumerate(orientation_positions):
            self.server.scene.add_icosphere(
                name=f"/orientation_{label_prefix}{i}",
                radius=node_size/2,
                color=invert_color(node_color),
                position=p,
            )
