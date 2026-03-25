from network import Network
from util import invert_color
import numpy as np
import viser
import quaternion
import time


class Visualizer:
    def __init__(self):
        self.server = viser.ViserServer()
        while len(self.server.get_clients()) < 1:
            print(f"waiting on viser client connection")
            time.sleep(1)

    def stop(self):
        self.server.stop()

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
        for k, (i, j) in enumerate(network.edges):
            edge_line_segment = np.array([[positions[i], positions[j]]])
            self.server.scene.add_line_segments(
                name=f"/edge_{label_prefix}{k}:({i}, {j})",
                points=edge_line_segment,
                colors=np.array(edge_color),
            )

        # orientation_positions = positions
        # orientation_positions += [
        #     node_size * agent.pose.rotation_mat() @ np.array([1, 0, 0])
        #     for agent in network.agents
        # ]
        # # orientations
        # for i, p in enumerate(orientation_positions):
        #     self.server.scene.add_icosphere(
        #         name=f"/orientation_{label_prefix}{i}",
        #         radius=node_size/2,
        #         color=invert_color(node_color),
        #         position=p,
        #     )

        # orientations
        for i, agent in enumerate(network.agents):
            wxyz = np.asarray(quaternion.as_float_array(agent.pose.orientation))
            self.server.scene.add_frame(
                name=f"/frame_{label_prefix}{i}",
                axes_length=2,
                axes_radius=0.2,
                position=agent.pose.position,
                # w is stored on the last index in our case
                wxyz=wxyz
            )
