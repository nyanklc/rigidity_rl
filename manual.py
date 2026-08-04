import copy
import numpy as np
import time
import sys
import os

from environment import Environment
from visualizer import Visualizer


if len(sys.argv) < 2:
    print("usage: python3 manual.py [environment_name]")
    quit()

env_name = sys.argv[1]
filepath = "./environments/" + env_name + ".json"

if not os.path.exists(filepath):
    print(f"{filepath} not found")
    quit()

raw_env = Environment()
raw_env.load(filepath)

netw = raw_env.network
n = len(netw.agents)

vis = Visualizer()

# ─── UI ──────────────────────────────────────────────────────────────────────

tab_group = vis.server.gui.add_tab_group()

# ── Tab 1: Quick edge entry via text ─────────────────────────────────────────
with tab_group.add_tab("Quick Edge"):
    vis.server.gui.add_markdown(
        f"Type **i, j** (0-indexed, 0..{n-1}) to specify an edge."
    )
    text_edge = vis.server.gui.add_text("Edge (i, j)", initial_value="0, 1")
    button_add_text = vis.server.gui.add_button("Add Edge")
    button_remove_text = vis.server.gui.add_button("Remove Edge")
    button_toggle_text = vis.server.gui.add_button("Toggle Edge")

# ── Tab 2: Adjacency grid with checkboxes ────────────────────────────────────
with tab_group.add_tab("Adjacency Grid"):
    vis.server.gui.add_markdown("Toggle directed edges directly. Row → Col.")

    # header row
    header = "` ` | " + " | ".join(f"**{j}**" for j in range(n))
    vis.server.gui.add_markdown(header)

    # create an n×n grid of checkboxes (skip diagonal)
    adj_checkboxes: dict[tuple[int, int], object] = {}
    for i in range(n):
        with vis.server.gui.add_folder(f"From node {i}"):
            for j in range(n):
                if i == j:
                    continue
                cb = vis.server.gui.add_checkbox(
                    f"{i} → {j}",
                    initial_value=bool(netw.edges[i, j]),
                )
                adj_checkboxes[(i, j)] = cb

# ── Always-visible actions ────────────────────────────────────────────────────
with vis.server.gui.add_folder("Actions"):
    button_reset = vis.server.gui.add_button("Reset")
    button_remove_all = vis.server.gui.add_button("Remove All Edges")
    button_complete = vis.server.gui.add_button("Complete Graph")
    button_sync_grid = vis.server.gui.add_button("Sync Grid ↔ Network")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_edge(text: str) -> tuple[int, int] | None:
    """Parse 'i, j' or 'i j' from text input. Returns None on bad input."""
    text = text.strip()
    # support both comma and space separators
    for sep in [",", " "]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) == 2:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    if 0 <= i < n and 0 <= j < n and i != j:
                        return (i, j)
                except ValueError:
                    pass
    return None


def sync_checkboxes_from_network():
    """Push the current network edge state into the checkbox grid."""
    for (i, j), cb in adj_checkboxes.items():
        cb.value = bool(netw.edges[i, j])


def sync_network_from_checkboxes():
    """Pull checkbox state into the network edge matrix."""
    for (i, j), cb in adj_checkboxes.items():
        netw.edges[i, j] = cb.value


def redraw(raw_env):
    vis.reset()
    vis.draw_viser(netw)

    mbr = netw.is_MBR()

    info_str = f"""
    edges: {netw.get_edge_list()}\n
    eigs: {netw.eigenvalues()}\n
    eigs sum: {np.sum(netw.eigenvalues())}\n
    MBR, IBR: {mbr[0]}, {mbr[1]}
    """

    vis.draw_info(info_str)
    vis.server.flush()


# ─── Initial draw ────────────────────────────────────────────────────────────
redraw(raw_env)


# ─── Main loop ───────────────────────────────────────────────────────────────

# track previous checkbox state to detect toggles
prev_cb_state: dict[tuple[int, int], bool] = {
    k: cb.value for k, cb in adj_checkboxes.items()
}

while True:
    vis.server.flush()
    time.sleep(0.05)

    # ── Text input: Add ──────────────────────────────────────────────────
    if button_add_text.value:
        button_add_text.value = False
        edge = parse_edge(text_edge.value)
        if edge:
            i, j = edge
            netw.add_edge(i, j)
            sync_checkboxes_from_network()
            redraw(raw_env)

    # ── Text input: Remove ───────────────────────────────────────────────
    if button_remove_text.value:
        button_remove_text.value = False
        edge = parse_edge(text_edge.value)
        if edge:
            i, j = edge
            netw.remove_edge(i, j)
            sync_checkboxes_from_network()
            redraw(raw_env)

    # ── Text input: Toggle ───────────────────────────────────────────────
    if button_toggle_text.value:
        button_toggle_text.value = False
        edge = parse_edge(text_edge.value)
        if edge:
            i, j = edge
            if netw.edge_exists(i, j):
                netw.remove_edge(i, j)
            else:
                netw.add_edge(i, j)
            sync_checkboxes_from_network()
            redraw(raw_env)

    # ── Adjacency grid: detect checkbox changes ──────────────────────────
    grid_changed = False
    for (i, j), cb in adj_checkboxes.items():
        current = cb.value
        if current != prev_cb_state[(i, j)]:
            grid_changed = True
            prev_cb_state[(i, j)] = current

    if grid_changed:
        sync_network_from_checkboxes()
        redraw(raw_env)

    # ── Bulk actions ─────────────────────────────────────────────────────
    if button_reset.value:
        button_reset.value = False
        raw_env.reset()
        sync_checkboxes_from_network()
        redraw(raw_env)

    if button_remove_all.value:
        button_remove_all.value = False
        for i, j in netw.get_edge_list():
            netw.remove_edge(i, j)
        sync_checkboxes_from_network()
        redraw(raw_env)

    if button_complete.value:
        button_complete.value = False
        for i in range(n):
            for j in range(n):
                if i != j:
                    netw.add_edge(i, j)
        sync_checkboxes_from_network()
        redraw(raw_env)

    if button_sync_grid.value:
        button_sync_grid.value = False
        sync_checkboxes_from_network()
        redraw(raw_env)
