###############################################################################
# File: main.py                                                                #
# Authors: Michael Reis, Jonathan Williams, Cole Sullivan                      #
# -----                                                                        #
# !!!! ADD DESCRIPTION OF FILE !!!!!
# -----                                                                        #
# Last Modified: Monday, April 15th 2024 17:12:55                              #
# Modified By: Michael Reis                                                    #
###############################################################################

import sys
import matplotlib.animation
import matplotlib.pyplot as plt
import my_networkx as my_nx
import networkx as nx
import numpy as np
import os

temp = 0

current_step = 0

# List of tuples (edges, title), necessary for generating animation
path = [([], "")]



def create_graph(cap_matrix):
    n = len(cap_matrix)
    graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if cap_matrix[i][j] > 0:
                graph.add_edge(i, j, residual_capacity=cap_matrix[i][j], capacity=cap_matrix[i][j], flow=0)
                if not graph.has_edge(j, i):
                    graph.add_edge(j, i, residual_capacity = 0, capacity=0, flow=0)
    return graph


def bfs(graph: nx.DiGraph, source, sink, parent):
    visited = [False] * len(graph.nodes)
    visited[source] = True
    global current_step
    parent[source] = -2
    queue = [(0, float('inf'))]
    

    while queue:
        curr, flow = queue.pop(0)
        for next in graph.neighbors(curr):
            if graph[curr][next]["residual_capacity"] > 0 and not visited[next]:
                new_flow = min(flow, graph[curr][next]["residual_capacity"] )
                visited[next] = True
                parent[next] = curr
                if (next == sink):
                    curr1 = sink
                    while curr1 != source:
                        prev = parent[curr1]
                        graph[prev][curr1]["flow"] += new_flow
                        graph[curr1][prev]["flow"] -= new_flow
                        curr1 = prev
                    path.append((nx.get_edge_attributes(graph, "flow"), f"Step {current_step}: Updated Path"))
                    current_step += 1
                    return new_flow
                queue.append((next, new_flow))
    return 0


# Function to calculate the maximum flow using Edmonds-Karp algorithm (BFS based)
def edmonds_karp(graph: nx.DiGraph, source, sink):
    # make sure to also update the global variables current_step and path
    global current_step
    global path
    global temp
    max_flow = 0
    parent = [-1] * len(graph.nodes)
    path.append((nx.get_edge_attributes(graph, "flow"), "Step 0: Start"))
    path.pop(0)
    current_step += 1

    while (new_flow := bfs(graph, source, sink, parent)):
        max_flow += new_flow
        curr = sink
        while curr != source:
            prev = parent[curr]
            graph[prev][curr]["residual_capacity"] -= new_flow
            graph[curr][prev]["residual_capacity"] += new_flow
            curr = prev
        path.append((nx.get_edge_attributes(graph, "flow"), f"Step {current_step}: Augmenting path found"))
        current_step += 1

    return max_flow


# Initializes Matplotlib animation
def init():
    ax.clear()  # Clear previous frame
    ax.set_xticks([])
    ax.set_yticks([])


# Generates next frame of Matplotlib animation
#  - num: frame number being generated
def update(num):

    # Clear previous frame
    ax.clear()

    # Set title according to corresponding path tuple
    ax.set_title(path[num][1], fontweight="bold")
    labels = {}
    for node in graph.nodes():
        labels[node] = node
    labels[0] = "S"
    labels[len(graph.nodes) - 1] = "T"

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        nodelist=graph.nodes(),
        node_color="lightgray",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels=labels,
        font_color="black",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[0],
        node_color="green",
        edgecolors="black",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[len(graph.nodes) - 1],
        node_color="red",
        edgecolors="black",
    )

    non_zero_capacity_edges = [
        (u, v) for u, v, d in graph.edges(data=True) if d["capacity"] != 0
    ]
    curved_edges = [
        (v, u) for u, v in non_zero_capacity_edges if (v, u) in non_zero_capacity_edges
    ]

    # Get list of edges with reversed counterpart
    arc_rad = 0.2
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edgelist=curved_edges,
        connectionstyle=f"arc3, rad = {arc_rad}",
        edge_color="gray",
    )

    # Get list of edges without reversed counterpart
    straight_edges = list(set(non_zero_capacity_edges) - set(curved_edges))

    nx.draw_networkx_edges(
        graph, pos=pos, ax=ax, edgelist=straight_edges, edge_color="gray"
    )
    edge_capacities = nx.get_edge_attributes(graph, "capacity")
    edge_flows = path[num][0]

    # Create list of labels for curved edges
    curved_edge_labels = {
        edge: f"{edge_flows[edge]} / {edge_capacities[edge]}" for edge in curved_edges
    }

    # Use custom function to draw curved labels
    my_nx.my_draw_networkx_edge_labels(
        graph,
        pos=pos,
        ax=ax,
        edge_labels=curved_edge_labels,
        rotate=True,
        rad=arc_rad,
        font_color="gray",
    )

    # Create list of labels for straight edges

    straight_edge_labels = {
        edge: f"{edge_flows[edge]} / {edge_capacities[edge]}" for edge in straight_edges
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        ax=ax,
        edge_labels=straight_edge_labels,
        rotate=True,
        font_color="gray",
    )


# Run the main function as an entry point if this program is the top level program executed
if __name__ == "__main__":
    # Exit and print error message if number of arguments provided != 3
    if len(sys.argv) != 2:
        sys.exit(
            "Error: not enough command-line arguments\nUsage: python main.py [capacity matrix]"
        )

    # Capacity matrix is read from second runtime argument
    cap = np.loadtxt(sys.argv[1])

    # Get number of rows and columns (should be the same)
    (cap_rows, cap_cols) = cap.shape

    # Exit and print error message if adjacency matrix and capacity matrix are not square
    if cap_rows != cap_cols:
        sys.exit("Error: matrices must be square")

    # Set node positions for graph visualization
    graph = create_graph(cap)
    pos = nx.spring_layout(graph)

    source = 0
    sink = len(graph.nodes) - 1
    print(f"Max Flow Check: {nx.maximum_flow_value(graph, source, sink)}")
    max_flow_value = edmonds_karp(graph, source, sink)
    print(f"\nMax Flow Ours:  {max_flow_value}")

    # Create Matplotlib animation
    fig, ax = plt.subplots()
    ani = matplotlib.animation.FuncAnimation(
        fig,
        update,
        frames=range(len(path)),
        init_func=init,
        interval=2000,
        repeat=False,
    )
    plt.show()
