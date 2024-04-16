###############################################################################
# File: main.py                                                                #
# Authors: Jonathan Williams !!! ADD OUR NAMES HERE !!!                        #
# -----                                                                        #
# !!!! ADD DESCRIPTION OF FILE !!!!!
# -----                                                                        #
# Last Modified: Monday, April 15th 2024 17:12:55                              #
# Modified By: Jonathan Williams                                               #
###############################################################################

import sys
import matplotlib.animation
import matplotlib.pyplot as plt
import my_networkx as my_nx
import networkx as nx
import numpy as np

temp = 0

current_step = 1

# List of tuples (node, title), necessary for generating animation
path = [(0, "Step 0: Start")]


def create_graph(cap_matrix):
    n = len(cap_matrix)
    graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if cap_matrix[i][j] > 0:
                graph.add_edge(i, j, capacity=cap_matrix[i][j], flow=0)
                if not graph.has_edge(j, i):
                    graph.add_edge(j, i, capacity=0, flow=0)
    return graph


def bfs(graph: nx.DiGraph, source, sink, parent):
    visited = [False] * len(graph.nodes)
    queue = [source]
    visited[source] = True

    while queue:
        u = queue.pop(0)
        for v in graph.neighbors(u):
            residual_capacity = graph[u][v]["capacity"] - graph[u][v]["flow"]
            if not visited[v] and residual_capacity > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False


# Function to calculate the maximum flow using Edmonds-Karp algorithm (BFS based)
def edmonds_karp(graph: nx.DiGraph, source, sink):
    # make sure to also update the global variables current_step and path
    global current_step
    global path
    global temp
    max_flow = 0
    parent = [-1] * len(graph.nodes)

    while bfs(graph, source, sink, parent):
        path.append((sink, f"Step {current_step}: Found sink"))
        current_step += 1
        path.append((sink, f"Step {current_step}: Augmenting path found"))
        current_step += 1

        path_flow = float("inf")
        s = sink
        while s != source:
            path_flow = min(
                path_flow, graph[parent[s]][s]["capacity"] - graph[parent[s]][s]["flow"]
            )
            s = parent[s]

        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v]["flow"] += path_flow
            graph[v][u]["flow"] -= path_flow
            v = u
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
        if node != 0 or node != len(graph.nodes) - 1:
            labels[node] = node
        labels[0] = "Source"
        labels[len(graph.nodes) - 1] = "Sink"

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        nodelist=graph.nodes(),
        node_color="white",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos,
        labels=labels,
        font_color="black",
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

    edge_flows = nx.get_edge_attributes(graph, "flow")
    edge_capacities = nx.get_edge_attributes(graph, "capacity")

    # Create list of labels for curved edges
    if path[num][1] == "Step 0: Start":
        curved_edge_labels = {
        edge: f"0 / {edge_capacities[edge]}" for edge in curved_edges
        }
    else:
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
    if path[num][1] == "Step 0: Start":
        straight_edge_labels = {
        edge: f" 0 / {edge_capacities[edge]}" for edge in straight_edges
        }
    else:
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
    max_flow_value = edmonds_karp(graph, source, sink)
    print(f"\nMax Flow Ours:  {max_flow_value}")
    print(f"Max Flow Check: {nx.maximum_flow_value(graph, source, sink)}")

    # Create Matplotlib animation
    fig, ax = plt.subplots()
    ani = matplotlib.animation.FuncAnimation(
        fig,
        update,
        frames=range(len(path)),
        init_func=init,
        interval=5000,
        repeat=False,
    )
    plt.show()

# c++ implementation from

# https://cp-algorithms.com/graph/edmonds_karp.html#:~:text=Edmonds%2DKarp%20algorithm%20is%20just,independently%20of%20the%20maximal%20flow.

# int n;
# vector<vector<int>> capacity;
# vector<vector<int>> adj;

# int bfs(int s, int t, vector<int>& parent) {
#     fill(parent.begin(), parent.end(), -1);
#     parent[s] = -2;
#     queue<pair<int, int>> q;
#     q.push({s, INF});

#     while (!q.empty()) {
#         int cur = q.front().first;
#         int flow = q.front().second;
#         q.pop();

#         for (int next : adj[cur]) {
#             if (parent[next] == -1 && capacity[cur][next]) {
#                 parent[next] = cur;
#                 int new_flow = min(flow, capacity[cur][next]);
#                 if (next == t)
#                     return new_flow;
#                 q.push({next, new_flow});
#             }
#         }
#     }

#     return 0;
# }

# int maxflow(int s, int t) {
#     int flow = 0;
#     vector<int> parent(n);
#     int new_flow;

#     while (new_flow = bfs(s, t, parent)) {
#         flow += new_flow;
#         int cur = t;
#         while (cur != s) {
#             int prev = parent[cur];
#             capacity[prev][cur] -= new_flow;
#             capacity[cur][prev] += new_flow;
#             cur = prev;
#         }
#     }

#     return flow;
# }
