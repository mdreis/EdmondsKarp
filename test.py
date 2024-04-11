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

import sys
import matplotlib.animation
import matplotlib.pyplot as plt
import my_networkx as my_nx
import networkx as nx
import numpy as np

current_step = 1
path = [(0, 'Step 0: Start')] # List of tuples (node, title), necessary for generating animation

def bfs(source, sink):
    global current_step
    global path
    path.append((1, f'Step {current_step}: Start'))
    current_step += 1

    parent = [-1] * graph.number_of_nodes()  # Initialize parent list with -1
    parent[source] = -2  # Set parent of source node to -2
    q = [(source, float('inf'))]  # Initialize queue with source node and infinite flow

    while q:
        cur, flow = q.pop(0)  # Get current node and flow from front of queue
        path.append((cur, f'Step {current_step}: Current node is {cur} with flow {flow}'))  # Append current node to path
        current_step += 1

        for next in graph.neighbors(cur):  # Iterate through neighbors of current node
            if parent[next] == -1 and graph.edges[cur, next]['capacity'] - graph.edges[cur, next]['flow'] > 0:  # If next node has not been visited and there is available capacity
                parent[next] = cur  # Set parent of next node to current node
                new_flow = min(flow, graph.edges[cur, next]['capacity'] - graph.edges[cur, next]['flow'])  # Calculate new flow
                if next == sink:  # If next node is sink
                    path.append((next, f'Step {current_step}: Found sink'))  # Append sink to path
                    current_step += 1
                    return new_flow  # Return new flow
                q.append((next, new_flow))  # Append next node and new flow to queue

    return 0


def max_flow(source, sink):
    global current_step
    global path
    flow = 0

    while new_flow := bfs(source, sink):  # While there is a new flow
        flow += new_flow  # Add new flow to total flow
        cur = sink  # Set current node to sink
        while cur != source:  # While current node is not source
            prev = parent = graph.edges[cur, parent]['source']  # Get parent of current node
            graph.edges[prev, cur]['flow'] += new_flow  # Add new flow to flow of edge from parent to current node
            graph.edges[cur, prev]['flow'] -= new_flow  # Subtract new flow from flow of edge from current to parent node
            cur = prev  # Set current node to parent node

    return flow

# Initializes Matplotlib animation
def init():
    ax.clear() # Clear previous frame
    ax.set_xticks([])
    ax.set_yticks([])

# Generates next frame of Matplotlib animation
#   - num: frame number being generated
def update(num):
    ax.clear() # Clear previous frame
    ax.set_title(path[num][1], fontweight="bold") # Set title according to corresponding path tuple

    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(), node_color="white", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(graph, pos=pos, labels=dict(zip(graph.nodes(), graph.nodes())), font_color="black")

    curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()] # Get list of edges with reversed counterpart
    arc_rad = 0.1
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}', edge_color="gray") 
    straight_edges = list(set(graph.edges()) - set(curved_edges)) # Get list of edges without reversed counterpart
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, edgelist=straight_edges, edge_color="gray")

    edge_flows = nx.get_edge_attributes(graph, 'flow')
    edge_capacities = nx.get_edge_attributes(graph, 'capacity')
    curved_edge_labels = {edge: f'{edge_flows[edge]} / {edge_capacities[edge]}' for edge in curved_edges} # Create list of labels for curved edges
    my_nx.my_draw_networkx_edge_labels(graph, pos=pos, ax=ax, edge_labels=curved_edge_labels, rotate=True, rad=arc_rad, font_color="gray") # Use custom function to draw curved labels
    straight_edge_labels = {edge: f'{edge_flows[edge]} / {edge_capacities[edge]}' for edge in straight_edges} # Create list of labels for straight edges
    nx.draw_networkx_edge_labels(graph, pos=pos, ax=ax, edge_labels=straight_edge_labels, rotate=True, font_color="gray")

if __name__ == '__main__':
    if len(sys.argv) != 3: # Exit and print error message if number of arguments provided != 3
        sys.exit('Error: not enough command-line arguments\nUsage: python main.py [flow matrix] [capacity matrix]')

    graph = nx.DiGraph()
    adj = np.loadtxt(sys.argv[1]) # Adjacency matrix is read from first runtime argument
    cap = np.loadtxt(sys.argv[2]) # Capacity matrix is read from second runtime argument
    (adj_rows, adj_cols) = adj.shape # Get number of rows and columns (should be the same)

    if adj.shape != cap.shape: # Exit and print error message if adjacency matrix and capacity matrix are not the same dimensions
        sys.exit('Error: flow matrix and capacity matrix must be the same size')
    elif adj_rows != adj_cols: # Exit and print error message if adjacency matrix and capacity matrix are not square
        sys.exit('Error: matrices must be square')

    graph.add_nodes_from(range(adj_rows)) # Add nodes
    for x in range(adj_rows):
        for y in range(adj_cols):
            if adj[x][y] != 0:
                graph.add_edge(x, y, flow=int(adj[x][y]), capacity=int(cap[x][y])) # Add edges, flows, and capacities
    pos = nx.planar_layout(graph) # Planar layout = minimized edge overlap

    max_flow(0, 5) # Calculate max flow from source 0 to sink 5

    fig, ax = plt.subplots(figsize=(6, 4)) # Build plot
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(path), init_func=init, interval=1000, repeat=True) # Generate animation
    plt.show()


