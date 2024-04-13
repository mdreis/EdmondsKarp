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

def create_graph(cap_matrix, flow_matrix):
    n = len(cap_matrix)
    graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if cap_matrix[i][j] > 0:
                graph.add_edge(i, j, capacity=cap_matrix[i][j], flow=flow_matrix[i][j])
                if not graph.has_edge(j, i):
                    graph.add_edge(j, i, capacity=0, flow=0)
    return graph

def bfs(graph, source, sink, parent):
    visited = [False] * len(graph.nodes)
    queue = [source]
    visited[source] = True
    
    while queue:
        u = queue.pop(0)
        
        for v in graph.neighbors(u):
            residual_capacity = graph[u][v]['capacity'] - graph[u][v]['flow']
            if not visited[v] and residual_capacity > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False

# Function to calculate the maximum flow using Edmonds-Karp algorithm (BFS based)
def edmonds_karp(graph, source, sink):
    #make sure to also update the global variables current_step and path
    global current_step
    global path
    max_flow = 0
    parent = [-1] * len(graph.nodes)

    while bfs(graph, source, sink, parent):
        path.append((sink, f'Step {current_step}: Found sink'))
        current_step += 1
        path.append((sink, f'Step {current_step}: Augmenting path found'))
        current_step += 1

        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s]['capacity'] - graph[parent[s]][s]['flow'])
            s = parent[s]

        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v]['flow'] += path_flow
            graph[v][u]['flow'] -= path_flow
            v = u

    return max_flow

    
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

    graph = create_graph(cap, adj)
    pos = nx.spring_layout(graph) # Set node positions for graph visualization

    source = 0
    sink = len(graph.nodes) - 1
    max_flow_value = edmonds_karp(graph, source, sink)

    # Create Matplotlib animation
    fig, ax = plt.subplots()
    ani = matplotlib.animation.FuncAnimation(fig, update, frames=range(len(path)), init_func=init, repeat=False)
    plt.show()





    #print to terminal
    print("Flow matrix:")
    print(adj)
    print("Capacity matrix:")
    print(cap)
    print("Max flow:")
    print(max_flow_value)
