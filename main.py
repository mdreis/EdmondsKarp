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
import numpy as np
import networkx as nx

if __name__ == '__main__':
    adj = np.loadtxt(sys.argv[1]) # Read adjacency matrix into NumPy matrix
    (adj_rows, adj_cols) = adj.shape # Get number of rows and columns (should be the same)
    graph = nx.Graph()
    graph.add_nodes_from(range(adj_rows)) # Add nodes
    for x in range(adj_rows):
        for y in range(adj_cols):
            if adj[x][y] != 0:
                graph.add_edge(x, y, weight=adj[x][y]) # Add edges and weights
    nx.draw(graph)
