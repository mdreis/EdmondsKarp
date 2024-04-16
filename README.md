## Current progress:
styling animation

### References
- NetworkX documentation: https://networkx.org/documentation/latest/reference/index.html
- Animating NetworkX: https://stackoverflow.com/questions/43646550/how-to-use-an-update-function-to-animate-a-networkx-graph-in-matplotlib-2-0-0
- Drawing multiple edges between 2 nodes with NetworkX: https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
- Matplotlib.animation documentation: https://matplotlib.org/stable/api/animation_api.html
- Edmonds-Karps Pseudocode: https://cp-algorithms.com/graph/edmonds_karp.html#:~:text=Edmonds%2DKarp%20algorithm%20is%20just,independently%20of%20the%20maximal%20flow.



## To run
```
python main.py [capacity matrix]
```
Capacity matrix files contain an adjacency matrix with weights representing the edge's flow capacity

## Generating graphs
Graphs can be generated using the included graph generator script:
```
python generate_graph.py [number of nodes] [name of file]
```
Created graphs are placed in the test subfolder. Example usage of the graph generator script to generate a graph with 12 nodes:
```
python generate_graph.py 12 Size12
```

### Dependencies
```
pip install matplotlib numpy networkx
```
