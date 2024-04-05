## Current progress:
Preliminary work done on animating graph using the update function

### References
- NetworkX documentation: https://networkx.org/documentation/latest/reference/index.html
- Animating NetworkX: https://stackoverflow.com/questions/43646550/how-to-use-an-update-function-to-animate-a-networkx-graph-in-matplotlib-2-0-0
- Drawing multiple edges between 2 nodes with NetworkX: https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
- Matplotlib.animation documentation: https://matplotlib.org/stable/api/animation_api.html

## To run
```
python main.py matrix.flow matrix.cap
```
matrix.flow files contain an adjacency matrix with weights representing the edge's current flow \
matrix.cap files contain a matching adjacency matrix with weights representing the edge's capacity

### Dependencies
```
pip install matplotlib numpy networkx
```
