import sys
import random
import numpy as np

if len(sys.argv) != 3:
    sys.exit('Error: wrong number of command-line arguments\nUsage: python generate_graph.py [number of nodes] [name of files]\n')
else:
    n = int(sys.argv[1])
    flow_path = f'test/{sys.argv[2]}.flow'
    cap_path = f'test/{sys.argv[2]}.cap'

    cap_matrix = np.random.randint(0, 100, size=(n, n))
    cap_matrix[n - 1] = np.zeros(n)
    cap_matrix[:, 0] = np.zeros(n)
    for i in range(n):
        for j in range(i + 1):
            if not bool(random.getrandbits(1)) or i == j:
                cap_matrix[i, j] = 0
                cap_matrix[j, i] = 0

    flow_matrix = np.empty((n, n))
    for i in range(cap_matrix.shape[0]):
        for j in range(cap_matrix.shape[1]):
            flow_matrix[i, j] = int(cap_matrix[i, j] * random.uniform(0, 1))

    with open(flow_path, "w") as file:
        for i in range(n):
            content = np.array2string(flow_matrix[i], precision=2, separator=' ', suppress_small=True)
            file.write(content.replace('\n','').replace('[','').replace(']','').replace('.','').replace('  ',' ').strip() + '\n')
    with open(cap_path, "w") as file:
        for i in range(n):
            content = np.array2string(cap_matrix[i], precision=2, separator=' ', suppress_small=True)
            file.write(content.replace('\n','').replace('[','').replace(']','').replace('.','').replace('  ',' ').strip() + '\n')
