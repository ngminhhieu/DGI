import os
import networkx as nx
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from scipy import sparse
data_dir = os.path.expanduser("./data/")

# edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
# node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None)

pm_dataset = pd.read_csv('./data/pm.csv')
pm_dataset = pm_dataset.replace("**", 0)
pm_dataset = pm_dataset.to_numpy()
pm_data = pm_dataset[:, 4:]
pm_data = pm_data.astype(np.float)
gauges = pm_data.shape[1]
graph = defaultdict(list)
features = np.empty(shape=(gauges, 1))
for i in range(gauges):
    features[i, 0] = pm_data[-1, i]

features = sparse.csr_matrix(features)

for i in range(gauges):
    source = []
    for j in range(gauges):
        ran = random.random()
        if ran < 0.1:
            source.append(j)
    graph[i] = source


