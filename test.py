import os
import networkx as nx
import pandas as pd
import random
from collections import defaultdict
data_dir = os.path.expanduser("./data/")

# edgelist = pd.read_csv(os.path.join(data_dir, "cora.cites"), sep='\t', header=None, names=["target", "source"])
# node_data = pd.read_csv(os.path.join(data_dir, "cora.content"), sep='\t', header=None)

pm_dataset = pd.read_csv('./data/pm.csv').to_numpy()
print(pm_dataset.shape[1]-4)

graph = defaultdict(list)

gauges = pm_dataset.shape[1] - 4

for i in range(gauges):
    source = []
    for j in range(gauges):
        ran = random.random()
        if ran < 0.1:
            source.append(j)
    graph[i] = source

print(type(graph))
print(graph) 
