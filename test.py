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


pm_dataset = pd.read_csv('./data/pm.csv')
pm_dataset = pm_dataset.replace("**", 0)
pm_dataset = pm_dataset.to_numpy()

# Perform the train validation split
train_data, val_data, test_data = train_val_test_split(pm_dataset, train_pct, valid_pct)

# Standardize the data to bring the inputs on a uniform scale
normalized_train, sc = standardizeData(train_data, train = True)
normalized_val, _ = standardizeData(val_data, sc)
normalized_test, _ = standardizeData(test_data, sc)

trainX, trainY = getData(normalized_train, sequence_length, horizon, output_dim)
valX, valY = getData(normalized_val, sequence_length, horizon, output_dim)
testX, testY = getData(normalized_test, sequence_length, horizon, output_dim)
trainY = trainY[:, 0, :]
valY = valY[:, 0, :]
testY = testY[:, 0, :]
trainX = torch.FloatTensor(trainX)
trainY = torch.FloatTensor(trainY)
valX = torch.FloatTensor(valX)
valY = torch.FloatTensor(valY)
testX = torch.FloatTensor(testX)
testY = torch.FloatTensor(testY)
