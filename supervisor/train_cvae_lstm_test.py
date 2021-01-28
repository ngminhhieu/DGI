from models.cvaelstm.cvae_lstm import VRAE
import numpy as np
import torch
import pandas as pd
import plotly
import sys
from torch.utils.data import DataLoader, TensorDataset
from utils.cvaelstm.process import train_val_test_split, getData, standardizeData
import matplotlib.pyplot as plt

# plotly.offline.init_notebook_mode()

pm_dataset = pd.read_csv('./data/pm.csv')
pm_dataset = pm_dataset.replace("**", 0)
pm_dataset = pm_dataset.to_numpy()
pm_data = pm_dataset[:, 4:]
pm_data = pm_data.astype(np.float)

# Perform the train validation split
train_data, val_data, test_data = train_val_test_split(pm_data, valid_size=0.2, test_size=0.2)

# Standardize the data to bring the inputs on a uniform scale
normalized_train, sc = standardizeData(train_data, train = True)
normalized_val, _ = standardizeData(val_data, sc)
normalized_test, _ = standardizeData(test_data, sc)

sequence_length = 14
number_of_features = train_data.shape[1] - 1
horizon = 1

trainX, trainY = getData(normalized_train, sequence_length, horizon)
valX, valY = getData(normalized_val, sequence_length, horizon)
valX_0 = valX
testX, testY = getData(normalized_test, sequence_length, horizon)
trainY = trainY[:, 0, :]
valY = valY[:, 0, :]
testY = testY[:, 0, :]
# trainX = TensorDataset(torch.from_numpy(trainX))
# trainY = TensorDataset(torch.from_numpy(trainY))
# valX = TensorDataset(torch.from_numpy(valX))
# valY = TensorDataset(torch.from_numpy(valY))
# testX = TensorDataset(torch.from_numpy(testX))
# testY = TensorDataset(torch.from_numpy(testY))

print(valX.shape)
# valX = valX.reshape(-1, valX.shape[-1])
plt.plot(valX[:100, -1, 1], label='groundtruth')
plt.show()