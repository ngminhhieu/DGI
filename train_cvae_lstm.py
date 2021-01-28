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

# pm_data = np.load('./log/dgi/trained/embeds.npz')['embeds']
pm_dataset = pd.read_csv('./data/cvae_lstm/pm.csv')
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
trainX = TensorDataset(torch.from_numpy(trainX))
trainY = TensorDataset(torch.from_numpy(trainY))
valX = TensorDataset(torch.from_numpy(valX))
valY = TensorDataset(torch.from_numpy(valY))
testX = TensorDataset(torch.from_numpy(testX))
testY = TensorDataset(torch.from_numpy(testY))

dload = './log/cvae_lstm'
hidden_size = 512
hidden_layer_depth = 1
latent_length = 20
batch_size = 256
learning_rate = 0.0001
n_epochs = 500
dropout_rate = 0
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
patience = 100
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options: LSTM, GRU
conditional = False

vrae = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            patience = patience,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer, 
            cuda = cuda,
            print_every=print_every, 
            clip=clip, 
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload,
            conditional = conditional)

vrae.fit(trainX)
vrae.load('./log/cvae_lstm/best_cvae_lstm.pkl')
z_run = vrae.reconstruct(valX)
z_run = np.swapaxes(z_run,0,1)
valX_0 = valX_0[:z_run.shape[0]]
z_run = z_run.reshape(-1, z_run.shape[-1])
valX_0 = valX_0.reshape(-1, valX_0.shape[-1])
plt.plot(z_run[:200, 1], label='generation')
plt.plot(valX_0[:200, 1], label='groundtruth')
plt.savefig(dload + '/cvae_lstm.png')
plt.legend()
plt.close()