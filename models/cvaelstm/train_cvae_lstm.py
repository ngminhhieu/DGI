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
def PrepareData(pm_dataset):
      # pm_data = np.load('./log/dgi/trained/embeds.npz')['embeds']
      pm_dataset = pm_dataset.replace("**", 0)
      pm_dataset = pm_dataset.to_numpy()
      pm_data = pm_dataset[:, 4:10].copy()
      pm_data = pm_data.astype(np.float)
      return pm_data



class ConfigCvaeLstm:
      def __init__(self,is_training=True, **kwargs):
            # take the dataset 
            dataset = pd.read_csv('./data/cvae_lstm/pm.csv')
            self.pm_data = PrepareData(dataset)

            # Perform the train validation split
            train_data, val_data, test_data = train_val_test_split(self.pm_data, valid_size=0.2, test_size=0.2)


            # Standardize the data to bring the inputs on a uniform scale
            normalized_train, sc = standardizeData(train_data, train = True)
            normalized_val, _ = standardizeData(val_data, sc)
            normalized_test, _ = standardizeData(test_data, sc)

            self.sequence_length = 14
            self.number_of_features = train_data.shape[1] - 1
            horizon = 1

            trainX, trainY = getData(normalized_train, self.sequence_length, horizon)
            valX, valY = getData(normalized_val, self.sequence_length, horizon)
            testX, testY = getData(normalized_test, self.sequence_length, horizon)
            trainX = trainX[:, :-1, :]
            valX = valX[:, -1, :]
            testX = testX[:, -1, :]
            trainY = trainY[:, -1, :]
            valY = valY[:, -1, :]
            testY = testY[:, -1, :]
            self.trainX = TensorDataset(torch.from_numpy(trainX))
            self.trainY = TensorDataset(torch.from_numpy(trainY))
            self.valX = TensorDataset(torch.from_numpy(valX))
            self.valY = TensorDataset(torch.from_numpy(valY))
            self.testX = TensorDataset(torch.from_numpy(testX))
            self.testY = TensorDataset(torch.from_numpy(testY))
            self._kwargs = kwargs
            self._model_kwargs = kwargs.get('model')
            self._data_kwargs = kwargs.get('data')
            self._train_kwargs = kwargs.get('train')
            self.dload = './log/cvae_lstm'
            self.hidden_size= self._model_kwargs.get('hidden_size')
            self.hidden_layer_depth = self._model_kwargs.get('hidden_layer_depth')
            self.latent_length = self._model_kwargs.get('latent_length')
            self.batch_size = self._data_kwargs.get('batch_size')
            self.learning_rate = self._model_kwargs.get('learning_rate')
            self.n_epochs = self._train_kwargs.get('n_epochs')
            self.dropout_rate = self._train_kwargs.get('dropout_rate')
            self.optimizer = self._train_kwargs.get('optimizer') # options: ADAM, SGD
            self.cuda = self._model_kwargs.get('cuda') # options: True, False
            self.print_every= self._model_kwargs.get('print_every')
            self.clip = self._model_kwargs.get('clip') # options: True, False
            self.max_grad_norm= self._model_kwargs.get('max_grad_norm')
            self.patience = self._train_kwargs.get('patience')
            self.loss = self._model_kwargs.get('loss') # options: SmoothL1Loss, MSELoss
            self.block = self._model_kwargs.get('block') # options: LSTM, GRU
            self.conditional = self._model_kwargs.get('conditional')

      def train(self):
            location = pd.read_csv('./data/cvae_lstm/locations.csv').to_numpy()
            location_lat_train = location[1:7]
            location_lat_test = location[7]
            vrae = VRAE(sequence_length=self.sequence_length,
                        condition = location_lat_train,
                        number_of_features = self.number_of_features,
                        patience = self.patience,
                        hidden_size = self.hidden_size, 
                        hidden_layer_depth = self.hidden_layer_depth,
                        latent_length = self.latent_length,
                        batch_size = self.batch_size,
                        learning_rate = self.learning_rate,
                        n_epochs = self.n_epochs,
                        dropout_rate = self.dropout_rate,
                        optimizer = self.optimizer, 
                        cuda = self.cuda,
                        print_every= self.print_every, 
                        clip= self.clip, 
                        max_grad_norm= self.max_grad_norm,
                        loss = self.loss,
                        block = self.block,
                        dload = self.dload,
                        conditional = self.conditional)
            
            vrae.fit(self.trainX)
            vrae.load('./log/cvae_lstm/best_cvae_lstm.pkl')
            z_run = vrae.reconstruct(self.valX, condition=location_lat_test)
            z_run = np.swapaxes(z_run,0,1)
            valY = self.valY[:z_run.shape[0]]
            z_run = z_run.reshape(-1, z_run.shape[-1])
            valY = valY.reshape(-1, valY.shape[-1])
            plt.plot(z_run[:200, 1], label='generation')
            plt.plot(valY[:200, 1], label='groundtruth')
            plt.savefig(self.dload + '/cvae_lstm.png')
            plt.legend()
            plt.close()
