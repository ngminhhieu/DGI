import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn

from models.dgi.dgi import DGI
from models.dgi.reg import MultipleRegression
from utils.dgi import process
import sys

def PrepareData(pm_dataset):
      # pm_data = np.load('./log/dgi/trained/embeds.npz')['embeds']
    pm_dataset = pm_dataset.replace("**", 0)
    pm_dataset = pm_dataset.to_numpy()
    pm_data = pm_dataset[:, 4:]
    pm_data = pm_data.astype(np.float)
    return pm_data

class ConfigDGI:
    
    def __init__(self, is_training=True, **kwargs):
        pm_dataset = pd.read_csv('./data/dgi/pm.csv')
        self.pm_data = PrepareData(pm_dataset)
        self.adj = process.build_graph('./data/dgi/locations.csv')
        idx_train, idx_val, idx_test = process.train_valid_test(self.pm_data, 0.6, 0.2)
        
        self.adj = process.normalize_adj(self.adj + sp.eye(self.adj.shape[0]))
        self.sparse = True
        if self.sparse:
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(self.adj)
        else:
            self.adj = (self.adj + sp.eye(self.adj.shape[0])).todense()
        
        if not self.sparse:
            self.adj = torch.FloatTensor(self.adj[np.newaxis])
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test = torch.LongTensor(idx_test)
        self._kwargs = kwargs
        self._model_kwargs = kwargs.get('model')
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self.dataset = self._data_kwargs.get('dataset')
        self.log_dgi = kwargs.get('log_dgi')
        # training params
        self.batch_size = self._data_kwargs.get('batch_size')
        self.nb_epochs = self._train_kwargs.get('nb_epochs')
        self.patience = self._train_kwargs.get('patience')
        self.lr = self._model_kwargs.get('lr')
        self.l2_coef = self._model_kwargs.get('l2_coef')
        self.drop_prob = self._model_kwargs.get('drop_prob')
        self.hid_units = self._model_kwargs.get('hid_units')
        self.sparse = self._model_kwargs.get('sparse')
        self.nonlinearity = self._model_kwargs.get('nonlinearity') # special name to separate parameters
    
    def train(self):
        mae_loss = nn.L1Loss()
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        embeds = np.empty(shape=(len(self.pm_data), self.pm_data.shape[1], self.hid_units))
        labels = np.empty(shape=(len(self.pm_data), self.pm_data.shape[1], 1))

        for i in range(len(self.pm_data)):
            features, label  = process.load_data_pm(self.pm_data)
            features, _ = process.preprocess_features(features)
            # so tram PM2.5
            nb_nodes = features.shape[0]
            # so features tuong ung voi moi tram
            ft_size = features.shape[1]
            features = torch.FloatTensor(features[np.newaxis])
            label = torch.FloatTensor(label[np.newaxis])
            labels[i, :, :] = label
            model = DGI(ft_size, self.hid_units, self.nonlinearity)
            optimiser = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_coef)
            if torch.cuda.is_available():
              model.cuda()
              features = features.cuda()
              if self.sparse:
                  self.sp_adj = self.sp_adj.cuda()
              else:
                  self.adj = self.adj.cuda()
              self.idx_train = self.idx_train.cuda()
              self.idx_val = self.idx_val.cuda()
              self.idx_test = self.idx_test.cuda()


            for epoch in range(self.nb_epochs):
                model.train()
                optimiser.zero_grad()

                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]

                lbl_1 = torch.ones(self.batch_size, nb_nodes)
                lbl_2 = torch.zeros(self.batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.cuda()
                    lbl = lbl.cuda()
                
                # forward
                logits = model(features, shuf_fts, self.sp_adj if self.sparse else self.adj, self.sparse, None, None, None)

                # hàm loss phải đạo hàm được nếu muốn tự config
                loss = b_xent(logits, lbl)

                print('Loss:', loss)

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), self.log_dgi+'/best_dgi.pkl')
                else:
                    cnt_wait += 1

                if cnt_wait == self.patience:
                    print('Early stopping!')
                    break

                # calculate gradient
                loss.backward()
                # update weight
                optimiser.step()

            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load(self.log_dgi+'/best_dgi.pkl'))
            embeds[i, :, :] = model.embed(features, self.sp_adj if self.sparse else self.adj, self.sparse, None)[0].cpu()

        np.savez(self.log_dgi+'/embeds3d.npz', embeds3d = embeds)
        embeds = torch.FloatTensor(embeds)
        labels = torch.FloatTensor(labels)

        # train_embs = embeds[idx_train, :, :]
        # val_embs = embeds[idx_val, :, :]
        # test_embs = embeds[idx_test, :, :]
        # train_lbls = labels[idx_train, :, :]
        # val_lbls = labels[idx_val, :, :]
        # test_lbls = labels[idx_test, :, :]

    # def test(self):
        cost = 0
        nb_testings = 1
        for _ in range(nb_testings):
            multReg = MultipleRegression(embeds.shape[2])
            opt = torch.optim.Adam(multReg.parameters(), lr=0.01, weight_decay=0.0)
            # multReg.cuda()

            for _ in range(100):
                multReg.train()
                opt.zero_grad()

                # logits = multReg(train_embs)
                # loss = mae_loss(logits, train_lbls)
                logits = multReg(embeds)
                loss = mae_loss(logits, labels)
                loss.backward()
                opt.step()

            # preds = multReg(test_embs)
            # cost += abs(preds - test_lbls)/test_lbls
            preds = multReg(embeds)
            cost += abs(preds - labels)/labels

        ebs = preds.detach().numpy()
        ebs = np.squeeze(ebs)
        np.savez(self.log_dgi+'/embeds.npz', embeds = ebs)
