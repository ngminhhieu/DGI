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

# take the dataset 
pm_dataset = pd.read_csv('./data/dgi/pm.csv')
pm_data = PrepareData(pm_dataset)

adj = process.build_graph('./data/dgi/locations.csv')
idx_train, idx_val, idx_test = process.train_valid_test(pm_data, 0.6, 0.2)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

class ConfigDGI:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._model_kwargs = kwargs.get('model')
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        dataset = self._data_kwargs.get('dataset')
        log_dgi = kwargs.get('log_dgi')
        # training params
        batch_size = self._data_kwargs.get('batch_size')
        nb_epochs = self._train_kwargs.get('nb_epochs')
        patience = self._train_kwargs.get('patience')
        lr = self._model_kwargs.get('lr')
        l2_coef = self._model_kwargs.get('l2_coef')
        drop_prob = self._model_kwargs.get('drop_prob')
        hid_units = self._model_kwargs.get('hid_units')
        sparse = self._model_kwargs.get('sparse')
        nonlinearity = self._model_kwargs.get('nonlinearity') # special name to separate parameters

        if torch.cuda.is_available():
            print('Using CUDA')
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        mae_loss = nn.L1Loss()
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        embeds = np.empty(shape=(len(pm_data), pm_data.shape[1], hid_units))
        labels = np.empty(shape=(len(pm_data), pm_data.shape[1], 1))
        for i in range(len(pm_data)):
            features, label  = process.load_data_pm(pm_data)
            features, _ = process.preprocess_features(features)
            # so tram PM2.5
            nb_nodes = features.shape[0]
            # so features tuong ung voi moi tram
            ft_size = features.shape[1]
            features = torch.FloatTensor(features[np.newaxis])
            label = torch.FloatTensor(label[np.newaxis])
            labels[i, :, :] = label
            model = DGI(ft_size, hid_units, nonlinearity)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
            if torch.cuda.is_available():
                features = features.cuda()

            for epoch in range(nb_epochs):
                model.train()
                optimiser.zero_grad()

                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]

                lbl_1 = torch.ones(batch_size, nb_nodes)
                lbl_2 = torch.zeros(batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.cuda()
                    lbl = lbl.cuda()
                
                # forward
                logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

                # hàm loss phải đạo hàm được nếu muốn tự config
                loss = b_xent(logits, lbl)

                print('Loss:', loss)

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), log_dgi+'/best_dgi.pkl')
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    break

                # calculate gradient
                loss.backward()
                # update weight
                optimiser.step()

            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load(log_dgi+'/best_dgi.pkl'))
            embeds[i, :, :], _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

        np.savez(log_dgi+'/embeds3d.npz', embeds3d = embeds)
        embeds = torch.FloatTensor(embeds)
        labels = torch.FloatTensor(labels)

        # train_embs = embeds[idx_train, :, :]
        # val_embs = embeds[idx_val, :, :]
        # test_embs = embeds[idx_test, :, :]
        # train_lbls = labels[idx_train, :, :]
        # val_lbls = labels[idx_val, :, :]
        # test_lbls = labels[idx_test, :, :]

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
        np.savez(log_dgi+'/embeds.npz', embeds = ebs)
