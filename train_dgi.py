import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn

from models.dgi.dgi import DGI
from models.dgi.reg import MultipleRegression
from utils.dgi import process
import sys

dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

pm_dataset = pd.read_csv('./data/pm.csv')
pm_dataset = pm_dataset.replace("**", 0)
pm_dataset = pm_dataset.to_numpy()
pm_data = pm_dataset[:, 4:]
pm_data = pm_data.astype(np.float)

adj = process.build_graph('./data/locations.csv')
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


if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
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
            torch.save(model.state_dict(), './data/trained/best_dgi.pkl')
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
    model.load_state_dict(torch.load('./data/trained/best_dgi.pkl'))
    embeds[i, :, :], _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

np.savez('./data/trained/embeds3d.npz', embeds3d = embeds)
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
np.savez('./data/trained/embeds.npz', embeds = ebs)

