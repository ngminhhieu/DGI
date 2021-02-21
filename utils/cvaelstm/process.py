from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator as SklearnBaseEstimator
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
plt.style.use("fivethirtyeight")

def getData(dataset, sequence_length, output_dim=1):
    
    T = dataset.shape[0]-sequence_length
    ips = np.empty(shape=(T, sequence_length, dataset.shape[1]-output_dim))
    ops = np.empty(shape=(T, sequence_length, output_dim))
    for i in range(T):
        ips[i, :, :] = dataset[i:i+sequence_length, 0:(dataset.shape[1]-output_dim)]
        ops[i, :, :] = dataset[i:i+sequence_length, -output_dim:]
    ips = np.transpose(ips, (0, 2, 1))
    ops = np.transpose(ops, (0, 2, 1))
    return ips, ops


def standardizeData(X, SS = None, train = False):
    """Given a list of input features, standardizes them to bring them onto a homogenous scale

    Args:
        X ([dataframe]): [A dataframe of all the input values]
        SS ([object], optional): [A MinMaxScaler object that holds mean and std of a standardized dataset]. Defaults to None.
        train (bool, optional): [If False, means validation set to be loaded and SS needs to be passed to scale it]. Defaults to False.
    """
    if train:
        SS = MinMaxScaler()
        new_X = SS.fit_transform(X)
        return (new_X, SS)
    else:
        new_X = SS.transform(X)
        return (new_X, None)

def train_val_test_split(data, valid_size, test_size):
    
    train_len = int(data.shape[0] * (1 - test_size - valid_size))
    valid_len = int(data.shape[0] * valid_size)

    train_set = data[0:train_len]
    valid_set = data[train_len: train_len + valid_len]
    test_set = data[train_len + valid_len:]

    return train_set, valid_set, test_set


class BaseEstimator(SklearnBaseEstimator):
    # http://msmbuilder.org/development/apipatterns.html

    def summarize(self):
        return 'NotImplemented'