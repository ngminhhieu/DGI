import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
from models.cvaelstm.train_cvae_lstm import ConfigCvaeLstm
from models.dgi.train_dgi import ConfigDGI

def seed():

    np.random.seed(2)

    rn.seed(12345)

    tf.random.set_seed(1234)
    
if __name__ == '__main__':
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='cvae_lstm_train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    # load config for seq2seq model
    with open(args.config_file) as f:
        config = yaml.load(f)

    if args.mode == 'cvae_lstm_train':
        model = ConfigCvaeLstm(is_training=True,**config)
        model.TakeData()
        model.Split()
        model.Train()
    elif args.mode == 'cvae_lstm_test':
        # predict
        model = ConfigCvaeLstm(is_training=False,**config)
        model.TakeData()
        model.Split()
        model.Test()
    elif args.mode == 'dgi_train':
        model = ConfigDGI(is_training=True,**config)
        model.train()
    elif args.mode == 'dgi_test':
        model = ConfigDGI(is_training=False,**config)
        model.test()
    else:
        raise RuntimeError("Mode needs to be train/test!")
