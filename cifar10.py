# coding: utf-8
import os
import pickle
import numpy as np


def load_cifar10(normalize=False, flatten=False, one_hot_label=False):
    # 数据集文件夹路径：当前文件所在目录下的 cifar-10-batches-py
    dataset_dir = os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"找不到 CIFAR-10 数据文件夹，请确认路径：{dataset_dir}")

    def _load_batch(filename):
        with open(os.path.join(dataset_dir, filename), 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            x = datadict[b'data']
            y = datadict[b'labels']
            x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
            y = np.array(y)
            if flatten:
                x = x.reshape(x.shape[0], -1)
            return x, y

    xs = []
    ys = []
    for b in range(1, 6):
        x, y = _load_batch(f'data_batch_{b}')
        xs.append(x)
        ys.append(y)

    x_train = np.concatenate(xs)
    t_train = np.concatenate(ys)
    x_test, t_test = _load_batch('test_batch')

    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    if one_hot_label:
        T_train = np.zeros((t_train.size, 10))
        for idx, row in enumerate(T_train):
            row[t_train[idx]] = 1
        t_train = T_train

        T_test = np.zeros((t_test.size, 10))
        for idx, row in enumerate(T_test):
            row[t_test[idx]] = 1
        t_test = T_test

    return (x_train, t_train), (x_test, t_test)