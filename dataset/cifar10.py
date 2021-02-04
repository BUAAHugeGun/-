from __future__ import print_function

import pickle
import numpy as np
import os
import platform


# 读取文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 取python版本号
    if version[0] == '2':
        return pickle.load(f)  # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)  # dict类型
        X = datadict['data']  # X, ndarray, 像素值
        Y = datadict['labels']  # Y, list, 标签, 分类

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []  # list
    ys = []

    # 训练集batch 1～5
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)  # 在list尾部添加对象X, x = [..., [X]]
        ys.append(Y)
        print("loading cifa10!")
    Xtr = np.concatenate(xs)  # [ndarray, ndarray] 合并为一个ndarray
    Ytr = np.concatenate(ys)
    del X, Y

    # 测试集
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
