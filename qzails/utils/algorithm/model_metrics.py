#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: qinbaoshuai
@ Date: 2019-09-11 10:27:04
@ Email: qinbaoshuai@cloudwalk.cn
@ LastEditors: qinbaoshuai
@ LastEditTime: 2019-09-17 18:00:37
@ Description: 模型评估
"""

from sklearn.metrics import roc_curve, auc
import numpy as np


def model_metrics(y_true, y_score, coordinate_size=100):
    """
    @ description: 画ks、roc曲线
    @ param y_true {list} y（标签列）的值的列表，元素定义域{0,1}
    @ param y_score {list} y的预测概率的列表，元素定义域[0,1]
    @ param coordinate_size {int} 坐标点的个数
    @ return: {dict} 形如   {
                                "auc": 0.75,
                                "ks": 0.5,
                                "roc_curve": {
                                    "roc": [[0.0, 0.0], ...],
                                },
                                "ks_curve": {
                                    "fpr": [[0.0, 0.0], ...],
                                    "tpr": [[0.0, 0.0], ...],
                                    "ks": [[0.0, 0.0], ...],
                                }
                            }
    """
    assert isinstance(y_true, list), '入参 y_true 必须为 list'
    assert isinstance(y_score, list), '入参 y_score 必须为 list'
    assert isinstance(coordinate_size, int), '入参 coordinate_size 必须为 int'
    assert coordinate_size >= 2, '入参 coordinate_size 必须大于等于 2'
    assert len(y_true) == len(y_score), '入参 y_score 与 y_true 必须为 list，且长度相同'
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    assert y_true.dtype == np.int_ or y_true.dtype == np.float_, '入参 y_true 必须为 list，且其元素为数值'
    assert y_score.dtype == np.int_ or y_score.dtype == np.float_, '入参 y_true 必须为 list，且其元素为数值'
    assert y_true[(y_true != 0) & (y_true != 1)].size == 0 and np.unique(y_true).size == 2, '入参 y_true 必须为 list，且其元素只能为 0 或 1 且同时含有 0 和 1'
    assert y_score[(y_score < 0) | (y_score > 1)].size == 0, '入参 y_true 必须为 list，且其元素只能在区间[0,1]内'
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    thresholds_size = thresholds.size
    if thresholds_size > coordinate_size:
        index = np.linspace(0, thresholds_size - 1, coordinate_size, dtype=np.int_)
        fpr = fpr[index]
        tpr = tpr[index]
        thresholds = thresholds[index]
        thresholds_size = thresholds.size
    ks = tpr - fpr
    thresholds = np.linspace(0, 1, thresholds_size)
    auc_score = auc(fpr, tpr)
    ks_score = ks.max()
    ret_dict = dict()
    ret_dict['auc'] = auc_score
    ret_dict['ks'] = ks_score
    ret_dict['roc_curve'] = dict()
    ret_dict['ks_curve'] = dict()
    ret_dict['roc_curve']['roc'] = np.dstack((fpr, tpr))[0].tolist()
    ret_dict['ks_curve']['fpr'] = np.dstack((thresholds, fpr))[0].tolist()
    ret_dict['ks_curve']['tpr'] = np.dstack((thresholds, tpr))[0].tolist()
    ret_dict['ks_curve']['ks'] = np.dstack((thresholds, ks))[0].tolist()
    return ret_dict


if __name__ == "__main__":
    import pandas as pd
    import time
    import matplotlib.pyplot as plt

    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    df = pd.read_csv('/home/pyxsqbs/Documents/模型工厂/model-factory-operators/utils/algorithm/train_indessa_testpred.csv')
    y = df['loan_status'].tolist()
    scores = df['y_pred'].tolist()
    del df
    for_count = 10

    start_t = time.clock()
    fpr, tpr, thresholds = roc_curve(y, scores)
    ks = tpr - fpr
    auc_score = auc(fpr, tpr)
    ks_score = ks.max()
    elapsed_t = (time.clock() - start_t)

    plt.figure(figsize=(16, 12))
    for i in range(for_count):
        start = time.clock()
        index = model_metrics(y, scores, coordinate_size=(i + 1) * 1000)
        elapsed = (time.clock() - start)
        print("{} Time used: {:.4f} auc: {:.4f} ks: {:.4f}".format(i, elapsed, index['auc'], index['ks']))
        plt.subplot(3, 4, i + 1)
        plt.plot(np.array(index['ks_curve']['ks'])[:, 0], np.array(index['ks_curve']['ks'])[:, 1])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('tpr')
        plt.xlabel('fpr')
    print("{} Time used: {:.4f} auc: {:.4f} ks: {:.4f}".format('*', elapsed_t, auc_score, ks_score))
    plt.tight_layout()
    plt.savefig('test/curve_sparse_comparison.png')
