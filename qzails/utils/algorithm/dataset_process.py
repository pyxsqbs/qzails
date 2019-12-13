#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: qinbaoshuai
@ Date: 2019-09-12 15:49:36
@ Email: qinbaoshuai@cloudwalk.cn
@ LastEditors: qinbaoshuai
@ LastEditTime: 2019-12-10 02:37:43
@ Description: 数据集处理
"""
import numpy as np
import pandas as pd
import chardet
import os
from sklearn.model_selection import train_test_split
import warnings


DEBUG = False
if not DEBUG:
    warnings.filterwarnings("ignore")


def _detect_encoding(file_path):
    """
    @ description: 探查数据集的编码，不一定完全正确
    @ param file_path {str} 数据集的文件路径
    @ return: {dict} 探查的结果, 包括encoding、confidence、language
    """
    with open(file_path, "rb") as f:
        detect_result = chardet.detect(f.read())
        encoding = detect_result.get('encoding')
        confidence = detect_result.get('confidence')
        language = detect_result.get('language')
    del f
    return encoding, confidence, language


def read_csv_ignore_encoding(file_path):
    """
    @ description: 读取csv文件，忽略编码
    @ param file_path {str} csv文件路径
    @ return: {tuple} df为csv转成的df对象，details为数据集的一些基本属性，包括e250001、confidence、language
    """
    encoding = 'utf-8'
    confidence = 1.0
    language = ''
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        encoding = 'utf-8'
    except:
        if DEBUG:
            print('\n{} is not utf-8'.format(file_path))
        try:
            df = pd.read_csv(file_path, encoding='gb18030')
            encoding = 'gb18030'
        except:
            if DEBUG:
                print('\n{} is not gb18030'.format(file_path))
            try:
                df = pd.read_csv(file_path, encoding='gb2312')
                encoding = 'gb2312'
            except:
                if DEBUG:
                    print('\n{} is not gb2312'.format(file_path))
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                    encoding = 'gbk'
                except:
                    try:
                        if DEBUG:
                            print('\n{} is not gbk'.format(file_path))
                        encoding, confidence, language = _detect_encoding(file_path)
                        if DEBUG:
                            print('\n{} is {} {} {}'.format(file_path, encoding, confidence, language))
                        df = pd.read_csv(file_path, encoding=encoding)
                    except:
                        if DEBUG:
                            print('\n{file_path}\'s encode is not supported'.format(file_path))
                        df = ''
                        confidence = 0
    if DEBUG:
        print('\n{} is {} {} {}'.format(file_path, encoding, confidence, language))

    details = {
        'encoding': encoding,
        'confidence': confidence,
        'language': language
    }

    for label, content in df.items():
        # object变量预处理
        if content.dtype == np.object_:
            for i in content.unique():
                if isinstance(i, str):
                    # df元素去空格
                    df[label] = content.str.strip()
                    # 将特征中含有[, ]的值转换为[， ]
                    if ', ' in i:
                        df[label] = content.apply(lambda x: x.replace(', ', '， '))
                    break

    return df, details


def dataset_train_test_split(df, test_size, output_dir_path, dataset_filename, flag=None):
    """
    @ description: 从一个数据集划分成两个数据集，一个训练集，一个测试集
    @ param df {dataframe} 数据集，dataframe格式
    @ param test_size {float} 测试集划分比例，定义域(0,1)
    @ param output_dir_path {str} 输出的文件父目录，绝对路径
    @ param dataset_filename {str} 数据集的文件名
    @ param flag {None or str} 可选，数据集的标签列名
    @ return: {dict} 形如   {
                                "dataset_train_file_path": "{output_dir_path}/{dataset_filename}_train.csv"    //训练集文件路径
                                "dataset_test_file_path': "{output_dir_path}/{dataset_filename}_test.csv"    //测试集文件路径
                                "drop_col': ["feature_name_1", ...]    //忽略特征列表
                            }
    """
    assert hasattr(df, '_typ') and df._typ == 'dataframe' and df.shape[0] >= 12, '数据集须 dataframe 格式且数据集样本数须大于等于12'
    assert isinstance(test_size, float) and 0 < test_size < 1, '测试划分比例为 float 类型且定义域为(0,1)'
    assert isinstance(output_dir_path, str), '输出的文件父目录须为 str 类型'
    assert isinstance(dataset_filename, str) and dataset_filename.split('.')[-1] == 'csv', '数据集文件名须为 str 类型且为 csv 文件'
    assert test_size * df.shape[0] > 0.5, '数据集样本数或测试划分比例 test_size 过小，导致划分的测试集为空'
    assert df.shape[0] - test_size * df.shape[0] > 1.5, '数据集样本数过小或测试划分比例 test_size 过大，导致划分的训练集少于两条样本'

    # 处理同列值的数据类型不一致的问题
    for col in df.columns.values:
        if df.loc[:, col].dtype == np.object:
            df.loc[:, col] = df.loc[:, col].astype(str)
    if flag:
        # 有标签，则按照标签的值分层抽样
        assert isinstance(flag, str) and flag in df.columns.values, '标签列名须为 str 类型且在数据集列名集合中'
        y = df.loc[:, flag]
        assert y.dtype == np.int_, '标签列的值须为 int 类型'
        assert y[(y != 0) & (y != 1)].size == 0 and np.unique(y).size == 2, '标签列的值只能为 0 或 1 且同时含有 0 和 1'
        assert y.dropna().nunique() > 1, '标签列的值除空值外至少大于1种'
        flag_zero = (y == 0).sum()
        assert flag_zero > 5 and y.shape[0] - flag_zero > 5, '正负样本个数均需大于5'
        # 正负样本各取5个
        pos_sample_index = np.argwhere(y == 1)[:, 0][0:5]
        neg_sample_index = np.argwhere(y == 0)[:, 0][0:5]
        pos_neg_samples = df.loc[np.hstack((pos_sample_index, neg_sample_index)), :]
        df = df.drop(np.hstack((pos_sample_index, neg_sample_index)), axis=0)
        y = y.drop(np.hstack((pos_sample_index, neg_sample_index)), axis=0)
        # 分层抽样划分数据集
        if y.nunique() == 1 or y.sum() == 1 or y.shape[0] - y.sum() == 1:
            df_test = df.sample(frac=test_size)
            df_train = df.drop(df_test.index.values, axis=0)
        else:
            df = df.drop([flag], axis=1)
            df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=test_size, stratify=y)
            df_test.loc[:, flag] = y_test
            df_train.loc[:, flag] = y_train
        if df_test.shape[0] == 0:
            df_test = df.loc[0:1, :]
            df_train = df.drop(df_test.index.values, axis=0)
        if df_test.loc[:, flag].nunique() == 1:
            y_unique = df_test.loc[:, flag].unique()[0]
            pos_neg_samples = pos_neg_samples.reset_index(drop=True)
            index_temp = np.argwhere(pos_neg_samples.loc[:, flag] != y_unique)[0][0]
            df_test = df_test.append(pos_neg_samples.loc[index_temp, :], ignore_index=True)
        df_train = df_train.append(pos_neg_samples, ignore_index=True)
    else:
        # 无标签，则随机抽样
        df_test = df.sample(frac=test_size)
        df_train = df.drop(df_test.index.values, axis=0)
    # 重建索引
    df_test = df_test.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    # 生成忽略特征列表
    df_train_nuique = df_train.nunique()
    drop_columns = df_train_nuique[df_train_nuique == 1].index.values
    assert df_train.shape[1] > drop_columns.size, '划分数据集错误，drop_col生成错误，drop_col包含数据集所有特征'
    # 生成文件路径
    filename = dataset_filename.replace(".csv", "")
    dataset_train_file_path = os.path.join(output_dir_path, '{}_train.csv'.format(filename))
    dataset_test_file_path = os.path.join(output_dir_path, '{}_test.csv'.format(filename))
    df_train.to_csv(dataset_train_file_path, index=False)
    df_test.to_csv(dataset_test_file_path, index=False)
    ret_dict = dict()
    ret_dict['dataset_train_file_path'] = dataset_train_file_path
    ret_dict['dataset_test_file_path'] = dataset_test_file_path
    ret_dict['drop_col'] = drop_columns.tolist()
    return ret_dict


if __name__ == "__main__":
    import time
    df, _ = read_csv_ignore_encoding('~/Desktop/train_v2_filtered.csv')
    # df = df[4:7].reset_index(drop=True)
    # df.loc[:, 'PAY_AMT1'] = df.loc[:, 'PAY_AMT1'].apply(lambda x: 1 if x == 5048 else 0)
    start = time.clock()
    print(dataset_train_test_split(df, 0.3, '~/Desktop/', 'train_v2_filtered.csv', 'loss'))
    elapsed = (time.clock() - start)
    print("dataset_train_test_split Time used: {:.4f}".format(elapsed))
