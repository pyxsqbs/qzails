#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: qinbaoshuai
@ Date: 2019-07-17 10:17:41
@ Email: qinbaoshuai@cloudwalk.cn
@ LastEditors: qinbaoshuai
@ LastEditTime: 2019-08-16 14:39:50
@ Description: 公共变量的保存、载入，所有算子都可以访问到公共变量
"""
import json
import os
import dill as pickle

# 公共变量的二进制文件的父目录
DATA_PK_PATH = 'data/serialize_variable'


def save_varible(_self, key, value):
    """
    @ description: 保存或更新公共变量
    @ param _self {dict} api参数字典，例如包括字典项"model_id"
    @ param key {str} 将要保存的变量的key值
    @ param value {any} 将要保存的变量的value
    @ return: None
    """
    assert 'model_id' in _self.keys()
    assert 'version_id' in _self.keys()
    model_id = _self['model_id']
    version_id = _self['version_id']
    with open(os.path.join(DATA_PK_PATH, '{}_{}_{}.pk'.format(key, model_id, version_id)), 'wb') as pk_f:
        pickle.dump(value, pk_f, protocol=pickle.HIGHEST_PROTOCOL)


def load_varible(_self, key):
    """
    @ description: 加载公共变量
    @ param _self {dict} api参数字典，例如包括"model_id"
    @ return: {any} 公共变量
    """
    assert 'model_id' in _self.keys()
    assert 'version_id' in _self.keys()
    model_id = _self['model_id']
    version_id = _self['version_id']
    with open(os.path.join(DATA_PK_PATH, '{}_{}_{}.pk'.format(key, model_id, version_id)), 'rb') as pk_f:
        value = pickle.load(pk_f)
    return value


if __name__ == "__main__":
    _self = {}
    _self['model_id'] = '12345'
    _self['version_id'] = '1.0.0'
    save_varible(_self, 'test_str', '什么鬼')
    print(load_varible(_self, 'test_str'))
