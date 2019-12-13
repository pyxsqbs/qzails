#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: qinbaoshuai
@ Date: 2019-08-23 15:24:11
@ Email: qinbaoshuai@cloudwalk.cn
@ LastEditors: qinbaoshuai
@ LastEditTime: 2019-09-20 15:16:20
@ Description: 异步线程的开始与停止
"""
from threading import Thread
import threading
import time
import inspect
import ctypes
import functools


def async_function(func):
    """
    @ description: 异步函数装饰器
    @ param func {function} 被装饰的异步函数
    @ return: {int} 异步函数的线程号
    """
    def wrapper(*t_args, **t_kwargs):
        thr = Thread(target=func, args=t_args, kwargs=t_kwargs)
        # thr.setDaemon(True)
        thr.start()
        # thr.
        return thr.ident
    return wrapper


def _async_raise(tid, exctype):
    """
    @ description: raises the exception, performs cleanup if needed
    @ param tid {int} 线程id
    @ param exctype {class} 异常类
    @ return: {int} exit_code 1：有效；0；无效线程id；其他：失败
    """
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    return res


def stop_thread(tid):
    """
    @ description: 停止异步线程
    @ param tid {int} 线程id号
    @ return: {int} exit_code 1：有效；0；无效线程id；其他：失败
    """
    return _async_raise(tid, SystemExit)


if __name__ == "__main__":
    @async_function
    def test():
        while True:
            print('-------')
            time.sleep(0.5)

    tid = test()
    print('tid: {} type: {}\n'.format(tid, type(tid)))
    time.sleep(5.2)
    print("main thread sleep finish")
    stop_thread(tid)
