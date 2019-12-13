#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: Qian Weng
@ Date: 2018-03-15 13:58:34
@ Email: qinbaoshuai@cloudwalk.cn
@ LastEditors: qinbaoshuai
@ LastEditTime: 2019-12-13 17:41:53
@ Description: logger_manager.py
"""

from logging import Logger
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import logging
import os


def init_logger(logdir, logger_name):
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if logger_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s]: PID:%(process)d file:%(filename)s line:%(lineno)s func:%(funcName)s %(levelname)s %(message)s"
        formatter = logging.Formatter(format_str, datefmt)

        # screen handler of level info
        handler = StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        # file handler of level info
        handler = TimedRotatingFileHandler(os.path.join(logdir, "info.log"), when='midnight', backupCount=30, encoding='utf-8')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        # file handler of level error
        handler = TimedRotatingFileHandler(os.path.join(logdir, 'error.log'), when='midnight', backupCount=30, encoding='utf-8')
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)

    logger = logging.getLogger(logger_name)
    return logger

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
log_name = "main"
logger = init_logger(log_dir, log_name)

logger.info('test')
