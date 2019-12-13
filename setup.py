#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: qinbaoshuai
@ Date: 2019-12-13 16:19:23
@ Email: qinbaoshuai@cloudwalk.cn
@ LastEditors: qinbaoshuai
@ LastEditTime: 2019-12-13 16:22:30
@ Description: setup.py
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qzails-pyxsqbs",
    version="0.0.1",
    author="qinbaoshuai",
    author_email="pyxsqbs@163.com",
    description="A machine learning engineering chemical tools library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyxsqbs/qzails",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)