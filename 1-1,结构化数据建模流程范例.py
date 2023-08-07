# -*- coding:utf-8 -*-
#@Time : 2022/3/17 6:57 下午
#@Author: 李涛
#@File : 1-1,结构化数据建模流程范例.py

import os
import datetime
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import dataloader,Dataset,TensorDataset

dftrain_raw = pd.read_csv("./data/titanic/train.csv")
dftest_raw = pd.read_csv("./data/titanic/test.csv")
show = dftest_raw.head(5)
print(dftrain_raw.describe())
print(show)
print(dftrain_raw["Survived"].value_counts())
print(dftrain_raw.query('Survived == 0'))
age_groups = dftrain_raw.query('Survived == 0')['Age'].value_counts()
print(age_groups.index)



import plotly.graph_objects as go
import plotly.offline as of  	# 这个为离线模式的导入方法
data = dftrain_raw["Age"][dftrain_raw["Survived"]==0].value_counts()
line1 = go.Bar(y=data.values, x=data.index, name='Survived')   # name定义每条线的名称
# line1 = go.Bar(y=dftrain_raw["Age"], x=df, name='Survived')
# line2 = go.Scatter(y=data['Sex'], x=data['Age'], name='Sex')
# line1.go.Bar()
fig = go.Figure([line1])
fig.update_layout(
    title = 'New Zealand Weather', #定义生成的plot 的标题
    xaxis_title = 'DATE',		#定义x坐标名称
    yaxis_title = 'Weather'		#定义y坐标名称
)
of.plot(fig)


data1 = dftrain_raw["Age"][dftrain_raw["Survived"]==0].value_counts()
data2 = dftrain_raw["Age"][dftrain_raw["Survived"]==1].value_counts()
data1 = data1.sort_index()
data2 = data2.sort_index()
# line1 = go.Bar(y=data1.index, x=data.values, name='Survived==0')   # name定义每条线的名称
# line2 = go.Bar(y=data2.index, x=data.values, name='Survives==1')   # name定义每条线的名称
print(data1.values)
print(data1.index)
# line1 = go.Bar(y=data1.values, x=data.index, name='Survived==0')   # name定义每条线的名称
# line2 = go.Bar(y=data2.values, x=data.index, name='Survives==1')   # name定义每条线的名称


line1 = go.Scatter(y=data1.values, x=data1.index, name='Survived==0',mode="markers+lines")   # name定义每条线的名称
line2 = go.Scatter(y=data2.values, x=data1.index, mode="lines+markers",name='Survives==1',)   # name定义每条线的名称

# line1 = go.Scatter(y=data1.index, x=data.values, name='Survived==0',mode="lines+markers")   # name定义每条线的名称
# line2 = go.Scatter(y=data2.index, x=data.values, name='Survives==1',mode="lines")   # name定义每条线的名称
# line1 = go.Bar(y=dftrain_raw["Age"], x=df, name='Survived')
# line2 = go.Scatter(y=data['Sex'], x=data['Age'], name='Sex')
# line1.go.Bar()
fig = go.Figure([line1,line2])
fig.update_layout(
    title = 'New Zealand Weather', #定义生成的plot 的标题
    xaxis_title = 'DATE',		#定义x坐标名称
    yaxis_title = 'Weather'		#定义y坐标名称
)
of.plot(fig)



