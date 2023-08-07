# -*- coding:utf-8 -*-
#@Time : 2022/4/1 9:39 上午
#@Author: 李涛
#@File : _pandas.py

import numpy as np
import pandas as pd

# score_list = np.random.randint(30,100,size=100)
# print(score_list)
# bins = [30,40,50,60,70,80,90,100]
# x  = pd.cut(score_list,bins,labels=[1,2,3,4,5,6,7])
# print(x)
#
# df = pd.DataFrame()
# df["score"] = pd.Series(score_list)
# df["bins"] = pd.cut(df["score"] ,bins,labels=[1,2,3,4,5,6,7])
# print(df)

df =  pd.read_excel("./taiwan/weather.xlsx")
print(df)
group = df.groupby("city")
print(group)
print(group.groups)
print(group.get_group("SH"))
print(group.max())

def foo(attr):
    return attr.max()-attr.min()
print(group.agg(foo))

df = pd.read_excel("./taiwan/salesfunnel.xlsx")
print(df.columns)




