# -*- coding:utf-8 -*-
#@Time : 2022/3/29 5:34 下午
#@Author: 李涛
#@File : data_al.py

import  pandas as pd
import datetime

df1 = pd.read_excel("./shuju1.xlsx",usecols=[2,3,12])
df2 = pd.read_excel("./shuju2.xlsx",usecols=[2,3,12])
df = pd.concat([df1,df2],axis=0)
print(df.info())
print(df.columns)
df["发表时间"] = pd.to_datetime(df["发表时间"])
# df = df[(df['发表时间'] > "2022-03-01 00:00:00"  ) ]
df_td = df[(df['发表时间'] > "2022-03-03 03:07:09"  ) & (df['发表时间'] <="2022-03-03 21:00:00" )]
df_btd = df[(df['发表时间'] <= "2022-03-03 03:07:09"  ) | (df['发表时间'] >"2022-03-03 21:00:00" )]
print(df_btd['作者ID'])
print(df_td["作者ID"])
df_finnal = df_btd[~df_btd['作者ID'].isin(df_td["作者ID"])]
result = pd.DataFrame()
result["作者名称"] = df_finnal["作者名称"]
result["作者ID"] = df_finnal["作者ID"]
result.drop_duplicates(subset=["作者名称","作者ID"],keep="first",inplace=True)
# result.to_excel("./dh3月以后.xlsx")
result.to_excel("./dh2月和3月.xlsx")
