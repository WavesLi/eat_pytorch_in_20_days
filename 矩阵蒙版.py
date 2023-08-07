# -*- coding:utf-8 -*-
#@Time : 2022/4/8 6:51 下午
#@Author: 李涛
#@File : 矩阵蒙版.py

import torch
import random
source=torch.tensor([x for x in range(10)],dtype=torch.float64, requires_grad=True)
pred = source.reshape((2,5))
r,l = pred.shape
masked = torch.tensor([1 if random.random()>0.5 else 0 for i in range(r*l)],dtype=torch.bool).reshape((r,l))
# Y = torch.masked_select(pred,masked)
# Y = pred.masked_fill(masked,0)
layer = torch.nn.Dropout(p=0.5)
Y = layer(pred)
x = torch.tensor(2.0,requires_grad=True)
out = (Y*x).mean()
print(out)
out.backward()
print(source.grad)


# MAX_WORDS=100
# class Net(torch.nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#
#         # 设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
#         self.embedding = torch.nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=300,padding_idx=1)
#         self.m1={}
#         self.m1["layer1"] = torch.nn.Linear(in_features=300, out_features=9000)
#         self.mask1_1 = torch.tensor([1 if random.random()>0.3 else 0 for i in range(100*9000)],dtype=torch.bool).reshape((100,9000))
#         self.mask1_2 = torch.tensor([1 if random.random()>0.5 else 0 for i in range(100*9000)],dtype=torch.bool).reshape((100,9000))
#         self.mask1_3 = torch.tensor([1 if random.random()>0.2 else 0 for i in range(100*9000)],dtype=torch.bool).reshape((100,9000))
#         self.mask1_4 = torch.tensor([1 if random.random()>0.1 else 0 for i in range(100*9000)],dtype=torch.bool).reshape((100,9000))
#         self.mask1_5 = torch.tensor([1 if random.random()>0.7 else 0 for i in range(100*9000)],dtype=torch.bool).reshape((100,9000))
#         self.m1["layer2"] = torch.nn.Linear(in_features=9000, out_features=27000)
#         self.mask2_1 = torch.tensor([1 if random.random()>0.2 else 0 for i in range(9000*27000)],dtype=torch.bool).reshape((9000,27000))
#         self.mask2_2 = torch.tensor([1 if random.random()>0.1 else 0 for i in range(9000*27000)],dtype=torch.bool).reshape((9000,27000))
#         self.mask2_3 = torch.tensor([1 if random.random()>0.5 else 0 for i in range(9000*27000)],dtype=torch.bool).reshape((9000,27000))
#         self.mask2_4 = torch.tensor([1 if random.random()>0.3 else 0 for i in range(9000*27000)],dtype=torch.bool).reshape((9000,27000))
#         self.mask2_5 = torch.tensor([1 if random.random()>0.6 else 0 for i in range(9000*27000)],dtype=torch.bool).reshape((9000,27000))
#         self.m1["layer3"] = torch.nn.Linear(in_features=27000, out_features=27000)                                              ,
#         self.mask3_1 = torch.tensor([1 if random.random()>0.2 else 0 for i in range(27000*27000)],dtype=torch.bool).reshape((27000,27000))
#         self.mask3_2 = torch.tensor([1 if random.random()>0.1 else 0 for i in range(27000*27000)],dtype=torch.bool).reshape((27000,27000))
#         self.mask3_3 = torch.tensor([1 if random.random()>0.5 else 0 for i in range(27000*27000)],dtype=torch.bool).reshape((27000,27000))
#         self.mask3_4 = torch.tensor([1 if random.random()>0.3 else 0 for i in range(27000*27000)],dtype=torch.bool).reshape((27000,27000))
#         self.mask3_5 = torch.tensor([1 if random.random()>0.6 else 0 for i in range(27000*27000)],dtype=torch.bool).reshape((27000,27000))
#         self.m1["layer4"] = torch.nn.Linear(in_features=27000, out_features=9000)
#         self.m1["layer5"] = torch.nn.Linear(in_features=9000, out_features=3000)
#         self.m1["layer6"] = torch.nn.Linear(in_features=9000, out_features=1000)
#         self.m1["layer7"] = torch.nn.Linear(in_features=1000, out_features=300)
#         self.m1["layer8"] = torch.nn.Linear(in_features=300, out_features=100)
#         self.m1["layer9"] = torch.nn.Linear(in_features=300, out_features=30)
#         self.m1["layer10"] = torch.nn.Linear(in_features=30, out_features=9)
#         print(self.m1)
#
#     def forward(self, x):
#
#         x = self.embedding(x).transpose(1, 2)
#         x = self.conv(x)
#         y = self.dense(x)
#         return y
#
#
# net = Net()
