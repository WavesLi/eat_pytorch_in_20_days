# -*- coding:utf-8 -*-
#@Time : 2022/3/30 2:48 下午
#@Author: 李涛
#@File : 2-2,自动微分机制.py

import torch

# x1 = torch.tensor([1,2,4],dtype=torch.float64,requires_grad=True)
# x = torch.tensor([1,2,3],dtype=torch.float64,requires_grad=True)
# a = torch.tensor(2,dtype=torch.float64)
# b = torch.tensor(-2,dtype=torch.float64)
# c = torch.tensor(1.0)
#  #方式一
# y = (a * torch.pow(x,2)+b*x+c+a * torch.pow(x1,2)).mean()
# y.backward()
# print(x.grad)
# print(x1.grad)
# #方式二
# y2 = torch.mean(a * torch.pow(x,2)+b*x+c+a * torch.pow(x1,2))
# torch.autograd.backward(y2)
# print(x.grad)
# print(x1.grad)
# # #方式三
# y3 = torch.mean(a * torch.pow(x,2)+b*x+c+a * torch.pow(x1,2))
# torch.autograd.backward(y)
# x.grad,x1.grad = torch.autograd.grad(y,[x,x1])
# print(x.grad)
# print(x1.grad)


import numpy as np
import torch
# f(x) = a*x**2 + b*x + c的最小值
# x = torch.tensor(100,requires_grad = True)

# optimizer = torch.optim.adam(params=[x],lr = 0.01)


# value =1
# lr = 0.01
# n = 10
# while abs(value**2 -n )>0.001 :
#     value = value-(value**2-n)*2*value*0.001
#     print(value)


# value =1
# lr = 0.01
# n = 10
# while abs(value**2 -n )>0.001 :
#     value = value-(value**2-n)*4*value*0.001
#     print(value)

# def f(x):
#     return x**2
#
# def d(x):
#     return 2*x
# x = 10/2
#
# y = y-(d(x)*step)


import torch

x = torch.tensor(2.0,requires_grad=True)
y = torch.tensor([2*x,3*3,4*x],requires_grad=True)
Y = y.mean()
print(y)
# print(x.shape)
# print(y.shape)
y.register_hook(lambda grad: print('y1 grad: ', grad))
Y.backward()
print(x.grad)
















