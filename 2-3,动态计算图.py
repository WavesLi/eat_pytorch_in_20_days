# -*- coding:utf-8 -*-
#@Time : 2022/4/2 9:55 上午
#@Author: 李涛
#@File : 2-3,动态计算图.py.py


import torch


class MyReLU(torch.autograd.Function):

        # 正向传播逻辑，可以用ctx存储一些值，供反向传播使用。
        @staticmethod
        def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)
        # 反向传播逻辑
        @staticmethod
        def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

import torch
w = torch.tensor([[3.0,1.0]])
b = torch.tensor([[3.0]])
X = torch.tensor([[-1.0,-1.0],[1.0,1.0]],requires_grad=True)
Y = torch.tensor([[2.0,3.0]])

# relu = MyReLU.apply # relu现在也可以具有正向传播和反向传播功能
Y_hat = X@w.t() + b
YT = Y_hat-Y
Y_F = torch.sum(YT)
Y_hat.register_hook(lambda grad: print('y1 grad: ', grad))
# Y_F.register_hook(lambda grad: print('y2 grad: ', grad))
YT.register_hook(lambda grad: print('y3 grad: ', grad))
Y_F.backward()
print(X.grad)


