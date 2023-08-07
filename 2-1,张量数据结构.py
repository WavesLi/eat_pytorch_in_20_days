import numpy as np
import torch


np_array = np.arange(1,12,1,dtype=np.float64)
print(np_array)
tensor_array = torch.from_numpy(np_array,)
tensor_to_array = tensor_array.clone().numpy()
tensor_array.add_(1)
print(tensor_array)
print(tensor_to_array)
# np_array = np.add(np_array,1)
np.add(np_array,1,out=np_array)
print(np_array)
print(tensor_array)
# print(help(torch.from_numpy))
# print(help(np.arange))
print(tensor_array.dtype)
zeros = torch.zeros(9,dtype=torch.float64)
zeros.add_(1)
print(zeros.dtype)
print(zeros)
a = zeros[0]
print(a.item())
print(zeros.tolist())