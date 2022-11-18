import torch

x = torch.tensor([0.4097, -0.2896, -0.4931])
print("x:",x)

#quantize_per_tensor函数就是使用给定的scale和zp来把一个float tensor转化为quantized tensor
#xq = round(x / scale + zero_point)
xq = torch.quantize_per_tensor(x, scale = 0.5, zero_point = 8, dtype=torch.quint8)
print("xq:",xq)

print("xq.int_repr():",xq.int_repr())

#quantize_per_tensor本质上和使用round函数进行转换类似
y = torch.round(x/0.5+8)
print("round_func:",y)

# 反量化xdq: tensor([ 0.5000, -0.5000, -0.5000])，与原始数据有误差
xdq = xq.dequantize()
print("xdq:",xdq)