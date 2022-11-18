# Post Training Dynamic Quantization只能量化一下算子,没有对bias进行量化
# Linear
# LSTM
# LSTMCell
# RNNCell
# GRUCell

import torch

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

# create a model instance
model_fp32 = M()
print(model_fp32)
# create a quantized model instance

# 直接使用quantize_dynamic函数对模型进行量化即可
model_int8 = torch.quantization.quantize_dynamic(model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

print(model_int8)

# run the model
input_fp32 = torch.randn(4, 4, 4, 4)
res = model_int8(input_fp32)




state_dict = model_fp32.state_dict()
for (k, v) in state_dict.items():
    print(k, v)
'''
# 可以用下面这种方式查看model_fp32模型参数，但是无法查看model_int8参数，因为量化会将模型的weight和bias进行打包，无法直接查看
for name,parameters in model_fp32.named_parameters():
    print(name,':',parameters.size())
for parameters in model_fp32.parameters():
    print(parameters)
'''
print("============")

state_dict = model_int8.state_dict()
for (k, v) in state_dict.items():
    print(k, v)


# TODO: 1 获取量化层中scale和zero_bias,使用tensor量化方法进行比较
# TODO: 2 保存量化后的模型权重