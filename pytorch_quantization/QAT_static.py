'''
  1 模型init函数增加quant和dequant参数
    self.quant = torch.quantization.QuantStub()
    self.dequant = torch.quantization.DeQuantStub()
  2 模型forward函数调用quant和dequant函数
    x = self.quant(x)
    ......
    x = self.dequant(x)
  3 定义模型
  4 设置原模型为eval模式，如果原模型不进行推理测试，不设置亦可
  model_fp32.eval()
  5 设置qconfig，与Post Training Static Quantization使用的函数不一样，原理也不一样
  model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
  6 fuse_model
  model_fp32_fused = torch.quantization.fuse_modules(model_fp32,[['conv', 'bn', 'relu']])
7 模型prepare（必须将fused设置为train模式进行训练）
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused.train())
8 模型训练
training_loop(model_fp32_prepared)
9 转换模型（模型必须设置为eval模型）
model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)
10 模型推理
res = model_int8(input_fp32)
'''


import torch
import torch.nn as nn
import torch.optim as optim

import os

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


def training_loop(model_fp32_prepared):
    # TODO: 待完善
    pass

# define a floating point model where some layers could benefit from QAT
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval for fusion to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# fuse the activations to preceding layers, where applicable
# this needs to be done manually depending on the model architecture
model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
    [['conv', 'bn', 'relu']])

# Prepare the model for QAT. This inserts observers and fake_quants in
# the model needs to be set to train for QAT logic to work
# the model that will observe weight and activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused.train())

# run the training loop (not shown)
training_loop(model_fp32_prepared)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, fuses modules where appropriate,
# and replaces key operators with quantized implementations.
model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
input_fp32 = torch.randn(4, 1, 4, 4)
res = model_int8(input_fp32)
print("input_fp32.shape:",input_fp32.shape)
print("input_fp32:",input_fp32)
print("res.shape:",res.shape)
print("res:",res)

print(model_fp32)
print(model_int8)


# Save quantized model.
model_dir = "saved_models"
quantized_model_filename = "offical_QAT_static_model.pt"
quantized_model_filepath = os.path.join(model_dir,
                                        quantized_model_filename)
save_torchscript_model(model=model_int8,
                       model_dir=model_dir,
                       model_filename=quantized_model_filename)

# Load quantized model.
quantized_jit_model = load_torchscript_model(
    model_filepath=quantized_model_filepath, device='cpu')

# TODO:偏置量化