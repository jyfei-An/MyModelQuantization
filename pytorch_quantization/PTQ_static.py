'''
  1 模型init函数增加quant和dequant参数
    self.quant = torch.quantization.QuantStub()
    self.dequant = torch.quantization.DeQuantStub()
  2 模型forward函数调用quant和dequant函数
    x = self.quant(x)
    ......
    x = self.dequant(x)
  3 定义模型
  4 设置模型为eval模式
  model_fp32.eval()
  5 设置qconfig
  model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  6 fuse_model
  model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
  只有如下的op和顺序才可以合并：
    Convolution, Batch normalization
    Convolution, Batch normalization, Relu
    Convolution, Relu
    Linear, Relu
    Batch normalization, Relu
7 模型prepare（给每个子module插入Observer，用来收集和定标数据）
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
8 数据校准（获取输入数据的分布），直接推理，不需要计算loss以及反向传播，因此整个过程模型使用eval模型即可
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)
9 转换模型
model_int8 = torch.quantization.convert(model_fp32_prepared)
10 模型推理
res = model_int8(input_fp32)
'''


import torch

# define a floating point model where some layers could be statically quantized
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

# create a model instance
model_fp32 = M()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)

print(model_fp32)
print(model_int8)