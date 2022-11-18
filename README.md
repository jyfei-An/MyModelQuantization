# PyTorch模型量化

参考资料：

https://zhuanlan.zhihu.com/p/299108528

https://pytorch.org/docs/stable/quantization.html#quantization-workflows

## Post Training Dynamic

参考资料：https://zhuanlan.zhihu.com/p/299108528
量化方法
1 直接使用quantize_dynamic函数对fp32_model进行量化即可

```python
只能量化以下算子
Linear、LSTM、LSTMCell、RNNCell、GRUCell
qconfig_spec = {
nn.Linear : default_dynamic_qconfig,
nn.LSTM : default_dynamic_qconfig,
nn.GRU : default_dynamic_qconfig,
nn.LSTMCell : default_dynamic_qconfig,
nn.RNNCell : default_dynamic_qconfig,
nn.GRUCell : default_dynamic_qconfig,
}
```

## Post Training Static Quantization

参考资料：https://zhuanlan.zhihu.com/p/299108528
量化方法：

1. 模型init函数增加quant和dequant参数
2. 模型forward函数调用quant和dequant函数
3. 定义模型
4. 设置模型为eval模式
5. 设置qconfig
6. fuse_model
7. 模型prepare（给每个子module插入Observer，用来收集和定标数据）
8. 数据校准（获取输入数据的分布），直接推理，不需要计算loss以及反向传播，因此整个过程模型使用eval模型即可
9. 转换模型
10. 模型推理

## Quantization Aware Training 

参考资料：https://zhuanlan.zhihu.com/p/299108528
量化方法：

1. 模型init函数增加quant和dequant参数
2. 模型forward函数调用quant和dequant函数
3. 定义模型
4. 设置原模型为eval模式，如果原模型不进行推理测试，不设置亦可
5. 设置qconfig，与Post Training Static Quantization使用的函数不一样，原理也不一样
6. fuse_model
7. 模型prepare（必须将fused设置为train模式进行训练）
8. 模型训练
9. 转换模型（模型必须设置为eval模型）
10. 模型推理



# Tensorflow模型量化