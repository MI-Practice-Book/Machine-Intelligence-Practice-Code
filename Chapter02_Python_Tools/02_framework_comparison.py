"""
对应教材 2.4.2 节：主流框架对比
功能：使用三大框架分别执行简单的加法运算，展示其风格差异。
"""
print("--- 1. PyTorch 风格 ---")
import torch
pt_t = torch.add(torch.ones(2, 2), 1)
print(pt_t)

print("\n--- 2. TensorFlow 风格 ---")
import tensorflow as tf
tf_t = tf.add(tf.ones([2, 2]), 1)
print(tf_t)

print("\n--- 3. PaddlePaddle 风格 ---")
import paddle
pd_t = paddle.add(paddle.ones([2, 2]), 1)
print(pd_t)