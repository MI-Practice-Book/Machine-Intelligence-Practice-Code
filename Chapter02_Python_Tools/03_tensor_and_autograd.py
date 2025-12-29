"""
对应教材 2.4.3 节：框架通用概念
功能：演示 4 维张量属性、设备迁移以及自动求导。
"""
import torch

# 1. 张量属性演示
# 模拟：[批次, 通道, 高度, 宽度]
images = torch.randn(32, 3, 224, 224)
print(f"张量形状: {images.shape}")
print(f"存储设备: {images.device}")

# 2. 设备迁移 (CPU -> GPU)
if torch.cuda.is_available():
    images = images.to('cuda')
    print(f"迁移后设备: {images.device}")

# 3. 自动微分演示
# 定义 y = x^2 + 2x + 1
x = torch.tensor([3.0], requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
# dy/dx = 2x + 2. 当 x=3 时，梯度为 8
print(f"在 x=3 处，y=x^2+2x+1 的梯度为: {x.grad.item()}")