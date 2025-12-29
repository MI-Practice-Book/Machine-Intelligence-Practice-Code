"""
对应教材 2.3.3 节：常用科学计算库演示
功能：演示 NumPy 数据生成、Pandas 结构化分析、Matplotlib 可视化。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 使用 NumPy 生成模拟信号（正弦波 + 随机噪声）
x = np.linspace(0, 10, 100)
noise = np.random.normal(0, 0.1, 100)
y = np.sin(x) + noise

# 2. 使用 Pandas 管理数据
df = pd.DataFrame({'Time': x, 'Signal': y})
print("--- 数据集前 5 行 ---")
print(df.head())

# 3. 使用 Matplotlib 绘图
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='采集信号', color='blue')
plt.title("Scientific Computing Stack Demo")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()