# 案例一：深度强化学习的行为模拟实践（单智能体）

本仓库对应课程 **案例一** 的完整实验代码，围绕 **连续控制场景下的单智能体行为学习** 展开。  
实验以 **Reacher-v4 机械臂目标点到达任务** 为例，使用 **PPO（Proximal Policy Optimization）** 算法，展示从环境验证、模型训练到结果分析与可视化的完整流程。

仓库中共包含 **5 份相互独立但逻辑连续的代码**，可按顺序运行，也可单独用于对应功能验证与分析。

---

## 代码功能说明

### 1️⃣ `code1_check_env.py` —— 环境验证与接口测试

**功能说明**  
用于验证 Gymnasium 与 MuJoCo 是否正确安装，并检查 Reacher-v4 环境的基本交互接口。

**主要内容**
- 创建 Reacher-v4 环境  
- 调用 `reset()`、`step()` 测试状态与动作接口  
- 测试离屏渲染（rgb_array）

**作用**
- 排查环境与依赖问题  
- 为后续强化学习训练提供稳定的运行基础  

---

### 2️⃣ `code2_train_ppo_reacher.py` —— PPO 训练主程序

**功能说明**  
基于 Stable-Baselines3 实现 PPO 算法，对 Reacher-v4 任务进行强化学习训练。

**主要内容**
- 构建并封装 Gym 环境  
- 定义 PPO 模型及关键超参数  
- 执行训练、评估并保存模型  

**输出结果**
- `models/`：训练过程中保存的模型  
- `ppo_reacher_final.zip`：最终训练完成的模型  
- `ppo_reacher_tb/`：TensorBoard 日志数据  

---

### 3️⃣ `code3_visualize_training.py` —— 训练过程可视化

**功能说明**  
对 PPO 训练过程中产生的日志数据进行离线分析和可视化。

**分析指标**
- 评估回报（Evaluation Reward）  
- 策略损失（Policy Loss）  
- 价值函数损失（Value Loss）  

**输出结果**
- 相关图像保存在 `figs_code3/` 目录下  

---

### 4️⃣ `code4_compare_before_after.py` —— 训练前后行为对比

**功能说明**  
在相同环境条件下，对比随机策略与 PPO 训练后策略的执行行为，并生成视频。

**对比对象**
- 随机策略（训练前）  
- PPO 策略（训练后，确定性执行）

**输出结果**
- `reacher_before_random.mp4`  
- `reacher_after_ppo.mp4`  

**作用**
- 从行为层面直观展示强化学习训练效果  

---

### 5️⃣ `code5_data_analysis.py` —— 训练数据分析

**功能说明**  
对训练或交互过程中采集的状态、动作与奖励数据进行统计与分析。

**分析内容**
- 关键状态分布（如末端与目标的相对位置）  
- 动作输出分布特征  
- 奖励随时间变化趋势  

**输出结果**
- 分析图像保存在 `figs_code5/` 目录下  

---

## 结果展示

### 训练前：随机策略

机械臂动作无明显目标导向，末端位置随机摆动。

- `reacher_before_random.mp4`

### 训练后：PPO 策略

机械臂能够连续、稳定地调整关节角度，使末端逐步逼近目标点。

- `reacher_after_ppo.mp4`

---

## 运行建议

1. 先运行 `code1_check_env.py`，确认环境配置正确  
2. 运行 `code2_train_ppo_reacher.py` 进行 PPO 训练  
3. 使用 `code3_visualize_training.py` 与 `code5_data_analysis.py` 进行结果分析  
4. 最后运行 `code4_compare_before_after.py` 生成行为对比视频  

---

## 最终文件目录如下
<img width="431" height="496" alt="image" src="https://github.com/user-attachments/assets/ae8a225d-799b-4f43-a194-44972c751222" />
