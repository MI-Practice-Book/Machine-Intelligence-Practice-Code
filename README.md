# 机器智能实践 - 配套教学代码库

## 📖 教材介绍

本代码库是《机器智能实践》教材的配套实践代码仓库，涵盖了机器智能领域的核心技术和实战项目。每个章节都提供了完整的、可运行的代码实现，帮助学生深入理解理论知识并掌握实践技能。


---

## 📚 章节目录

### 📂 [Chapter02 - Python工具实践](./Chapter02_Python_Tools)
**主题**: Python编程基础与常用工具库  
**内容**:
- NumPy数组操作与科学计算
- Matplotlib数据可视化
- Pandas数据处理与分析
- Scikit-learn机器学习基础

**学习目标**: 掌握Python数据科学生态系统的核心工具

---

### 📂 [Chapter03 - 结构仿真设计](./Chapter03_Structure_Simulation)
**主题**: 计算机视觉基础与目标检测/跟踪

**内容**: 
  - 自定义数据集训练
  - 模型评估与优化
  - 实时目标检测
  - 卡尔曼滤波运动预测
  - 级联匹配数据关联
  - 外观特征深度学习
  
**学习目标**: 理解目标检测与跟踪的完整pipeline

---

### 📂 [Chapter04 - 功能仿真设计](./Chapter04_Function_Simulation)
**主题**: 功能模拟与智能系统构建
**内容**:
- 基于产生式规则的专家系统
- 知识库构建与推理机设计
- RAG检索增强生成技术
- 智能问答系统构建与优化

**学习目标**: 掌握功能模拟系统的设计原理与实现方法，理解符号推理与语义理解两类智能系统的构建流程

---

### 📂 [Chapter05 - 行为仿真设计](./Chapter05_Behavior_Simulation)
**主题**: 强化学习与智能体控制

**项目亮点**:
- 多智能体协同控制
- 环境建模与仿真
- 策略优化算法
- 实时决策系统

**学习目标**: 理解智能体的行为决策机制

---

### 📂 [Chapter06 - LLM技术应用](./Chapter06_LLM_Technology)
**主题**: 大语言模型与自然语言处理  
**内容**:
- Transformer架构解析
- 预训练模型微调
- Prompt工程实践
- RAG检索增强生成

**学习目标**: 掌握大语言模型的应用开发

---

### 📂 [Chapter07 - 竞赛项目实战](./Chapter07_Competitions)
**主题**: 算法竞赛与项目实战  
**内容**:
- Kaggle竞赛案例分析
- 特征工程技巧
- 模型集成方法
- 项目答辩与展示

**学习目标**: 提升综合问题解决能力

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.7+
- **CUDA**: 11.1+ (GPU训练推荐)
- **操作系统**: Linux / MacOS / Windows

### 统一安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-repo/Machine-Intelligence-Practice-Code.git
cd Machine-Intelligence-Practice-Code

# 创建虚拟环境(推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装基础依赖
pip install -r requirements.txt
```

### 各章节独立运行

每个章节目录都包含独立的 `README.md` 和 `requirements.txt`：

```bash
# 进入章节目录
cd Chapter03_Structure_Simulation/Task2_DeepSORT_Tracking

# 安装章节特定依赖
pip install -r requirements.txt

# 查看详细使用说明
cat README.md

# 运行示例
python run_tracking.py --video test.mp4 --show
```

---

## 💻 使用指南

### 项目结构模板

每个任务/项目通常包含以下结构：

```
TaskX_ProjectName/
├── models/          # 模型定义
├── data/            # 数据加载与处理
├── utils/           # 工具函数
├── configs/         # 配置文件
├── train.py         # 训练脚本
├── evaluate.py      # 评估脚本
├── demo.py          # 演示脚本
├── requirements.txt # 依赖列表
└── README.md        # 项目说明
```

---

## 📊 数据集说明

### 公开数据集

| 数据集 | 章节 | 下载链接 | 说明 |
|--------|------|----------|------|
| COCO | Ch03-Task1 | [官网](https://cocodataset.org/) | 目标检测标准数据集 |
| MOT17 | Ch03-Task2 | [官网](https://motchallenge.net/) | 多目标跟踪数据集 |
| ImageNet | Ch04 | [官网](https://image-net.org/) | 图像分类数据集 |

### 自定义数据集

部分章节提供了数据集生成脚本或标注工具，详见各章节README。

---

## 🛠️ 开发工具推荐

### IDE / 编辑器
- **VSCode**: 推荐安装Python、Jupyter插件
- **PyCharm**: 专业Python IDE
- **Jupyter Notebook**: 交互式开发

### 调试工具
- **TensorBoard**: 可视化训练过程
- **WandB**: 实验跟踪与管理
- **pdb**: Python调试器

### 版本管理
- **Git**: 代码版本控制
- **DVC**: 数据版本控制

