# Chapter 3: 基于YOLOv5的目标检测系统

## 📚 项目简介

本项目是《机器智能》教材第3章"结构模拟设计"的实践任务1,实现了基于YOLOv5的目标检测系统。通过本项目,学习者将掌握:

- VOC数据集的处理和格式转换
- YOLOv5模型的训练、评估和推理流程
- 目标检测系统的完整开发流程
- 深度学习项目的工程化实践

**对应教材章节**: 3.1 实践任务1：YOLOv5目标检测

---

## 📂 项目结构

```
Task1_YOLOv5_Detection/
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── logger.py                   # 日志管理工具
│   ├── voc_converter.py            # VOC格式转换工具
│   └── dataset_analyzer.py         # 数据集分析工具
├── scripts/                        # 脚本模块
│   ├── prepare_dataset.py          # 数据集准备主脚本
│   ├── train_yolov5.py            # 训练封装脚本
│   ├── evaluate_model.py          # 评估脚本
│   └── batch_inference.py         # 批量推理脚本
├── configs/                        # 配置文件(运行后生成)
│   └── VOC2012.yaml               # 数据集配置
├── requirements.txt                # 依赖列表
└── README.md                       # 本文件
```

---

## 🚀 快速开始

### 1. 环境配置

**系统要求**:
- Python 3.8+
- CUDA 11.3+ (GPU训练)
- 8GB+ RAM
- 20GB+ 磁盘空间

**安装步骤**:

```bash
# 1. 克隆YOLOv5仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0  # 使用稳定版本

# 2. 创建虚拟环境(推荐)
conda create -n yolov5 python=3.8
conda activate yolov5

# 3. 安装PyTorch (根据CUDA版本选择)
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# 4. 安装YOLOv5依赖
pip install -r requirements.txt

# 5. 安装本项目依赖
cd ../Task1_YOLOv5_Detection
pip install -r requirements.txt
```

**验证安装**:
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

---

## 📖 使用指南

### 阶段1: 数据准备 (对应教材 3.1.3节)

**功能**: 自动下载VOC数据集、格式转换、质量分析

```bash
# 准备VOC2012数据集(完整流程)
python scripts/prepare_dataset.py \
    --data_root data \
    --year 2012

# 只转换格式(数据已下载)
python scripts/prepare_dataset.py \
    --data_root data \
    --year 2012 \
    --skip_download

# 准备VOC2007数据集
python scripts/prepare_dataset.py \
    --data_root data \
    --year 2007
```

**输出文件**:
```
data/
├── VOC2012/              # 原始VOC数据
├── labels/
│   ├── train/           # YOLO格式训练集标注
│   └── val/             # YOLO格式验证集标注
├── reports/
│   ├── train_analysis.json
│   └── val_analysis.json
├── logs/
│   └── prepare_dataset.log
└── VOC2012.yaml         # YOLOv5配置文件
```

**相关模块**:
- `utils/voc_converter.py`: VOC→YOLO格式转换
- `utils/dataset_analyzer.py`: 数据集统计分析

---

### 阶段2: 模型训练 (对应教材 3.1.4节)

**功能**: 封装YOLOv5训练流程,支持参数化配置和断点续训

**基础训练**:
```bash
python scripts/train_yolov5.py \
    --yolov5_root /path/to/yolov5 \
    --data data/VOC2012.yaml \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --weights yolov5s.pt \
    --device 0 \
    --name voc_train_exp1
```

**高级配置**:
```bash
# 多GPU训练 + 余弦学习率 + 多尺度
python scripts/train_yolov5.py \
    --yolov5_root /path/to/yolov5 \
    --data data/VOC2012.yaml \
    --batch 32 \
    --epochs 300 \
    --device 0,1 \
    --optimizer AdamW \
    --lr0 0.001 \
    --cos-lr \
    --multi-scale \
    --cache ram \
    --name voc_advanced

# 断点续训
python scripts/train_yolov5.py \
    --yolov5_root /path/to/yolov5 \
    --data data/VOC2012.yaml \
    --resume \
    --name voc_train_exp1
```

**训练参数说明**:

| 参数 | 说明 | 默认值 | 教材章节 |
|------|------|--------|---------|
| `--img` | 输入图像尺寸 | 640 | 3.1.4 |
| `--batch` | 批次大小 | 16 | 3.1.4 |
| `--epochs` | 训练轮数 | 100 | 3.1.4 |
| `--lr0` | 初始学习率 | 0.01 | 3.1.4 |
| `--optimizer` | 优化器(SGD/Adam/AdamW) | SGD | 3.1.4 |
| `--cos-lr` | 余弦学习率调度 | False | 3.1.4 |
| `--multi-scale` | 多尺度训练 | False | 3.1.4 |
| `--cache` | 缓存图像(ram/disk) | None | 3.1.4 |

**输出文件**:
```
runs/train/voc_train_exp1/
├── weights/
│   ├── best.pt         # 最佳权重
│   └── last.pt         # 最终权重
├── logs/
│   └── train_*.log     # 训练日志
├── train_config.json   # 训练配置
├── results.png         # 训练曲线
└── confusion_matrix.png
```

**相关模块**:
- `scripts/train_yolov5.py`: 训练流程封装
- `utils/logger.py`: 日志和时间统计

---

### 阶段3: 模型评估 (对应教材 3.1.4节)

**功能**: 计算mAP、生成混淆矩阵和PR曲线

**基础评估**:
```bash
python scripts/evaluate_model.py \
    --yolov5_root /path/to/yolov5 \
    --weights runs/train/voc_train_exp1/weights/best.pt \
    --data data/VOC2012.yaml \
    --img 640 \
    --batch 32 \
    --device 0 \
    --name eval_exp1
```

**调整阈值评估**:
```bash
python scripts/evaluate_model.py \
    --yolov5_root /path/to/yolov5 \
    --weights best.pt \
    --data data/VOC2012.yaml \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --save-txt \
    --save-json \
    --verbose
```

**评估指标说明**:

| 指标 | 含义 | 教材章节 |
|------|------|---------|
| Precision | 精确率 = TP/(TP+FP) | 3.1.4 |
| Recall | 召回率 = TP/(TP+FN) | 3.1.4 |
| mAP@0.5 | IoU=0.5时的平均精度 | 3.1.4 |
| mAP@0.5:0.95 | IoU从0.5到0.95的平均mAP | 3.1.4 |
| F1-Score | 精确率和召回率的调和平均 | 3.1.4 |

**输出文件**:
```
runs/val/eval_exp1/
├── logs/
│   └── eval_*.log
├── confusion_matrix.png    # 混淆矩阵
├── F1_curve.png           # F1曲线
├── P_curve.png            # 精确率曲线
├── R_curve.png            # 召回率曲线
├── PR_curve.png           # PR曲线
└── evaluation_results.json
```

**相关模块**:
- `scripts/evaluate_model.py`: 模型评估

---

### 阶段4: 模型推理 (对应教材 3.1.4节)

**功能**: 批量处理图像/视频,自动生成统计报告

**图像文件夹推理**:
```bash
python scripts/batch_inference.py \
    --yolov5_root /path/to/yolov5 \
    --weights runs/train/voc_train_exp1/weights/best.pt \
    --source data/images/test \
    --img 640 \
    --conf-thres 0.25 \
    --device 0 \
    --name test_infer
```

**视频推理**:
```bash
python scripts/batch_inference.py \
    --yolov5_root /path/to/yolov5 \
    --weights best.pt \
    --source video.mp4 \
    --save-crop \
    --name video_detect
```

**高级功能**:
```bash
# 只检测特定类别(person=0, car=2)
python scripts/batch_inference.py \
    --yolov5_root /path/to/yolov5 \
    --weights best.pt \
    --source images/ \
    --classes 0 2 \
    --hide-labels \
    --class-names person bicycle car motorbike

# 高置信度 + 保存裁剪
python scripts/batch_inference.py \
    --yolov5_root /path/to/yolov5 \
    --weights best.pt \
    --source images/ \
    --conf-thres 0.5 \
    --save-crop \
    --line-thickness 2
```

**输出文件**:
```
runs/detect/test_infer/
├── logs/
│   └── infer_*.log
├── labels/                    # txt格式检测结果
│   ├── image1.txt
│   └── image2.txt
├── crops/                     # 裁剪的检测框
│   ├── person/
│   └── car/
├── image1.jpg                 # 标注后的图像
├── image2.jpg
└── inference_statistics.json  # 统计报告
```

**相关模块**:
- `scripts/batch_inference.py`: 批量推理和统计分析

---

## 🔧 工具模块详解

### logger.py - 日志管理工具

**功能**: 统一日志记录、时间统计、进度显示

**使用示例**:
```python
from utils.logger import setup_logger, TaskTimer, ProgressLogger

# 初始化日志
setup_logger(log_file="logs/my_task.log")

# 任务计时
with TaskTimer("数据处理"):
    # 你的代码
    pass

# 批量处理进度
progress = ProgressLogger(total=1000, task_name="图像处理")
for i in range(1000):
    # 处理单个图像
    progress.update()
progress.finish()
```

**关键类**:
- `TaskTimer`: 任务计时器,自动记录耗时
- `ProgressLogger`: 进度记录器,显示进度和预估剩余时间
- `setup_logger`: 配置日志系统

---

### voc_converter.py - VOC格式转换工具

**功能**: VOC XML → YOLO txt格式转换,坐标归一化

**使用示例**:
```python
from utils.voc_converter import VOCConverter

voc_classes = ['person', 'car', 'bicycle', ...]
converter = VOCConverter(class_names=voc_classes)

# 批量转换
converter.convert_dataset(
    annotations_dir=Path("VOC2012/Annotations"),
    output_dir=Path("labels/train"),
    subset='train'
)
```

**转换规则**:
```
VOC格式: (xmin, ymin, xmax, ymax) - 绝对坐标
    ↓
YOLO格式: (x_center, y_center, width, height) - 归一化坐标

x_center = (xmin + xmax) / (2 × img_width)
y_center = (ymin + ymax) / (2 × img_height)
width = (xmax - xmin) / img_width
height = (ymax - ymin) / img_height
```

**错误检测**:
- ✅ 坐标顺序检查
- ✅ 坐标范围检查
- ✅ 边界框尺寸检查
- ✅ 未知类别检测

---

### dataset_analyzer.py - 数据集分析工具

**功能**: 数据统计、质量检查、可视化分析

**使用示例**:
```python
from utils.dataset_analyzer import DatasetAnalyzer

analyzer = DatasetAnalyzer(class_names=voc_classes)

# 分析数据集
stats = analyzer.analyze_dataset(
    labels_dir=Path("labels/train"),
    image_size=(640, 640),
    subset='train'
)

# 保存报告
analyzer.save_report(Path("reports/train_analysis.json"))
```

**分析内容**:
- 基本统计(图像数、标注数)
- 类别分布和不平衡检测
- 边界框尺寸分布(小/中/大目标)
- 宽高比统计
- 数据质量问题检测

---

## 📊 实验结果示例

### VOC2012数据集统计

| 子集 | 图像数 | 标注数 | 平均标注数/图 |
|------|--------|--------|--------------|
| Train | 5,717 | 13,609 | 2.38 |
| Val | 5,823 | 13,841 | 2.38 |

### 训练结果(YOLOv5s, 100 epochs)

| 指标 | 数值 |
|------|------|
| Precision | 0.847 |
| Recall | 0.783 |
| mAP@0.5 | 0.821 |
| mAP@0.5:0.95 | 0.586 |

### 各类别性能(Top 5)

| 类别 | Precision | Recall | mAP@0.5 |
|------|-----------|--------|---------|
| person | 0.89 | 0.87 | 0.90 |
| car | 0.91 | 0.86 | 0.89 |
| dog | 0.88 | 0.82 | 0.87 |
| bicycle | 0.86 | 0.81 | 0.85 |
| bird | 0.84 | 0.79 | 0.83 |

---

## ⚠️ 常见问题

### 1. CUDA Out of Memory

**原因**: GPU显存不足

**解决方案**:
```bash
# 减小批次大小
--batch 8  # 或 4

# 减小图像尺寸
--img 416  # 或 320

# 使用图像缓存
--cache disk
```

### 2. 数据集下载失败

**原因**: 网络问题或VOC服务器不稳定

**解决方案**:
```bash
# 手动下载数据集
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# 放到指定目录后跳过下载
python scripts/prepare_dataset.py --skip_download
```

### 3. 训练中断续训

**解决方案**:
```bash
# 使用--resume参数
python scripts/train_yolov5.py \
    --yolov5_root /path/to/yolov5 \
    --data data/VOC2012.yaml \
    --resume \
    --name original_exp_name  # 使用原来的实验名称
```

### 4. 模型评估指标偏低

**可能原因和解决方案**:

1. **训练不充分**:
   - 增加epochs: `--epochs 300`
   - 使用更大的模型: `--weights yolov5m.pt`

2. **学习率不合适**:
   - 降低初始学习率: `--lr0 0.005`
   - 使用余弦学习率: `--cos-lr`

3. **数据增强不足**:
   - 使用多尺度训练: `--multi-scale`
   - 增加数据增强: 修改hyp.yaml

4. **类别不平衡**:
   - 使用图像加权采样: 需要修改YOLOv5源码

---



## 🔗 参考资源

### 官方文档
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [YOLOv5 官方文档](https://docs.ultralytics.com/)
- [PASCAL VOC 官网](http://host.robots.ox.ac.uk/pascal/VOC/)

### 论文
- YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- YOLOv1: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016
- PASCAL VOC: Everingham et al., "The Pascal Visual Object Classes Challenge", IJCV 2010

### 教程
- [YOLOv5 训练自定义数据集教程](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [目标检测评估指标详解](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

---

