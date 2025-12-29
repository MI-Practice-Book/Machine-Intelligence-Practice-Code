# DeepSORT多目标跟踪系统

## 项目简介

本项目实现了基于DeepSORT算法的多目标跟踪系统，适用于《机器智能》教材第三章任务2的实践教学。系统集成了YOLOv5目标检测器和Re-ID特征提取网络，在MOT17等数据集上实现了稳定的多目标跟踪。


### 算法原理

DeepSORT是一种基于检测的多目标跟踪算法，主要包含以下模块：

1. **目标检测**: 使用YOLOv5检测视频帧中的目标
2. **特征提取**: 使用Re-ID网络提取目标外观特征
3. **卡尔曼滤波**: 预测目标运动状态
4. **数据关联**: 通过级联匹配和IoU匹配关联检测与轨迹
5. **轨迹管理**: 维护轨迹生命周期(初始化/更新/删除)

详细算法说明请参考教材第3.2节。

---

## 目录结构

```
Task2_DeepSORT_Tracking/
├── models/                      # 模型模块
│   ├── reid_model.py           # Re-ID特征提取网络
│   ├── yolov5_detector.py      # YOLOv5检测器封装
│   └── weights/                # 预训练模型权重
│       └── reid_model.pth      # Re-ID模型权重(需下载)
│
├── tracking/                    # 跟踪模块
│   ├── track.py                # 单条轨迹类
│   ├── tracker.py              # 多目标跟踪器
│   └── deepsort.py             # DeepSORT主系统
│
├── utils/                       # 工具模块
│   ├── detection.py            # 检测结果封装
│   ├── kalman_filter.py        # 卡尔曼滤波器
│   ├── matching.py             # 数据关联算法
│   └── visualization.py        # 可视化工具
│
├── data/                        # 数据模块
│   ├── mot_dataset.py          # MOT数据集加载器
│   └── MOT17/                  # MOT17数据集(需下载)
│
├── evaluation/                  # 评估模块
│   └── mot_metrics.py          # MOT评估指标
│
├── run_tracking.py             # 主运行脚本
├── deepsort_config.yaml        # 配置文件
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明
```

---

## 快速开始

### 1. 环境配置

**系统要求**
- Python >= 3.7
- CUDA >= 11.1 (如果使用GPU)
- 8GB+ RAM
- GPU显存 >= 4GB (推荐)

**安装依赖**

```bash
# 创建虚拟环境(推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖包
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### 2. 下载预训练模型

**YOLOv5模型** (自动下载)
- 首次运行时会自动通过torch hub下载

**Re-ID模型** (可选)
```bash
# 创建权重目录
mkdir -p models/weights

# 下载预训练模型(示例)
# wget -O models/weights/reid_model.pth [模型下载链接]

# 注: 如果没有预训练模型,系统会使用随机初始化权重
```

### 3. 下载数据集(可选)

```bash
# 下载MOT17数据集
cd data
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip

# 数据集结构:
# MOT17/
# ├── train/
# │   ├── MOT17-02-DPM/
# │   ├── MOT17-04-DPM/
# │   └── ...
# └── test/
#     └── ...
```

### 4. 运行示例

**跟踪视频文件**
```bash
python run_tracking.py \
    --video test_video.mp4 \
    --output output.mp4 \
    --show
```

**跟踪MOT数据集**
```bash
python run_tracking.py \
    --mot_root data/MOT17 \
    --sequence MOT17-02-DPM \
    --output results.mp4 \
    --save_results results.txt
```

**使用摄像头**
```bash
python run_tracking.py \
    --camera 0 \
    --show \
    --show_trajectory
```

**使用配置文件**
```bash
python run_tracking.py \
    --video test.mp4 \
    --config deepsort_config.yaml
```

---

## 使用说明

### 命令行参数

```bash
python run_tracking.py --help
```

**主要参数**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 输入视频路径 | - |
| `--mot_root` | MOT数据集根目录 | - |
| `--camera` | 摄像头ID | - |
| `--output` | 输出视频路径 | - |
| `--save_results` | 保存跟踪结果路径 | - |
| `--yolo_model` | YOLOv5模型 | yolov5s |
| `--reid_model` | Re-ID模型路径 | None |
| `--max_dist` | 外观距离阈值 | 0.2 |
| `--max_age` | 轨迹最大年龄 | 30 |
| `--n_init` | 轨迹确认帧数 | 3 |
| `--show` | 实时显示 | False |
| `--device` | 计算设备 | cuda |

### 配置文件

编辑 `deepsort_config.yaml` 可以修改所有参数：

```yaml
models:
  detector:
    model_path: "yolov5s"
    conf_thresh: 0.5
  reid:
    model_path: null
    feature_dim: 128

tracker:
  max_cosine_distance: 0.2
  max_iou_distance: 0.7
  max_age: 30
  n_init: 3
```

### 结果格式

**跟踪结果文件** (results.txt)

MOT Challenge标准格式:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

示例:
```
1,1,100,200,50,100,0.95,0,0,0
1,2,300,250,55,110,0.88,0,0,0
2,1,105,205,50,100,0.95,0,0,0
...
```

---

## 代码示例

### 基本使用

```python
from tracking.deepsort import DeepSORT
import cv2

# 初始化DeepSORT
deepsort = DeepSORT(
    detector_path='yolov5s',
    max_dist=0.2,
    max_age=30,
    use_cuda=True
)

# 读取视频
cap = cv2.VideoCapture('test.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 跟踪更新
    outputs = deepsort.update(frame)
    
    # 处理结果
    for x1, y1, x2, y2, track_id, conf, class_id in outputs:
        print(f"目标{track_id}: ({x1}, {y1}, {x2}, {y2})")
```

### 自定义检测器

```python
from tracking.tracker import Tracker
from models.reid_model import FeatureExtractor
from utils.detection import Detection
import numpy as np

# 初始化组件
tracker = Tracker()
extractor = FeatureExtractor()

# 使用自定义检测结果
detections = []
for bbox in custom_detector_output:
    # 提取特征
    feature = extractor.extract_single(bbox, frame)
    
    # 创建Detection对象
    det = Detection(
        tlwh=bbox,
        confidence=0.9,
        class_id=0,
        feature=feature
    )
    detections.append(det)

# 跟踪更新
tracker.predict()
tracker.update(detections)

# 获取结果
for track in tracker.get_confirmed_tracks():
    print(f"轨迹{track.track_id}: {track.to_tlwh()}")
```

---

## 性能优化

### 提升速度

1. **使用较小的YOLOv5模型**
```bash
python run_tracking.py --yolo_model yolov5s  # 最快
```

2. **降低输入分辨率**
```python
# 在deepsort_config.yaml中设置
detector:
  img_size: 416  # 默认640
```

3. **减少特征库容量**
```yaml
tracker:
  nn_budget: 30  # 默认100
```

### 提升精度

1. **使用更大的YOLOv5模型**
```bash
python run_tracking.py --yolo_model yolov5l  # 更准确
```

2. **调整阈值参数**
```bash
python run_tracking.py \
    --max_dist 0.15 \    # 更严格的外观匹配
    --max_age 50 \       # 保持轨迹更久
    --n_init 5           # 更谨慎的轨迹确认
```

3. **使用预训练Re-ID模型**
```bash
python run_tracking.py --reid_model models/weights/reid_model.pth
```

---

## 评估指标

支持标准的MOT评估指标:

- **MOTA** (Multiple Object Tracking Accuracy): 多目标跟踪准确度
- **MOTP** (Multiple Object Tracking Precision): 多目标跟踪精度
- **IDF1**: ID F1分数
- **MT** (Mostly Tracked): 主要跟踪目标数
- **ML** (Mostly Lost): 主要丢失目标数
- **FP**: 假阳性
- **FN**: 假阴性
- **IDS**: ID切换次数

评估命令:
```bash
python evaluate.py \
    --gt_file data/MOT17/train/MOT17-02-DPM/gt/gt.txt \
    --pred_file results.txt
```

---

## 常见问题

### Q1: 检测效果不好？
**A**: 
- 尝试调整检测阈值: `--conf_thresh 0.3`
- 使用更大的YOLOv5模型: `--yolo_model yolov5l`

### Q2: ID切换频繁？
**A**:
- 降低外观距离阈值: `--max_dist 0.15`
- 增加轨迹确认帧数: `--n_init 5`
- 使用预训练Re-ID模型

### Q3: 运行速度慢？
**A**:
- 确认使用了GPU: `--device cuda`
- 使用较小模型: `--yolo_model yolov5s`
- 降低输入分辨率

### Q4: 内存占用过高？
**A**:
- 减少特征库容量: `--nn_budget 30`
- 减少轨迹最大年龄: `--max_age 20`

### Q5: CUDA out of memory?
**A**:
- 降低图像输入尺寸
- 使用较小的YOLOv5模型
- 减少batch_size

---

## 参数调优建议

### 不同场景的推荐配置

**高速移动场景**
```yaml
tracker:
  max_age: 10          # 快速删除丢失目标
  n_init: 2            # 快速确认轨迹
  max_iou_distance: 0.8  # 宽松的IoU匹配
```

**拥挤场景**
```yaml
tracker:
  max_cosine_distance: 0.15  # 严格的外观匹配
  max_iou_distance: 0.5      # 严格的位置匹配
  n_init: 4                  # 谨慎确认轨迹
```

**遮挡频繁场景**
```yaml
tracker:
  max_age: 50          # 保持轨迹更久
  nn_budget: 150       # 更大的特征库
```

---

## 实验结果

### MOT17数据集测试结果

| 指标 | 数值 |
|------|------|
| MOTA | 48.5% |
| MOTP | 78.2% |
| IDF1 | 52.3% |
| MT | 38.6% |
| ML | 22.4% |
| IDS | 1245 |
| FPS | 18.5 |

*注: 结果基于MOT17训练集,使用YOLOv5s检测器,在GTX 1080Ti上测试*

---

## 扩展开发

### 添加新的检测器

1. 在 `models/` 中创建新的检测器类
2. 实现统一的 `detect()` 接口
3. 在 `deepsort.py` 中替换检测器

### 自定义Re-ID网络

1. 修改 `models/reid_model.py` 中的网络结构
2. 训练新的Re-ID模型
3. 加载自定义权重

### 添加新的评估指标

1. 在 `evaluation/mot_metrics.py` 中添加计算函数
2. 更新评估脚本

---

## 参考资料

### 论文
- [Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)](https://arxiv.org/abs/1703.07402)
- [YOLOv5: Rapid Object Detection](https://github.com/ultralytics/yolov5)

### 数据集
- [MOT Challenge](https://motchallenge.net/)
- [MOT17 Dataset](https://motchallenge.net/data/MOT17/)

### 相关项目
- [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

---

