"""
YOLOv5检测器封装模块
提供统一的目标检测接口,可复用Task1的YOLOv5模型或使用官方预训练模型

主要功能:
- 加载YOLOv5模型(支持本地训练模型或官方模型)
- 图像检测(单张/批量)
- 坐标格式转换(YOLO格式 -> MOT格式)
- 类别过滤(仅保留行人类别)
"""

import torch
import cv2
import numpy as np
import logging
from typing import List, Union, Tuple
import time
import os
import sys

# 配置日志
logger = logging.getLogger(__name__)


class YOLOv5Detector:
    """
    YOLOv5检测器封装类

    提供统一接口用于:
    - 加载YOLOv5模型
    - 执行目标检测
    - 格式转换和后处理
    - 类别过滤
    """

    def __init__(
            self,
            model_path: str = None,
            conf_thresh: float = 0.5,
            iou_thresh: float = 0.45,
            device: str = 'cuda',
            img_size: int = 640,
            use_official: bool = True
    ):
        """
        初始化检测器

        Args:
            model_path: 模型路径(本地.pt文件或官方模型名如'yolov5s')
            conf_thresh: 置信度阈值
            iou_thresh: NMS的IoU阈值
            device: 计算设备 ('cuda' or 'cpu')
            img_size: 输入图像尺寸
            use_official: 是否使用官方torch hub加载
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        if use_official:
            self.model = self._load_official_model(model_path or 'yolov5s')
        else:
            self.model = self._load_local_model(model_path)

        self.model.to(self.device)
        self.model.eval()

        # COCO数据集类别名称(YOLOv5官方训练使用)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # 行人类别ID(COCO数据集中person为0)
        self.person_class_id = 0

        logger.info(
            f"YOLOv5检测器初始化完成: device={self.device}, "
            f"conf={conf_thresh}, iou={iou_thresh}, size={img_size}"
        )

    def _load_official_model(self, model_name: str):
        """
        加载官方预训练模型(通过torch.hub)

        Args:
            model_name: 模型名称 ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')

        Returns:
            model: YOLOv5模型
        """
        try:
            logger.info(f"加载YOLOv5官方模型: {model_name}")
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            logger.info(f"成功加载官方模型: {model_name}")
            return model
        except Exception as e:
            logger.error(f"加载官方模型失败: {e}")
            logger.info("尝试从本地缓存加载...")

            # 尝试从本地YOLOv5仓库加载
            yolov5_path = os.path.join(os.path.dirname(__file__), '../../yolov5')
            if os.path.exists(yolov5_path):
                sys.path.insert(0, yolov5_path)
                model = torch.hub.load(yolov5_path, model_name, source='local')
                logger.info(f"从本地仓库加载成功: {yolov5_path}")
                return model
            else:
                raise RuntimeError(f"无法加载YOLOv5模型: {model_name}")

    def _load_local_model(self, model_path: str):
        """
        加载本地训练的模型

        Args:
            model_path: 模型文件路径(.pt)

        Returns:
            model: YOLOv5模型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            logger.info(f"加载本地模型: {model_path}")
            model = torch.load(model_path, map_location=self.device)

            # 兼容不同的保存格式
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'ema' in model:
                    model = model['ema']

            logger.info(f"成功加载本地模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载本地模型失败: {e}")
            raise

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        图像预处理

        Args:
            img: BGR格式图像

        Returns:
            tensor: 预处理后的张量
        """
        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 缩放到模型输入尺寸(保持宽高比)
        h, w = img_rgb.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h))

        # Padding到正方形
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w

        img_padded = cv2.copyMakeBorder(
            img_resized,
            0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )

        # 归一化到[0, 1]
        img_norm = img_padded.astype(np.float32) / 255.0

        # 转换为Tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)

        # 增加batch维度
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, scale, (pad_w, pad_h)

    def _postprocess(
            self,
            predictions,
            img_shape: Tuple[int, int],
            scale: float,
            padding: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        后处理:NMS、坐标还原、类别过滤

        Args:
            predictions: 模型原始输出
            img_shape: 原始图像尺寸 (height, width)
            scale: 缩放比例
            padding: padding尺寸 (pad_w, pad_h)

        Returns:
            detections: 检测结果列表 [[left, top, width, height, conf, class], ...]
        """
        # 处理YOLOv5输出格式
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # 应用NMS
        predictions = self._non_max_suppression(
            predictions,
            conf_thres=self.conf_thresh,
            iou_thres=self.iou_thresh
        )

        detections = []

        for det in predictions:
            if det is None or len(det) == 0:
                continue

            # 坐标还原到原始图像
            # det: [x1, y1, x2, y2, conf, class]
            det[:, :4] = self._scale_coords(
                (self.img_size, self.img_size),
                det[:, :4],
                img_shape,
                scale,
                padding
            )

            # 转换格式: xyxy -> tlwh
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = xyxy
                left = x1.item()
                top = y1.item()
                width = (x2 - x1).item()
                height = (y2 - y1).item()
                confidence = conf.item()
                class_id = int(cls.item())

                # 仅保留行人类别
                if class_id == self.person_class_id:
                    detections.append([left, top, width, height, confidence, class_id])

        return detections

    def _non_max_suppression(
            self,
            prediction: torch.Tensor,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            max_det: int = 300
    ):
        """
        执行非极大值抑制(NMS)

        Args:
            prediction: 模型输出 (batch, num_boxes, 85)
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            max_det: 最大检测数

        Returns:
            output: NMS后的检测结果
        """
        # 简化版NMS实现
        output = []

        for x in prediction:
            # 置信度过滤
            x = x[x[:, 4] > conf_thres]

            if not x.shape[0]:
                output.append(torch.zeros((0, 6)))
                continue

            # 计算类别得分
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # 获取最高得分的类别
            box = x[:, :4]
            conf, j = x[:, 5:].max(1, keepdim=True)

            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # 按置信度排序
            x = x[x[:, 4].argsort(descending=True)[:max_det]]

            # 按类别执行NMS
            c = x[:, 5:6] * 4096  # 类别偏移
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self._torchvision_nms(boxes, scores, iou_thres)

            output.append(x[i])

        return output

    def _torchvision_nms(
            self,
            boxes: torch.Tensor,
            scores: torch.Tensor,
            iou_threshold: float
    ):
        """
        调用torchvision的NMS

        Args:
            boxes: 边界框 (N, 4)
            scores: 置信度 (N,)
            iou_threshold: IoU阈值

        Returns:
            keep_indices: 保留的索引
        """
        try:
            from torchvision.ops import nms
            return nms(boxes, scores, iou_threshold)
        except ImportError:
            logger.warning("torchvision未安装,使用简化NMS")
            return self._simple_nms(boxes, scores, iou_threshold)

    def _simple_nms(
            self,
            boxes: torch.Tensor,
            scores: torch.Tensor,
            iou_threshold: float
    ):
        """
        简化的NMS实现(numpy版本)

        Args:
            boxes: 边界框
            scores: 置信度
            iou_threshold: IoU阈值

        Returns:
            keep_indices: 保留的索引
        """
        keep = []

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()

        order = scores_np.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算IoU
            xx1 = np.maximum(boxes_np[i, 0], boxes_np[order[1:], 0])
            yy1 = np.maximum(boxes_np[i, 1], boxes_np[order[1:], 1])
            xx2 = np.minimum(boxes_np[i, 2], boxes_np[order[1:], 2])
            yy2 = np.minimum(boxes_np[i, 3], boxes_np[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
            area_others = (boxes_np[order[1:], 2] - boxes_np[order[1:], 0]) * \
                          (boxes_np[order[1:], 3] - boxes_np[order[1:], 1])

            iou = inter / (area_i + area_others - inter + 1e-6)

            # 保留IoU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return torch.tensor(keep, dtype=torch.long)

    def _scale_coords(
            self,
            img1_shape: Tuple[int, int],
            coords: torch.Tensor,
            img0_shape: Tuple[int, int],
            scale: float,
            padding: Tuple[int, int]
    ) -> torch.Tensor:
        """
        坐标从缩放后的图像还原到原始图像

        Args:
            img1_shape: 缩放后图像尺寸
            coords: 坐标 (N, 4)
            img0_shape: 原始图像尺寸 (height, width)
            scale: 缩放比例
            padding: padding尺寸 (pad_w, pad_h)

        Returns:
            coords: 还原后的坐标
        """
        pad_w, pad_h = padding

        # 去除padding
        coords[:, [0, 2]] -= pad_w
        coords[:, [1, 3]] -= pad_h

        # 还原缩放
        coords[:, :4] /= scale

        # 裁剪到图像边界
        coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
        coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])

        return coords

    def detect(self, img: np.ndarray) -> List[np.ndarray]:
        """
        执行目标检测

        Args:
            img: 输入图像(BGR格式)

        Returns:
            detections: 检测结果 [[left, top, width, height, conf, class], ...]
        """
        start_time = time.time()

        h, w = img.shape[:2]

        # 预处理
        img_tensor, scale, padding = self._preprocess(img)
        img_tensor = img_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # 后处理
        detections = self._postprocess(predictions, (h, w), scale, padding)

        elapsed_time = time.time() - start_time
        logger.debug(
            f"检测完成: 检测到{len(detections)}个目标, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return detections


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试YOLOv5检测器")
    print("=" * 50)

    # 初始化检测器(使用官方预训练模型)
    print("\n初始化检测器...")
    detector = YOLOv5Detector(
        model_path='yolov5s',
        conf_thresh=0.5,
        iou_thresh=0.45,
        device='cpu',  # 测试使用CPU
        use_official=True
    )

    # 创建测试图像
    print("\n创建测试图像...")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 绘制一些矩形(模拟目标)
    cv2.rectangle(test_img, (100, 100), (200, 300), (255, 0, 0), -1)
    cv2.rectangle(test_img, (350, 150), (450, 350), (0, 255, 0), -1)

    # 执行检测
    print("\n执行检测...")
    detections = detector.detect(test_img)

    print(f"\n检测结果:")
    print(f"检测到目标数: {len(detections)}")

    for i, det in enumerate(detections):
        left, top, width, height, conf, cls = det
        print(f"\n目标{i + 1}:")
        print(f"  位置: ({left:.1f}, {top:.1f})")
        print(f"  尺寸: {width:.1f} x {height:.1f}")
        print(f"  置信度: {conf:.3f}")
        print(f"  类别: {detector.class_names[int(cls)]}")

    print("\n测试完成!")