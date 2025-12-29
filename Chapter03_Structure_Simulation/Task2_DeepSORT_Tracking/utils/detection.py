"""
检测结果封装模块
提供Detection类,封装目标检测结果的边界框、置信度、类别等信息
"""

import numpy as np
import logging
from typing import Optional, Tuple

# 配置日志
logger = logging.getLogger(__name__)


class Detection:
    """
    目标检测结果封装类

    该类封装单个检测目标的所有信息,包括:
    - 边界框坐标(支持多种格式)
    - 检测置信度
    - 目标类别
    - 外观特征向量(用于Re-ID)

    Attributes:
        tlwh: 边界框坐标 [left, top, width, height]
        confidence: 检测置信度 [0, 1]
        class_id: 目标类别ID
        feature: 外观特征向量(128维或256维)
    """

    def __init__(
            self,
            tlwh: np.ndarray,
            confidence: float,
            class_id: int = 0,
            feature: Optional[np.ndarray] = None
    ):
        """
        初始化检测结果

        Args:
            tlwh: 边界框 [left, top, width, height]
            confidence: 检测置信度
            class_id: 目标类别(默认0表示行人)
            feature: 外观特征向量(可选)
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.feature = feature

        # 验证输入有效性
        self._validate()

        logger.debug(
            f"创建Detection: bbox={self.tlwh}, conf={self.confidence:.3f}, "
            f"class={self.class_id}"
        )

    def _validate(self):
        """验证检测结果的有效性"""
        if len(self.tlwh) != 4:
            raise ValueError(f"边界框必须是4维向量,当前: {self.tlwh}")

        if self.tlwh[2] <= 0 or self.tlwh[3] <= 0:
            raise ValueError(f"边界框宽高必须>0,当前: {self.tlwh[2:]}")

        if not 0 <= self.confidence <= 1:
            logger.warning(f"置信度超出[0,1]范围: {self.confidence}")

    def to_tlbr(self) -> np.ndarray:
        """
        转换为[left, top, right, bottom]格式

        Returns:
            tlbr格式的边界框
        """
        ret = self.tlwh.copy()
        ret[2] = ret[0] + ret[2]  # right = left + width
        ret[3] = ret[1] + ret[3]  # bottom = top + height
        return ret

    def to_xyah(self) -> np.ndarray:
        """
        转换为[center_x, center_y, aspect_ratio, height]格式
        用于卡尔曼滤波器的观测向量

        Returns:
            xyah格式 [cx, cy, aspect_ratio, height]
        """
        ret = self.tlwh.copy()
        ret[0] = ret[0] + ret[2] / 2  # center_x = left + width/2
        ret[1] = ret[1] + ret[3] / 2  # center_y = top + height/2
        ret[2] = ret[2] / ret[3]  # aspect_ratio = width/height
        return ret

    def get_area(self) -> float:
        """
        计算边界框面积

        Returns:
            面积(像素平方)
        """
        return float(self.tlwh[2] * self.tlwh[3])

    def set_feature(self, feature: np.ndarray):
        """
        设置外观特征向量

        Args:
            feature: Re-ID网络提取的特征向量
        """
        if feature is None:
            self.feature = None
            return

        # 确保是一维向量
        feature = np.asarray(feature).flatten()

        # L2归一化
        norm = np.linalg.norm(feature)
        if norm > 0:
            self.feature = feature / norm
        else:
            logger.warning("特征向量范数为0,无法归一化")
            self.feature = feature

        logger.debug(f"设置特征向量: 维度={len(self.feature)}")


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    计算两个边界框的交并比(IoU)

    Args:
        bbox1: 边界框1 [left, top, width, height]
        bbox2: 边界框2 [left, top, width, height]

    Returns:
        IoU值 [0, 1]
    """
    # 转换为[left, top, right, bottom]
    box1 = bbox1.copy()
    box1[2:] = box1[:2] + box1[2:]

    box2 = bbox2.copy()
    box2[2:] = box2[:2] + box2[2:]

    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算并集区域
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    批量计算IoU矩阵(向量化实现)

    Args:
        bboxes1: N个边界框 [N, 4] (tlwh格式)
        bboxes2: M个边界框 [M, 4] (tlwh格式)

    Returns:
        IoU矩阵 [N, M]
    """
    # 转换为tlbr格式
    boxes1 = bboxes1.copy()
    boxes1[:, 2:] = boxes1[:, :2] + boxes1[:, 2:]

    boxes2 = bboxes2.copy()
    boxes2[:, 2:] = boxes2[:, :2] + boxes2[:, 2:]

    # 计算交集
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = np.maximum(0, rb - lt)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # 计算并集
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - intersection

    # 避免除零
    iou_matrix = np.zeros_like(intersection)
    mask = union > 0
    iou_matrix[mask] = intersection[mask] / union[mask]

    return iou_matrix


def convert_bbox_format(
        bbox: np.ndarray,
        from_format: str,
        to_format: str,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None
) -> np.ndarray:
    """
    边界框格式转换工具函数

    支持的格式:
    - 'tlwh': [left, top, width, height]
    - 'tlbr': [left, top, right, bottom]
    - 'xywh': [center_x, center_y, width, height]
    - 'xywh_norm': 归一化的xywh格式(YOLO)

    Args:
        bbox: 输入边界框
        from_format: 输入格式
        to_format: 输出格式
        img_width: 图像宽度(归一化转换时需要)
        img_height: 图像高度(归一化转换时需要)

    Returns:
        转换后的边界框
    """
    bbox = np.asarray(bbox, dtype=np.float32).copy()

    # 统一转换为tlwh中间格式
    if from_format == 'tlwh':
        tlwh = bbox
    elif from_format == 'tlbr':
        tlwh = bbox.copy()
        tlwh[2] = bbox[2] - bbox[0]
        tlwh[3] = bbox[3] - bbox[1]
    elif from_format == 'xywh':
        tlwh = bbox.copy()
        tlwh[0] = bbox[0] - bbox[2] / 2
        tlwh[1] = bbox[1] - bbox[3] / 2
    elif from_format == 'xywh_norm':
        if img_width is None or img_height is None:
            raise ValueError("归一化格式转换需要提供图像尺寸")
        tlwh = bbox.copy()
        tlwh[0] = (bbox[0] - bbox[2] / 2) * img_width
        tlwh[1] = (bbox[1] - bbox[3] / 2) * img_height
        tlwh[2] = bbox[2] * img_width
        tlwh[3] = bbox[3] * img_height
    else:
        raise ValueError(f"不支持的输入格式: {from_format}")

    # 从tlwh转换为目标格式
    if to_format == 'tlwh':
        return tlwh
    elif to_format == 'tlbr':
        result = tlwh.copy()
        result[2] = tlwh[0] + tlwh[2]
        result[3] = tlwh[1] + tlwh[3]
        return result
    elif to_format == 'xywh':
        result = tlwh.copy()
        result[0] = tlwh[0] + tlwh[2] / 2
        result[1] = tlwh[1] + tlwh[3] / 2
        return result
    elif to_format == 'xywh_norm':
        if img_width is None or img_height is None:
            raise ValueError("归一化格式转换需要提供图像尺寸")
        result = tlwh.copy()
        result[0] = (tlwh[0] + tlwh[2] / 2) / img_width
        result[1] = (tlwh[1] + tlwh[3] / 2) / img_height
        result[2] = tlwh[2] / img_width
        result[3] = tlwh[3] / img_height
        return result
    else:
        raise ValueError(f"不支持的输出格式: {to_format}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试Detection类")
    print("=" * 50)

    # 创建检测结果
    det = Detection(
        tlwh=[100, 200, 50, 100],
        confidence=0.85,
        class_id=0
    )

    print(f"原始tlwh: {det.tlwh}")
    print(f"转换tlbr: {det.to_tlbr()}")
    print(f"转换xyah: {det.to_xyah()}")
    print(f"边界框面积: {det.get_area()}")

    # 测试特征设置
    feature = np.random.randn(128)
    det.set_feature(feature)
    print(f"特征向量范数: {np.linalg.norm(det.feature):.6f}")

    print("\n" + "=" * 50)
    print("测试IoU计算")
    print("=" * 50)

    bbox1 = np.array([100, 100, 50, 50])
    bbox2 = np.array([120, 120, 50, 50])
    print(f"Box1: {bbox1}")
    print(f"Box2: {bbox2}")
    print(f"IoU: {iou(bbox1, bbox2):.4f}")

    # 测试批量IoU
    bboxes1 = np.array([[100, 100, 50, 50], [200, 200, 60, 60]])
    bboxes2 = np.array([[120, 120, 50, 50], [190, 190, 70, 70], [300, 300, 40, 40]])
    iou_matrix = iou_batch(bboxes1, bboxes2)
    print(f"\nIoU矩阵:\n{iou_matrix}")

    print("\n" + "=" * 50)
    print("测试格式转换")
    print("=" * 50)

    bbox_tlwh = np.array([100, 200, 50, 100])
    print(f"tlwh: {bbox_tlwh}")

    bbox_tlbr = convert_bbox_format(bbox_tlwh, 'tlwh', 'tlbr')
    print(f"tlbr: {bbox_tlbr}")

    bbox_xywh = convert_bbox_format(bbox_tlwh, 'tlwh', 'xywh')
    print(f"xywh: {bbox_xywh}")

    bbox_norm = convert_bbox_format(
        bbox_tlwh, 'tlwh', 'xywh_norm',
        img_width=1920, img_height=1080
    )
    print(f"xywh_norm: {bbox_norm}")

    print("\n测试完成!")