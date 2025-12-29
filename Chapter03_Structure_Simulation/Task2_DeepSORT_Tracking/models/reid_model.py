"""
Re-ID特征提取模块
实现行人重识别(Person Re-Identification)网络,用于提取目标外观特征

网络架构:
- 输入: 128×64 RGB图像
- 主干: 轻量级卷积网络(6层卷积 + 2层全连接)
- 输出: 128维或256维特征向量(L2归一化)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Union, List, Tuple
import time
import os

# 配置日志
logger = logging.getLogger(__name__)


class ReIDNetwork(nn.Module):
    """
    轻量级Re-ID网络

    用于从检测框中提取外观特征向量,支持跨帧目标重识别
    网络在大规模行人数据集上预训练,学习对光照、视角、姿态变化鲁棒的表示
    """

    def __init__(self, feature_dim: int = 128):
        """
        初始化网络结构

        Args:
            feature_dim: 输出特征维度(128或256)
        """
        super(ReIDNetwork, self).__init__()

        self.feature_dim = feature_dim

        # 卷积层(特征提取)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, feature_dim)

        logger.info(f"ReID网络初始化完成: 特征维度={feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, 128, 64)

        Returns:
            features: 特征向量 (B, feature_dim)
        """
        # 卷积特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 64×32

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 32×16

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 16×8

        x = F.relu(self.bn4(self.conv4(x)))

        x = F.relu(self.bn5(self.conv5(x)))

        x = F.relu(self.bn6(self.conv6(x)))

        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # L2归一化
        x = F.normalize(x, p=2, dim=1)

        return x


class FeatureExtractor:
    """
    特征提取器封装类

    提供统一接口用于:
    - 加载预训练模型
    - 图像预处理
    - 批量特征提取
    - CPU/GPU设备管理
    """

    def __init__(
            self,
            model_path: str = None,
            device: str = 'cuda',
            feature_dim: int = 128,
            use_cuda: bool = True
    ):
        """
        初始化特征提取器

        Args:
            model_path: 预训练模型路径(.pth文件)
            device: 计算设备 ('cuda' or 'cpu')
            feature_dim: 特征维度
            use_cuda: 是否使用GPU
        """
        self.feature_dim = feature_dim
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # 初始化网络
        self.model = ReIDNetwork(feature_dim=feature_dim)
        self.model.to(self.device)

        # 加载预训练权重
        if model_path is not None and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            logger.warning(
                f"未找到预训练模型: {model_path}, 使用随机初始化权重"
            )

        # 设为评估模式
        self.model.eval()

        # 图像预处理参数(ImageNet标准)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.input_size = (64, 128)  # (width, height)

        logger.info(
            f"特征提取器初始化完成: device={self.device}, "
            f"feature_dim={feature_dim}"
        )

    def _load_weights(self, model_path: str):
        """
        加载预训练权重

        Args:
            model_path: 模型文件路径
        """
        try:
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        图像预处理

        Args:
            img: BGR格式图像

        Returns:
            tensor: 预处理后的张量 (1, 3, 128, 64)
        """
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 缩放到输入尺寸
        img = cv2.resize(img, self.input_size)

        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0

        # 标准化(减均值除标准差)
        img = (img - self.mean) / self.std

        # 转换为Tensor: (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # 增加batch维度
        img = img.unsqueeze(0)

        return img

    def _preprocess_batch(self, imgs: List[np.ndarray]) -> torch.Tensor:
        """
        批量图像预处理

        Args:
            imgs: 图像列表

        Returns:
            tensor: 批量张量 (B, 3, 128, 64)
        """
        batch = []
        for img in imgs:
            tensor = self._preprocess(img)
            batch.append(tensor)

        return torch.cat(batch, dim=0)

    def extract_single(
            self,
            bbox: np.ndarray,
            img: np.ndarray
    ) -> np.ndarray:
        """
        从单个检测框提取特征

        Args:
            bbox: 边界框 [left, top, width, height]
            img: 原始图像(BGR格式)

        Returns:
            feature: 特征向量 (feature_dim,)
        """
        start_time = time.time()

        # 裁剪目标区域
        left, top, width, height = map(int, bbox)
        crop = img[top:top + height, left:left + width]

        if crop.size == 0:
            logger.warning(f"裁剪区域为空: bbox={bbox}")
            return np.zeros(self.feature_dim, dtype=np.float32)

        # 预处理
        input_tensor = self._preprocess(crop).to(self.device)

        # 前向传播
        with torch.no_grad():
            feature = self.model(input_tensor)

        # 转换为numpy
        feature = feature.cpu().numpy().flatten()

        elapsed_time = time.time() - start_time
        logger.debug(
            f"提取单个特征: bbox=({left},{top},{width},{height}), "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return feature

    def extract_batch(
            self,
            bboxes: List[np.ndarray],
            img: np.ndarray
    ) -> np.ndarray:
        """
        批量提取特征(推荐,效率更高)

        Args:
            bboxes: 边界框列表 [(left, top, width, height), ...]
            img: 原始图像(BGR格式)

        Returns:
            features: 特征矩阵 (N, feature_dim)
        """
        start_time = time.time()

        if len(bboxes) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # 裁剪所有目标
        crops = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            left, top, width, height = map(int, bbox)
            crop = img[top:top + height, left:left + width]

            if crop.size > 0:
                crops.append(crop)
                valid_indices.append(i)
            else:
                logger.warning(f"裁剪区域{i}为空: bbox={bbox}")

        if len(crops) == 0:
            return np.zeros((len(bboxes), self.feature_dim), dtype=np.float32)

        # 批量预处理
        input_batch = self._preprocess_batch(crops).to(self.device)

        # 批量前向传播
        with torch.no_grad():
            features = self.model(input_batch)

        # 转换为numpy
        features = features.cpu().numpy()

        # 处理无效区域
        result = np.zeros((len(bboxes), self.feature_dim), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            result[idx] = features[i]

        elapsed_time = time.time() - start_time
        logger.debug(
            f"批量提取特征: N={len(bboxes)}, valid={len(crops)}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return result

    def __call__(
            self,
            bboxes: Union[np.ndarray, List[np.ndarray]],
            img: np.ndarray
    ) -> np.ndarray:
        """
        便捷调用接口

        Args:
            bboxes: 单个边界框或边界框列表
            img: 原始图像

        Returns:
            features: 特征向量或特征矩阵
        """
        if isinstance(bboxes, np.ndarray) and bboxes.ndim == 1:
            # 单个边界框
            return self.extract_single(bboxes, img)
        else:
            # 多个边界框
            return self.extract_batch(bboxes, img)


def create_mock_model(save_path: str, feature_dim: int = 128):
    """
    创建模拟预训练模型(用于测试)

    Args:
        save_path: 保存路径
        feature_dim: 特征维度
    """
    model = ReIDNetwork(feature_dim=feature_dim)
    torch.save(model.state_dict(), save_path)
    logger.info(f"模拟模型已保存: {save_path}")


def compute_cosine_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    计算两个特征向量的余弦距离

    Args:
        features1: 特征向量1
        features2: 特征向量2

    Returns:
        distance: 余弦距离 [0, 2]
    """
    # 余弦距离 = 1 - 余弦相似度
    similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
    )
    return 1.0 - similarity


def compute_euclidean_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    计算两个特征向量的欧氏距离

    Args:
        features1: 特征向量1
        features2: 特征向量2

    Returns:
        distance: 欧氏距离
    """
    return np.linalg.norm(features1 - features2)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试ReID网络")
    print("=" * 50)

    # 创建模拟模型
    model_dir = "/home/claude/Task2_DeepSORT_Tracking/models/weights"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "reid_model.pth")

    if not os.path.exists(model_path):
        print("创建模拟预训练模型...")
        create_mock_model(model_path, feature_dim=128)

    # 初始化特征提取器
    print("\n初始化特征提取器...")
    extractor = FeatureExtractor(
        model_path=model_path,
        device='cpu',  # 测试使用CPU
        feature_dim=128
    )

    # 创建测试图像
    print("\n创建测试图像...")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 测试单个特征提取
    print("\n" + "=" * 50)
    print("测试单个特征提取")
    print("=" * 50)

    bbox1 = np.array([100, 100, 50, 100])
    feature1 = extractor.extract_single(bbox1, test_img)

    print(f"边界框: {bbox1}")
    print(f"特征维度: {feature1.shape}")
    print(f"特征范数: {np.linalg.norm(feature1):.6f}")
    print(f"特征前5维: {feature1[:5]}")

    # 测试批量特征提取
    print("\n" + "=" * 50)
    print("测试批量特征提取")
    print("=" * 50)

    bboxes = [
        np.array([100, 100, 50, 100]),
        np.array([200, 150, 60, 120]),
        np.array([300, 200, 55, 110])
    ]

    features = extractor.extract_batch(bboxes, test_img)

    print(f"边界框数量: {len(bboxes)}")
    print(f"特征矩阵形状: {features.shape}")
    print(f"各特征范数: {[f'{np.linalg.norm(f):.6f}' for f in features]}")

    # 测试距离计算
    print("\n" + "=" * 50)
    print("测试距离计算")
    print("=" * 50)

    bbox2 = np.array([105, 105, 50, 100])  # 与bbox1接近
    bbox3 = np.array([300, 300, 50, 100])  # 与bbox1较远

    feature2 = extractor.extract_single(bbox2, test_img)
    feature3 = extractor.extract_single(bbox3, test_img)

    cos_dist_12 = compute_cosine_distance(feature1, feature2)
    cos_dist_13 = compute_cosine_distance(feature1, feature3)

    euc_dist_12 = compute_euclidean_distance(feature1, feature2)
    euc_dist_13 = compute_euclidean_distance(feature1, feature3)

    print(f"特征1 vs 特征2(接近位置):")
    print(f"  余弦距离: {cos_dist_12:.4f}")
    print(f"  欧氏距离: {euc_dist_12:.4f}")

    print(f"\n特征1 vs 特征3(较远位置):")
    print(f"  余弦距离: {cos_dist_13:.4f}")
    print(f"  欧氏距离: {euc_dist_13:.4f}")

    # 性能测试
    print("\n" + "=" * 50)
    print("性能测试")
    print("=" * 50)

    n_boxes = 20
    test_bboxes = [
        np.array([i * 30, i * 20, 50, 100]) for i in range(n_boxes)
    ]

    start = time.time()
    batch_features = extractor.extract_batch(test_bboxes, test_img)
    batch_time = time.time() - start

    start = time.time()
    single_features = np.array([
        extractor.extract_single(bbox, test_img) for bbox in test_bboxes
    ])
    single_time = time.time() - start

    print(f"边界框数量: {n_boxes}")
    print(f"批量提取耗时: {batch_time * 1000:.2f}ms")
    print(f"逐个提取耗时: {single_time * 1000:.2f}ms")
    print(f"加速比: {single_time / batch_time:.2f}x")

    print("\n测试完成!")