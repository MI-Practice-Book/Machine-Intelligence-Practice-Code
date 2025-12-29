"""
数据关联模块
实现多目标跟踪中的数据关联算法:
1. 匈牙利算法求解最优分配
2. 级联匹配策略(按轨迹年龄分层)
3. IoU距离和马氏距离计算
4. 代价矩阵构建与门控
"""

import numpy as np
import logging
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
import time

# 配置日志
logger = logging.getLogger(__name__)


def min_cost_matching(
        distance_metric,
        max_distance: float,
        tracks: List,
        detections: List,
        track_indices: List[int] = None,
        detection_indices: List[int] = None
) -> Tuple[List, List, List]:
    """
    使用匈牙利算法求解最优分配问题

    求解目标: min Σ cost[i,j] * x[i,j]
    约束条件: 每个轨迹最多匹配一个检测,每个检测最多匹配一个轨迹

    Args:
        distance_metric: 距离度量函数
        max_distance: 最大允许距离(门控阈值)
        tracks: 轨迹列表
        detections: 检测列表
        track_indices: 参与匹配的轨迹索引(None表示全部)
        detection_indices: 参与匹配的检测索引(None表示全部)

    Returns:
        matches: 匹配对列表 [(track_idx, detection_idx), ...]
        unmatched_tracks: 未匹配的轨迹索引
        unmatched_detections: 未匹配的检测索引
    """
    start_time = time.time()

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # 空集处理
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    # 构建代价矩阵
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices
    )

    # 应用门控:超过阈值的设为无穷大
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # 调用匈牙利算法
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # 整理结果
    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # 过滤超过阈值的匹配
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    elapsed_time = time.time() - start_time
    logger.debug(
        f"匈牙利匹配: 轨迹{len(track_indices)} x 检测{len(detection_indices)}, "
        f"匹配{len(matches)}对, "
        f"耗时={elapsed_time * 1000:.2f}ms"
    )

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric,
        max_distance: float,
        cascade_depth: int,
        tracks: List,
        detections: List,
        track_indices: List[int] = None,
        detection_indices: List[int] = None
) -> Tuple[List, List, List]:
    """
    级联匹配策略

    按轨迹年龄(自上次匹配以来的帧数)从小到大依次匹配:
    1. 年龄=0的轨迹优先(刚匹配过,最可靠)
    2. 年龄=1,2,...的轨迹依次匹配
    3. 每轮匹配后,成功匹配的检测从候选集中移除

    这种策略避免了长时未匹配的不稳定轨迹"抢走"稳定轨迹的检测

    Args:
        distance_metric: 距离度量函数
        max_distance: 最大允许距离
        cascade_depth: 级联深度(最大年龄)
        tracks: 轨迹列表
        detections: 检测列表
        track_indices: 参与匹配的轨迹索引
        detection_indices: 参与匹配的检测索引

    Returns:
        matches: 匹配对列表
        unmatched_tracks: 未匹配的轨迹索引
        unmatched_detections: 未匹配的检测索引
    """
    start_time = time.time()

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []

    # 按年龄从小到大依次匹配
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break

        # 选择年龄为level的轨迹
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == level
        ]

        if len(track_indices_l) == 0:
            continue

        # 对当前层级进行匹配
        matches_l, unmatched_tracks_l, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections
        )

        matches.extend(matches_l)

        logger.debug(
            f"级联匹配层级{level}: 轨迹{len(track_indices_l)}, "
            f"检测{len(detection_indices)}, 匹配{len(matches_l)}对"
        )

    # 收集所有未匹配的轨迹
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    elapsed_time = time.time() - start_time
    logger.info(
        f"级联匹配完成: 总匹配{len(matches)}对, "
        f"未匹配轨迹{len(unmatched_tracks)}, "
        f"未匹配检测{len(unmatched_detections)}, "
        f"耗时={elapsed_time * 1000:.2f}ms"
    )

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf,
        cost_matrix: np.ndarray,
        tracks: List,
        detections: List,
        track_indices: List[int],
        detection_indices: List[int],
        gated_cost: float = 1e5,
        only_position: bool = False
) -> np.ndarray:
    """
    使用马氏距离对代价矩阵进行门控

    将马氏距离超过卡方分布95%分位数的位置设为极大值,
    从而在匈牙利算法中排除不合理的匹配

    Args:
        kf: 卡尔曼滤波器实例
        cost_matrix: 原始代价矩阵 (N, M)
        tracks: 轨迹列表
        detections: 检测列表
        track_indices: 轨迹索引
        detection_indices: 检测索引
        gated_cost: 门控后的代价值
        only_position: 是否仅使用位置信息

    Returns:
        gated_cost_matrix: 门控后的代价矩阵
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]

    measurements = np.asarray([
        detections[i].to_xyah() for i in detection_indices
    ])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix


# 卡方分布95%分位数表(预计算)
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


def iou_cost(
        tracks: List,
        detections: List,
        track_indices: List[int] = None,
        detection_indices: List[int] = None
) -> np.ndarray:
    """
    计算IoU距离矩阵

    IoU距离 = 1 - IoU(bbox1, bbox2)
    值域 [0, 1]: 0表示完全重叠,1表示完全不重叠

    Args:
        tracks: 轨迹列表
        detections: 检测列表
        track_indices: 轨迹索引
        detection_indices: 检测索引

    Returns:
        cost_matrix: IoU距离矩阵 (N, M)
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            # 对于超过1帧未匹配的轨迹,IoU不可靠
            cost_matrix[row, :] = 1.0
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([
            detections[i].tlwh for i in detection_indices
        ])

        # 批量计算IoU
        cost_matrix[row, :] = 1.0 - iou_batch(bbox, candidates)

    return cost_matrix


def iou_batch(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    批量计算单个框与多个候选框的IoU

    Args:
        bbox: 单个边界框 [left, top, width, height]
        candidates: 候选边界框 (N, 4)

    Returns:
        iou_values: IoU数组 (N,)
    """
    # 转换为tlbr格式
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]

    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    # 计算交集
    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)
    wh = np.maximum(0., br - tl)

    area_intersection = wh[:, 0] * wh[:, 1]
    area_bbox = bbox[2] * bbox[3]
    area_candidates = candidates[:, 2] * candidates[:, 3]

    # 计算并集
    area_union = area_bbox + area_candidates - area_intersection

    return area_intersection / np.maximum(area_union, 1e-6)


def fuse_motion(
        kf,
        cost_matrix: np.ndarray,
        tracks: List,
        detections: List,
        track_indices: List[int] = None,
        detection_indices: List[int] = None,
        lambda_: float = 0.98
) -> np.ndarray:
    """
    融合运动信息到代价矩阵

    使用卡尔曼滤波预测的马氏距离作为运动代价,
    与外观代价(如余弦距离)进行加权融合

    Args:
        kf: 卡尔曼滤波器实例
        cost_matrix: 外观代价矩阵 (N, M)
        tracks: 轨迹列表
        detections: 检测列表
        track_indices: 轨迹索引
        detection_indices: 检测索引
        lambda_: 门控参数(较大的值使门控更严格)

    Returns:
        fused_cost_matrix: 融合后的代价矩阵
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    gating_dim = 2  # 仅使用位置
    gating_threshold = chi2inv95[gating_dim]

    measurements = np.asarray([
        detections[i].to_xyah() for i in detection_indices
    ])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position=True
        )
        # 超过门控阈值的位置,外观代价无效
        cost_matrix[row, gating_distance > gating_threshold] = 1e5
        # 门控内的位置,使用卡方CDF加权
        cost_matrix[row] = (
                lambda_ * cost_matrix[row] +
                (1 - lambda_) * gating_distance
        )

    return cost_matrix


class NearestNeighborDistanceMetric:
    """
    最近邻距离度量

    用于计算轨迹特征库与检测特征的距离:
    distance(track, detection) = min_{f in track.features} d(f, detection.feature)

    支持的距离类型:
    - 'cosine': 余弦距离
    - 'euclidean': 欧氏距离
    """

    def __init__(self, metric: str, matching_threshold: float, budget: int = 100):
        """
        初始化距离度量

        Args:
            metric: 距离类型 ('cosine' or 'euclidean')
            matching_threshold: 匹配阈值
            budget: 特征库容量(每条轨迹保存的特征数)
        """
        if metric not in ['cosine', 'euclidean']:
            raise ValueError(f"不支持的距离类型: {metric}")

        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget

        # 每条轨迹的特征库: {track_id: List[feature]}
        self.samples = {}

        logger.info(
            f"初始化距离度量: metric={metric}, "
            f"threshold={matching_threshold}, budget={budget}"
        )

    def partial_fit(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            active_targets: List[int]
    ):
        """
        更新特征库

        Args:
            features: 特征矩阵 (N, feature_dim)
            targets: 对应的轨迹ID (N,)
            active_targets: 当前活跃的轨迹ID列表
        """
        for feature, target in zip(features, targets):
            if target not in self.samples:
                self.samples[target] = []
            self.samples[target].append(feature)

            # 维持FIFO队列,限制容量
            if len(self.samples[target]) > self.budget:
                self.samples[target].pop(0)

        # 删除不活跃轨迹的特征
        self.samples = {
            k: v for k, v in self.samples.items()
            if k in active_targets
        }

        logger.debug(
            f"更新特征库: 新增{len(features)}个特征, "
            f"活跃轨迹{len(active_targets)}条"
        )

    def distance(
            self,
            features: np.ndarray,
            targets: np.ndarray
    ) -> np.ndarray:
        """
        计算距离矩阵

        Args:
            features: 检测特征矩阵 (M, feature_dim)
            targets: 轨迹ID数组 (N,)

        Returns:
            cost_matrix: 距离矩阵 (N, M)
        """
        cost_matrix = np.zeros((len(targets), len(features)))

        for i, target in enumerate(targets):
            if target not in self.samples:
                # 新轨迹,特征库为空
                cost_matrix[i, :] = self.matching_threshold + 1
                continue

            # 计算与特征库所有特征的距离,取最小值
            track_features = np.asarray(self.samples[target])

            if self.metric == 'cosine':
                # 余弦距离 = 1 - 余弦相似度
                # features和track_features都已L2归一化,可直接点积
                distances = 1.0 - np.dot(track_features, features.T)
                cost_matrix[i, :] = distances.min(axis=0)
            elif self.metric == 'euclidean':
                # 欧氏距离
                distances = np.linalg.norm(
                    track_features[:, None, :] - features[None, :, :],
                    axis=2
                )
                cost_matrix[i, :] = distances.min(axis=0)

        return cost_matrix


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试IoU批量计算")
    print("=" * 50)

    bbox = np.array([100, 100, 50, 50])
    candidates = np.array([
        [120, 120, 50, 50],  # 有重叠
        [200, 200, 50, 50],  # 无重叠
        [100, 100, 50, 50]  # 完全重叠
    ])

    ious = iou_batch(bbox, candidates)
    print(f"参考框: {bbox}")
    for i, (cand, iou_val) in enumerate(zip(candidates, ious)):
        print(f"候选框{i + 1}: {cand}, IoU={iou_val:.4f}")

    print("\n" + "=" * 50)
    print("测试最近邻距离度量")
    print("=" * 50)

    metric = NearestNeighborDistanceMetric('cosine', matching_threshold=0.7, budget=100)

    # 模拟特征更新
    features = np.random.randn(3, 128)
    features /= np.linalg.norm(features, axis=1, keepdims=True)  # L2归一化
    targets = np.array([1, 2, 1])  # 轨迹ID

    metric.partial_fit(features, targets, active_targets=[1, 2])
    print(f"特征库: {list(metric.samples.keys())}")
    print(f"轨迹1特征数: {len(metric.samples[1])}")
    print(f"轨迹2特征数: {len(metric.samples[2])}")

    # 计算距离
    test_features = np.random.randn(2, 128)
    test_features /= np.linalg.norm(test_features, axis=1, keepdims=True)
    test_targets = np.array([1, 2])

    cost_matrix = metric.distance(test_features, test_targets)
    print(f"\n距离矩阵:\n{cost_matrix}")

    print("\n测试完成!")