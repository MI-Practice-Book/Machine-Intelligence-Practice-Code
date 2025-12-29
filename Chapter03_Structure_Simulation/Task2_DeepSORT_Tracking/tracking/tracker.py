"""
多轨迹管理器模块
实现Tracker类,负责管理所有轨迹的生命周期:
- 预测所有轨迹的状态
- 执行级联匹配和IoU匹配
- 更新匹配的轨迹
- 初始化新轨迹
- 删除失效轨迹
"""

import numpy as np
import logging
from typing import List
import time

from .track import Track, TrackState
from utils.kalman_filter import KalmanFilter
from utils.matching import (
    matching_cascade,
    min_cost_matching,
    iou_cost,
    NearestNeighborDistanceMetric
)

# 配置日志
logger = logging.getLogger(__name__)


class Tracker:
    """
    多目标跟踪器

    管理所有轨迹,执行完整的跟踪流程:
    1. 预测: 对所有轨迹执行卡尔曼预测
    2. 级联匹配: 按轨迹年龄分层匹配(优先处理稳定轨迹)
    3. IoU匹配: 对剩余的暂定态轨迹进行IoU匹配
    4. 更新: 更新成功匹配的轨迹
    5. 初始化: 为未匹配的检测创建新轨迹
    6. 清理: 删除失效轨迹

    Attributes:
        tracks: 当前所有轨迹列表
        kf: 卡尔曼滤波器
        metric: 外观距离度量
    """

    def __init__(
            self,
            metric: str = 'cosine',
            max_iou_distance: float = 0.7,
            max_age: int = 30,
            n_init: int = 3,
            nn_budget: int = 100,
            max_cosine_distance: float = 0.2
    ):
        """
        初始化跟踪器

        Args:
            metric: 外观距离类型 ('cosine' or 'euclidean')
            max_iou_distance: IoU匹配的最大距离阈值
            max_age: 轨迹最大允许的未匹配帧数
            n_init: 轨迹确认所需的连续匹配次数
            nn_budget: 每条轨迹的特征库容量
            max_cosine_distance: 外观匹配的最大余弦距离
        """
        self.metric = NearestNeighborDistanceMetric(
            metric, max_cosine_distance, nn_budget
        )
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

        logger.info(
            f"Tracker初始化: metric={metric}, "
            f"max_iou_distance={max_iou_distance}, "
            f"max_age={max_age}, n_init={n_init}"
        )

    def predict(self):
        """
        对所有轨迹执行卡尔曼预测

        在新一帧到来时首先调用,预测所有轨迹在当前帧的位置
        """
        start_time = time.time()

        for track in self.tracks:
            track.predict(self.kf)

        elapsed_time = time.time() - start_time
        logger.debug(
            f"预测所有轨迹: 数量={len(self.tracks)}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

    def update(self, detections: List):
        """
        使用检测结果更新跟踪器

        执行完整的更新流程:
        1. 级联匹配(确认轨迹)
        2. IoU匹配(暂定态轨迹)
        3. 更新匹配轨迹
        4. 标记未匹配轨迹
        5. 初始化新轨迹

        Args:
            detections: Detection对象列表
        """
        start_time = time.time()

        # 执行匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # 更新匹配成功的轨迹
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx]
            )

        # 标记未匹配的轨迹
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 初始化新轨迹
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 删除失效轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # 更新距离度量的特征库
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id] * len(track.features)
            # 限制特征数量
            track.features = track.features[-self.metric.budget:]

        self.metric.partial_fit(
            np.asarray(features) if len(features) > 0 else np.array([]),
            np.asarray(targets),
            active_targets
        )

        elapsed_time = time.time() - start_time
        logger.info(
            f"更新完成: 检测{len(detections)}, 匹配{len(matches)}, "
            f"新轨迹{len(unmatched_detections)}, "
            f"总轨迹{len(self.tracks)}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

    def _match(self, detections: List):
        """
        执行数据关联

        两阶段匹配:
        1. 级联匹配: 确认轨迹 + 外观/运动融合代价
        2. IoU匹配: 暂定态轨迹 + IoU距离

        Args:
            detections: 检测列表

        Returns:
            matches: 匹配对 [(track_idx, detection_idx), ...]
            unmatched_tracks: 未匹配的轨迹索引
            unmatched_detections: 未匹配的检测索引
        """
        start_time = time.time()

        # 分离确认轨迹和暂定态轨迹
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # 阶段1: 级联匹配(确认轨迹)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching_cascade(
                self._gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks
            )

        # 阶段2: IoU匹配(暂定态轨迹 + 级联匹配中未匹配的确认轨迹)
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(
                iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections
            )

        # 合并结果
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        elapsed_time = time.time() - start_time
        logger.debug(
            f"匹配完成: 级联{len(matches_a)}对, IoU{len(matches_b)}对, "
            f"总匹配{len(matches)}对, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return matches, unmatched_tracks, unmatched_detections

    def _gated_metric(
            self,
            tracks: List[Track],
            dets: List,
            track_indices: List[int],
            detection_indices: List[int]
    ):
        """
        融合外观和运动信息的门控距离度量

        计算步骤:
        1. 提取检测特征
        2. 计算外观距离矩阵(余弦距离)
        3. 使用马氏距离进行门控(过滤不合理匹配)

        Args:
            tracks: 轨迹列表
            dets: 检测列表
            track_indices: 参与匹配的轨迹索引
            detection_indices: 参与匹配的检测索引

        Returns:
            cost_matrix: 代价矩阵 (N, M)
        """
        # 提取特征
        features = np.array([dets[i].feature for i in detection_indices])
        targets = np.array([tracks[i].track_id for i in track_indices])

        # 计算外观距离
        cost_matrix = self.metric.distance(features, targets)

        # 马氏距离门控
        cost_matrix = self._gate_cost_matrix(
            cost_matrix, tracks, dets, track_indices, detection_indices
        )

        return cost_matrix

    def _gate_cost_matrix(
            self,
            cost_matrix: np.ndarray,
            tracks: List[Track],
            detections: List,
            track_indices: List[int],
            detection_indices: List[int],
            only_position: bool = False
    ):
        """
        使用马氏距离对代价矩阵进行门控

        将马氏距离超过阈值的位置设为极大值,排除不合理匹配

        Args:
            cost_matrix: 外观代价矩阵
            tracks: 轨迹列表
            detections: 检测列表
            track_indices: 轨迹索引
            detection_indices: 检测索引
            only_position: 是否仅使用位置(忽略面积和宽高比)

        Returns:
            gated_cost_matrix: 门控后的代价矩阵
        """
        gating_dim = 2 if only_position else 4
        gating_threshold = self._chi2inv95[gating_dim]

        measurements = np.asarray([
            detections[i].to_xyah() for i in detection_indices
        ])

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            gating_distance = self.kf.gating_distance(
                track.mean, track.covariance, measurements, only_position
            )
            cost_matrix[row, gating_distance > gating_threshold] = 1e5

        return cost_matrix

    # 卡方分布95%分位数
    _chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070
    }

    def _initiate_track(self, detection):
        """
        初始化新轨迹

        为未匹配的检测创建新的暂定态轨迹

        Args:
            detection: Detection对象
        """
        mean, covariance = self.kf.initiate(detection.to_xyah())

        track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature
        )

        self.tracks.append(track)
        self._next_id += 1

        logger.debug(f"初始化新轨迹: ID={track.track_id}")

    def get_confirmed_tracks(self) -> List[Track]:
        """
        获取所有确认状态的轨迹

        Returns:
            confirmed_tracks: 确认轨迹列表
        """
        return [t for t in self.tracks if t.is_confirmed()]

    def get_all_tracks(self) -> List[Track]:
        """
        获取所有轨迹(包括暂定态)

        Returns:
            all_tracks: 所有轨迹列表
        """
        return self.tracks

    def get_track_count(self) -> dict:
        """
        获取各状态轨迹数量统计

        Returns:
            count_dict: {'confirmed': int, 'tentative': int, 'total': int}
        """
        confirmed = sum(1 for t in self.tracks if t.is_confirmed())
        tentative = sum(1 for t in self.tracks if t.is_tentative())

        return {
            'confirmed': confirmed,
            'tentative': tentative,
            'total': len(self.tracks)
        }

    def reset(self):
        """
        重置跟踪器(清空所有轨迹)
        """
        self.tracks = []
        self._next_id = 1
        Track.reset_id_counter()
        logger.info("跟踪器已重置")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试Tracker类")
    print("=" * 50)

    # 导入依赖
    import sys

    sys.path.insert(0, '/home/claude/Task2_DeepSORT_Tracking')
    from utils.detection import Detection

    # 初始化跟踪器
    print("\n初始化跟踪器...")
    tracker = Tracker(
        metric='cosine',
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        max_cosine_distance=0.2
    )

    print(f"初始轨迹数: {len(tracker.tracks)}")

    # 模拟多帧跟踪
    print("\n" + "=" * 50)
    print("模拟多帧跟踪流程")
    print("=" * 50)

    np.random.seed(42)

    # 第1帧: 3个检测
    print("\n--- 第1帧 ---")
    detections_1 = []
    for i in range(3):
        bbox = np.array([100 + i * 150, 200, 50, 100])
        feature = np.random.randn(128)
        feature /= np.linalg.norm(feature)

        det = Detection(
            tlwh=bbox,
            confidence=0.9,
            class_id=0,
            feature=feature
        )
        detections_1.append(det)

    tracker.predict()
    tracker.update(detections_1)

    count = tracker.get_track_count()
    print(f"检测: {len(detections_1)}")
    print(f"轨迹统计: {count}")

    # 第2-4帧: 持续跟踪(使轨迹确认)
    for frame in range(2, 5):
        print(f"\n--- 第{frame}帧 ---")

        detections = []
        for i, track in enumerate(tracker.tracks):
            # 模拟检测结果(在轨迹预测位置附近)
            bbox = track.to_tlwh() + np.random.randn(4) * 3
            feature = np.random.randn(128)
            feature /= np.linalg.norm(feature)

            det = Detection(
                tlwh=bbox,
                confidence=0.85,
                class_id=0,
                feature=feature
            )
            detections.append(det)

        tracker.predict()
        tracker.update(detections)

        count = tracker.get_track_count()
        print(f"检测: {len(detections)}")
        print(f"轨迹统计: {count}")

    # 第5帧: 新增1个目标
    print("\n--- 第5帧: 新增目标 ---")
    detections_5 = []

    # 原有目标
    for track in tracker.get_confirmed_tracks():
        bbox = track.to_tlwh() + np.random.randn(4) * 3
        feature = np.random.randn(128)
        feature /= np.linalg.norm(feature)

        det = Detection(tlwh=bbox, confidence=0.85, class_id=0, feature=feature)
        detections_5.append(det)

    # 新目标
    new_bbox = np.array([500, 250, 55, 110])
    new_feature = np.random.randn(128)
    new_feature /= np.linalg.norm(new_feature)
    new_det = Detection(tlwh=new_bbox, confidence=0.9, class_id=0, feature=new_feature)
    detections_5.append(new_det)

    tracker.predict()
    tracker.update(detections_5)

    count = tracker.get_track_count()
    print(f"检测: {len(detections_5)}")
    print(f"轨迹统计: {count}")

    # 第6帧: 1个目标消失
    print("\n--- 第6帧: 1个目标消失 ---")
    detections_6 = []

    # 仅保留前2个轨迹
    for track in tracker.get_confirmed_tracks()[:2]:
        bbox = track.to_tlwh() + np.random.randn(4) * 3
        feature = np.random.randn(128)
        feature /= np.linalg.norm(feature)

        det = Detection(tlwh=bbox, confidence=0.85, class_id=0, feature=feature)
        detections_6.append(det)

    tracker.predict()
    tracker.update(detections_6)

    count = tracker.get_track_count()
    print(f"检测: {len(detections_6)}")
    print(f"轨迹统计: {count}")

    # 查看所有轨迹详情
    print("\n" + "=" * 50)
    print("当前所有轨迹")
    print("=" * 50)

    for track in tracker.tracks:
        print(f"\n{track}")
        info = track.get_info()
        print(f"  位置: ({info['position'][0]:.1f}, {info['position'][1]:.1f})")
        print(f"  速度: ({info['velocity'][0]:.2f}, {info['velocity'][1]:.2f})")
        print(f"  特征数: {info['n_features']}")

    # 测试重置
    print("\n" + "=" * 50)
    print("测试重置功能")
    print("=" * 50)

    print(f"重置前轨迹数: {len(tracker.tracks)}")
    tracker.reset()
    print(f"重置后轨迹数: {len(tracker.tracks)}")

    print("\n测试完成!")