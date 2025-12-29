"""
轨迹管理模块
实现Track类,负责单条轨迹的状态维护、特征管理和生命周期控制

轨迹状态:
- Tentative(暂定态): 新创建的轨迹,需要连续匹配N次才能确认
- Confirmed(确认): 稳定的轨迹,参与输出
- Deleted(删除): 已失效的轨迹,待清理
"""

import numpy as np
import logging
from typing import Optional
import time

# 配置日志
logger = logging.getLogger(__name__)


class TrackState:
    """
    轨迹状态枚举
    """
    Tentative = 1  # 暂定态(试用期)
    Confirmed = 2  # 确认(正式轨迹)
    Deleted = 3  # 删除(已失效)


class Track:
    """
    单条轨迹类

    封装目标在时间序列上的完整信息:
    - 运动状态(卡尔曼滤波器维护)
    - 外观特征(特征库)
    - 生命周期管理(状态转换)
    - 匹配历史

    Attributes:
        track_id: 唯一轨迹ID
        state: 当前状态(Tentative/Confirmed/Deleted)
        mean: 卡尔曼滤波器状态向量 (8,)
        covariance: 状态协方差矩阵 (8, 8)
        features: 外观特征库
        hits: 连续匹配成功次数
        age: 轨迹总帧数
        time_since_update: 自上次更新以来的帧数
    """

    # 全局轨迹ID计数器
    _count = 0

    def __init__(
            self,
            mean: np.ndarray,
            covariance: np.ndarray,
            track_id: int,
            n_init: int = 3,
            max_age: int = 30,
            feature: Optional[np.ndarray] = None
    ):
        """
        初始化轨迹

        Args:
            mean: 初始状态向量
            covariance: 初始协方差矩阵
            track_id: 轨迹ID
            n_init: 确认所需的连续匹配次数
            max_age: 最大允许的未匹配帧数
            feature: 初始外观特征
        """
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1  # 初始化计为1次匹配
        self.age = 1  # 初始帧数为1
        self.time_since_update = 0  # 刚初始化,未失配

        self.state = TrackState.Tentative  # 初始状态为暂定态

        # 特征库
        self.features = []
        if feature is not None:
            self.features.append(feature)

        # 参数
        self._n_init = n_init
        self._max_age = max_age

        logger.debug(
            f"创建轨迹: ID={track_id}, "
            f"pos=({mean[0]:.1f},{mean[1]:.1f}), "
            f"state={self._state_name()}"
        )

    def _state_name(self) -> str:
        """获取状态名称(用于日志)"""
        if self.state == TrackState.Tentative:
            return "Tentative"
        elif self.state == TrackState.Confirmed:
            return "Confirmed"
        elif self.state == TrackState.Deleted:
            return "Deleted"
        return "Unknown"

    def to_tlwh(self) -> np.ndarray:
        """
        获取当前边界框(tlwh格式)

        从状态向量 [u, v, s, r, ...] 转换为 [left, top, width, height]

        Returns:
            bbox: [left, top, width, height]
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # area * ratio = width * height * (width/height) = width^2
        ret[2] = np.sqrt(ret[2])  # width = sqrt(width^2)
        ret[3] = ret[2] / ret[3]  # height = width / ratio
        ret[:2] -= ret[2:] / 2  # center -> top-left
        return ret

    def to_tlbr(self) -> np.ndarray:
        """
        获取当前边界框(tlbr格式)

        Returns:
            bbox: [left, top, right, bottom]
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """
        预测下一帧的状态

        调用卡尔曼滤波器执行预测步骤

        Args:
            kf: 卡尔曼滤波器实例
        """
        start_time = time.time()

        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

        elapsed_time = time.time() - start_time
        logger.debug(
            f"轨迹{self.track_id}预测: age={self.age}, "
            f"time_since_update={self.time_since_update}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

    def update(self, kf, detection):
        """
        使用检测结果更新轨迹

        执行卡尔曼滤波器更新、特征库更新、状态转换

        Args:
            kf: 卡尔曼滤波器实例
            detection: Detection对象
        """
        start_time = time.time()

        # 卡尔曼滤波器更新
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )

        # 更新外观特征
        if detection.feature is not None:
            self.features.append(detection.feature)

        # 更新统计信息
        self.hits += 1
        self.time_since_update = 0

        # 状态转换: Tentative -> Confirmed
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
            logger.info(
                f"轨迹{self.track_id}确认: hits={self.hits}/{self._n_init}"
            )

        elapsed_time = time.time() - start_time
        logger.debug(
            f"轨迹{self.track_id}更新: hits={self.hits}, "
            f"pos=({self.mean[0]:.1f},{self.mean[1]:.1f}), "
            f"state={self._state_name()}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

    def mark_missed(self):
        """
        标记为未匹配

        当前帧未找到匹配的检测框时调用
        检查是否需要删除轨迹
        """
        # 暂定态轨迹:首次未匹配即删除
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            logger.debug(f"轨迹{self.track_id}删除: Tentative首次未匹配")

        # 确认轨迹:超过最大年龄才删除
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            logger.info(
                f"轨迹{self.track_id}删除: "
                f"time_since_update={self.time_since_update}>{self._max_age}"
            )

    def is_tentative(self) -> bool:
        """是否为暂定态轨迹"""
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """是否为确认轨迹"""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """是否已删除"""
        return self.state == TrackState.Deleted

    def get_feature_count(self) -> int:
        """获取特征库中的特征数量"""
        return len(self.features)

    def get_latest_feature(self) -> Optional[np.ndarray]:
        """获取最新的外观特征"""
        if len(self.features) == 0:
            return None
        return self.features[-1]

    def get_feature_history(self, n: int = 10) -> np.ndarray:
        """
        获取最近n个特征

        Args:
            n: 特征数量

        Returns:
            features: 特征矩阵 (min(n, len), feature_dim)
        """
        if len(self.features) == 0:
            return np.array([])

        recent_features = self.features[-n:]
        return np.array(recent_features)

    def clear_old_features(self, max_features: int = 100):
        """
        清理旧特征(FIFO策略)

        Args:
            max_features: 最大保留特征数
        """
        if len(self.features) > max_features:
            n_remove = len(self.features) - max_features
            self.features = self.features[n_remove:]
            logger.debug(
                f"轨迹{self.track_id}清理特征: "
                f"移除{n_remove}个, 保留{len(self.features)}个"
            )

    def get_info(self) -> dict:
        """
        获取轨迹详细信息(用于调试/可视化)

        Returns:
            info: 包含轨迹各项信息的字典
        """
        bbox_tlwh = self.to_tlwh()
        bbox_tlbr = self.to_tlbr()

        return {
            'track_id': self.track_id,
            'state': self._state_name(),
            'bbox_tlwh': bbox_tlwh.tolist(),
            'bbox_tlbr': bbox_tlbr.tolist(),
            'position': (self.mean[0], self.mean[1]),
            'velocity': (self.mean[4], self.mean[5]),
            'area': self.mean[2],
            'aspect_ratio': self.mean[3],
            'hits': self.hits,
            'age': self.age,
            'time_since_update': self.time_since_update,
            'n_features': len(self.features)
        }

    @staticmethod
    def next_id() -> int:
        """
        生成下一个轨迹ID

        Returns:
            track_id: 新的轨迹ID
        """
        Track._count += 1
        return Track._count

    @staticmethod
    def reset_id_counter():
        """重置ID计数器(用于测试)"""
        Track._count = 0

    def __repr__(self) -> str:
        """字符串表示"""
        bbox = self.to_tlwh()
        return (
            f"Track(id={self.track_id}, "
            f"state={self._state_name()}, "
            f"bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}], "
            f"hits={self.hits}, age={self.age}, "
            f"time_since_update={self.time_since_update})"
        )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试Track类")
    print("=" * 50)

    # 重置ID计数器
    Track.reset_id_counter()

    # 创建卡尔曼滤波器(需要导入)
    import sys

    sys.path.insert(0, '/home/claude/Task2_DeepSORT_Tracking')
    from utils.kalman_filter import KalmanFilter
    from utils.detection import Detection

    kf = KalmanFilter()

    # 创建初始检测
    measurement = np.array([100.0, 200.0, 5000.0, 0.5])  # [u, v, area, ratio]
    mean, covariance = kf.initiate(measurement)

    # 创建特征
    feature = np.random.randn(128)
    feature /= np.linalg.norm(feature)

    # 创建轨迹
    print("\n创建轨迹...")
    track = Track(
        mean=mean,
        covariance=covariance,
        track_id=Track.next_id(),
        n_init=3,
        max_age=30,
        feature=feature
    )

    print(f"初始状态: {track}")
    print(f"状态: {track._state_name()}")
    print(f"边界框(tlwh): {track.to_tlwh()}")
    print(f"边界框(tlbr): {track.to_tlbr()}")

    # 模拟多帧的预测-更新循环
    print("\n" + "=" * 50)
    print("模拟预测-更新循环")
    print("=" * 50)

    for frame in range(1, 6):
        print(f"\n--- 第{frame}帧 ---")

        # 预测
        track.predict(kf)
        print(f"预测后: age={track.age}, time_since_update={track.time_since_update}")

        # 模拟检测结果
        det_bbox = track.to_tlwh() + np.random.randn(4) * 2  # 添加噪声
        det_feature = np.random.randn(128)
        det_feature /= np.linalg.norm(det_feature)

        detection = Detection(
            tlwh=det_bbox,
            confidence=0.8,
            class_id=0,
            feature=det_feature
        )

        # 更新
        track.update(kf, detection)
        print(f"更新后: {track}")
        print(f"位置: ({track.mean[0]:.1f}, {track.mean[1]:.1f})")
        print(f"速度: ({track.mean[4]:.2f}, {track.mean[5]:.2f})")
        print(f"特征数: {track.get_feature_count()}")

    # 测试状态转换
    print("\n" + "=" * 50)
    print("测试状态转换")
    print("=" * 50)

    print(f"\n当前状态: {track._state_name()}")
    print(f"是否确认: {track.is_confirmed()}")
    print(f"是否暂定: {track.is_tentative()}")

    # 测试未匹配
    print("\n模拟连续未匹配...")
    for i in range(35):
        track.predict(kf)
        track.mark_missed()
        if track.is_deleted():
            print(f"轨迹在第{i + 1}次未匹配后被删除")
            break

    # 测试特征管理
    print("\n" + "=" * 50)
    print("测试特征管理")
    print("=" * 50)

    # 创建新轨迹
    Track.reset_id_counter()
    mean2, cov2 = kf.initiate(measurement)
    track2 = Track(mean2, cov2, Track.next_id(), feature=feature)

    # 添加多个特征
    for i in range(10):
        feat = np.random.randn(128)
        feat /= np.linalg.norm(feat)
        track2.features.append(feat)

    print(f"特征总数: {track2.get_feature_count()}")

    recent = track2.get_feature_history(n=5)
    print(f"最近5个特征形状: {recent.shape}")

    # 测试特征清理
    track2.clear_old_features(max_features=5)
    print(f"清理后特征数: {track2.get_feature_count()}")

    # 获取轨迹信息
    print("\n" + "=" * 50)
    print("轨迹详细信息")
    print("=" * 50)

    info = track2.get_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\n测试完成!")