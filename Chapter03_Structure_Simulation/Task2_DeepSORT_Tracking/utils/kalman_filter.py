"""
卡尔曼滤波器模块
实现用于目标跟踪的卡尔曼滤波器,负责运动状态建模、预测和更新

状态向量: x = [u, v, s, r, u_dot, v_dot, s_dot, 0]^T
- (u, v): 边界框中心坐标
- s: 边界框面积 (width * height)
- r: 边界框宽高比 (width / height)
- (u_dot, v_dot, s_dot): 对应的速度
- 宽高比假设恒定,速度为0
"""

import numpy as np
import logging
from typing import Tuple
import time

# 配置日志
logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    卡尔曼滤波器用于目标跟踪的运动预测

    实现了经典的卡尔曼滤波预测-更新循环:
    1. 预测阶段: 基于运动模型预测下一时刻状态
    2. 更新阶段: 融合观测值修正状态估计

    采用匀速运动模型 (Constant Velocity Model)
    """

    def __init__(self):
        """
        初始化卡尔曼滤波器参数

        定义状态转移矩阵、观测矩阵和噪声协方差矩阵
        """
        ndim = 4  # 观测维度
        dt = 1.0  # 时间间隔(帧)

        # 状态转移矩阵 F (8x8)
        # x_t = F * x_{t-1} + w
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 观测矩阵 H (4x8)
        # z_t = H * x_t + v
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 过程噪声标准差
        # 位置噪声较小,速度噪声较大
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        logger.info("卡尔曼滤波器初始化完成")
        logger.debug(f"状态转移矩阵F形状: {self._motion_mat.shape}")
        logger.debug(f"观测矩阵H形状: {self._update_mat.shape}")

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化新轨迹的状态和协方差

        Args:
            measurement: 初始观测 [u, v, s, r] (center_x, center_y, area, aspect_ratio)

        Returns:
            mean: 初始状态向量 (8,)
            covariance: 初始协方差矩阵 (8, 8)
        """
        start_time = time.time()

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]  # 拼接为8维向量

        # 初始协方差矩阵
        # 位置不确定性较小,速度不确定性较大
        std = [
            2 * self._std_weight_position * measurement[2],  # u的标准差
            2 * self._std_weight_position * measurement[2],  # v的标准差
            1e-2,  # s的标准差
            2 * self._std_weight_position * measurement[2],  # r的标准差
            10 * self._std_weight_velocity * measurement[2],  # u_dot的标准差
            10 * self._std_weight_velocity * measurement[2],  # v_dot的标准差
            1e-5,  # s_dot的标准差
            10 * self._std_weight_velocity * measurement[2]  # r_dot的标准差(实际不用)
        ]
        covariance = np.diag(np.square(std))

        elapsed_time = time.time() - start_time
        logger.debug(
            f"初始化轨迹: mean={mean[:4]}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return mean, covariance

    def predict(
            self,
            mean: np.ndarray,
            covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测下一时刻的状态(预测阶段)

        基于匀速运动模型:
        x_t = F * x_{t-1} + w
        P_t = F * P_{t-1} * F^T + Q

        Args:
            mean: 当前状态向量 (8,)
            covariance: 当前协方差矩阵 (8, 8)

        Returns:
            mean: 预测状态向量 (8,)
            covariance: 预测协方差矩阵 (8, 8)
        """
        start_time = time.time()

        # 过程噪声协方差 Q
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-2,
            self._std_weight_position * mean[2]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[2],
            1e-5,
            self._std_weight_velocity * mean[2]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 预测状态
        mean = np.dot(self._motion_mat, mean)
        covariance = (
                np.linalg.multi_dot((
                    self._motion_mat,
                    covariance,
                    self._motion_mat.T
                )) + motion_cov
        )

        elapsed_time = time.time() - start_time
        logger.debug(
            f"预测状态: pos=({mean[0]:.1f}, {mean[1]:.1f}), "
            f"vel=({mean[4]:.2f}, {mean[5]:.2f}), "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return mean, covariance

    def project(
            self,
            mean: np.ndarray,
            covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将状态空间投影到观测空间

        z = H * x
        S = H * P * H^T + R

        Args:
            mean: 状态向量 (8,)
            covariance: 协方差矩阵 (8, 8)

        Returns:
            mean: 投影后的观测向量 (4,)
            covariance: 观测空间的协方差 (4, 4) (新息协方差)
        """
        # 观测噪声协方差 R
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-1,
            self._std_weight_position * mean[2]
        ]
        innovation_cov = np.diag(np.square(std))

        # 投影到观测空间
        mean = np.dot(self._update_mat, mean)
        covariance = (
                np.linalg.multi_dot((
                    self._update_mat,
                    covariance,
                    self._update_mat.T
                )) + innovation_cov
        )

        return mean, covariance

    def update(
            self,
            mean: np.ndarray,
            covariance: np.ndarray,
            measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用观测值更新状态估计(更新阶段)

        计算卡尔曼增益并融合观测:
        K = P * H^T * S^{-1}
        x = x + K * (z - H*x)
        P = (I - K*H) * P

        Args:
            mean: 预测状态向量 (8,)
            covariance: 预测协方差矩阵 (8, 8)
            measurement: 观测向量 [u, v, s, r] (4,)

        Returns:
            mean: 更新后状态向量 (8,)
            covariance: 更新后协方差矩阵 (8, 8)
        """
        start_time = time.time()

        # 投影到观测空间
        projected_mean, projected_cov = self.project(mean, covariance)

        # 计算卡尔曼增益
        # K = P * H^T * S^{-1}
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T

        # 计算新息(innovation)
        innovation = measurement - projected_mean

        # 更新状态
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain,
            projected_cov,
            kalman_gain.T
        ))

        elapsed_time = time.time() - start_time
        logger.debug(
            f"更新状态: innovation={innovation[:2]}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return new_mean, new_covariance

    def gating_distance(
            self,
            mean: np.ndarray,
            covariance: np.ndarray,
            measurements: np.ndarray,
            only_position: bool = False
    ) -> np.ndarray:
        """
        计算状态与观测之间的马氏距离(用于门控/数据关联)

        马氏距离考虑了协方差,比欧氏距离更适合匹配:
        d^2 = (z - H*x)^T * S^{-1} * (z - H*x)

        Args:
            mean: 状态向量 (8,)
            covariance: 协方差矩阵 (8, 8)
            measurements: 观测矩阵 (N, 4)
            only_position: 是否仅使用位置信息(忽略面积和宽高比)

        Returns:
            distances: 马氏距离数组 (N,)
        """
        start_time = time.time()

        # 投影到观测空间
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        # 计算马氏距离
        # d = sqrt((z - mu)^T * Sigma^{-1} * (z - mu))
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True,
            check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)

        elapsed_time = time.time() - start_time
        logger.debug(
            f"计算马氏距离: N={len(measurements)}, "
            f"耗时={elapsed_time * 1000:.2f}ms"
        )

        return squared_maha


# 为了避免scipy依赖,提供简化版本
try:
    import scipy.linalg
except ImportError:
    logger.warning("未安装scipy,使用numpy实现(可能较慢)")


    # 使用numpy替代scipy的Cholesky分解
    class scipy:
        class linalg:
            @staticmethod
            def cho_factor(a, lower=True, check_finite=True):
                """Cholesky分解: A = L * L^T"""
                chol = np.linalg.cholesky(a)
                return chol, lower

            @staticmethod
            def cho_solve(c_and_lower, b, check_finite=True):
                """求解 A*x = b, 其中A已做Cholesky分解"""
                c, lower = c_and_lower
                # 先求解 L*y = b
                y = np.linalg.solve(c, b)
                # 再求解 L^T*x = y
                x = np.linalg.solve(c.T, y)
                return x

            @staticmethod
            def solve_triangular(a, b, lower=True, check_finite=True, overwrite_b=False):
                """求解三角线性系统"""
                if lower:
                    return np.linalg.solve(a, b)
                else:
                    return np.linalg.solve(a.T, b)


def chi2inv95(df: int) -> float:
    """
    卡方分布的95%分位数
    用于马氏距离门控阈值

    Args:
        df: 自由度

    Returns:
        卡方分布95%分位数
    """
    # 预计算的常用值
    chi2_table = {
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

    if df in chi2_table:
        return chi2_table[df]
    else:
        logger.warning(f"自由度{df}的卡方值未预计算,使用近似公式")
        return df + 2 * np.sqrt(2 * df)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 50)
    print("测试卡尔曼滤波器")
    print("=" * 50)

    kf = KalmanFilter()

    # 初始化轨迹
    measurement = np.array([100.0, 200.0, 5000.0, 0.5])  # [u, v, area, ratio]
    mean, covariance = kf.initiate(measurement)

    print(f"\n初始状态:")
    print(f"位置: ({mean[0]:.1f}, {mean[1]:.1f})")
    print(f"面积: {mean[2]:.1f}")
    print(f"宽高比: {mean[3]:.3f}")
    print(f"速度: ({mean[4]:.2f}, {mean[5]:.2f})")

    # 模拟5帧的预测-更新循环
    print(f"\n{'=' * 50}")
    print("模拟预测-更新循环")
    print(f"{'=' * 50}")

    measurements = [
        np.array([105.0, 205.0, 5100.0, 0.5]),
        np.array([110.0, 210.0, 5200.0, 0.5]),
        np.array([115.0, 215.0, 5300.0, 0.5]),
        np.array([120.0, 220.0, 5400.0, 0.5]),
        np.array([125.0, 225.0, 5500.0, 0.5])
    ]

    for i, meas in enumerate(measurements, 1):
        # 预测
        mean, covariance = kf.predict(mean, covariance)
        print(f"\n第{i}帧预测:")
        print(f"  预测位置: ({mean[0]:.1f}, {mean[1]:.1f})")
        print(f"  预测速度: ({mean[4]:.2f}, {mean[5]:.2f})")

        # 更新
        mean, covariance = kf.update(mean, covariance, meas)
        print(f"  更新位置: ({mean[0]:.1f}, {mean[1]:.1f})")
        print(f"  观测位置: ({meas[0]:.1f}, {meas[1]:.1f})")
        print(f"  更新速度: ({mean[4]:.2f}, {mean[5]:.2f})")

    # 测试马氏距离
    print(f"\n{'=' * 50}")
    print("测试马氏距离计算")
    print(f"{'=' * 50}")

    test_measurements = np.array([
        [125.0, 225.0, 5500.0, 0.5],  # 匹配良好
        [130.0, 230.0, 5600.0, 0.5],  # 较近
        [200.0, 300.0, 6000.0, 0.6]  # 较远
    ])

    distances = kf.gating_distance(mean, covariance, test_measurements)

    for i, (meas, dist) in enumerate(zip(test_measurements, distances)):
        print(f"观测{i + 1}: pos=({meas[0]:.1f}, {meas[1]:.1f}), "
              f"马氏距离²={dist:.2f}")

    # 卡方阈值
    threshold = chi2inv95(4)
    print(f"\n95%置信度门控阈值(自由度=4): {threshold:.2f}")
    print(f"通过门控的观测: {np.sum(distances < threshold)}/{len(distances)}")

    print("\n测试完成!")