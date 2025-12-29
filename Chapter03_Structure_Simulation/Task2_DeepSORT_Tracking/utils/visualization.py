"""
可视化工具模块
提供多种可视化功能:
- 绘制边界框和ID标签
- 绘制运动轨迹
- 生成可视化视频
- 统计图表绘制
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os

# 配置日志
logger = logging.getLogger(__name__)


class Colors:
    """
    颜色管理类
    提供固定和动态生成的颜色方案
    """

    # 预定义的20种不同颜色(BGR格式)
    PALETTE = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255),
        (192, 192, 192), (128, 128, 128)
    ]

    @staticmethod
    def get_color(idx: int) -> Tuple[int, int, int]:
        """
        根据索引获取颜色

        Args:
            idx: 索引(通常是track_id)

        Returns:
            color: BGR颜色元组
        """
        return Colors.PALETTE[idx % len(Colors.PALETTE)]

    @staticmethod
    def hsv_to_bgr(h: float, s: float = 1.0, v: float = 1.0) -> Tuple[int, int, int]:
        """
        HSV转BGR

        Args:
            h: 色调 [0, 360]
            s: 饱和度 [0, 1]
            v: 明度 [0, 1]

        Returns:
            color: BGR颜色元组
        """
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
        return (int(b * 255), int(g * 255), int(r * 255))


class TrackVisualizer:
    """
    跟踪结果可视化器

    提供多种可视化功能:
    - 单帧边界框绘制
    - 多帧轨迹绘制
    - 轨迹历史显示
    """

    def __init__(self, max_history: int = 30):
        """
        初始化可视化器

        Args:
            max_history: 轨迹历史最大长度
        """
        self.max_history = max_history
        self.track_history = defaultdict(list)  # {track_id: [(x, y), ...]}

    def draw_boxes(
            self,
            img: np.ndarray,
            boxes: List[Tuple],
            show_id: bool = True,
            show_conf: bool = False,
            thickness: int = 2
    ) -> np.ndarray:
        """
        绘制边界框

        Args:
            img: 输入图像
            boxes: 边界框列表 [(x1, y1, x2, y2, track_id, conf, class_id), ...]
            show_id: 是否显示ID
            show_conf: 是否显示置信度
            thickness: 线条粗细

        Returns:
            vis_img: 绘制后的图像
        """
        vis_img = img.copy()

        for box in boxes:
            if len(box) < 5:
                continue

            x1, y1, x2, y2, track_id = map(int, box[:5])
            conf = box[5] if len(box) > 5 else 1.0
            class_id = int(box[6]) if len(box) > 6 else 0

            # 获取颜色
            color = Colors.get_color(track_id)

            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签
            label_parts = []
            if show_id:
                label_parts.append(f"ID:{track_id}")
            if show_conf:
                label_parts.append(f"{conf:.2f}")

            if label_parts:
                label = " ".join(label_parts)

                # 计算标签尺寸
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # 绘制标签背景
                cv2.rectangle(
                    vis_img,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w, y1),
                    color, -1
                )

                # 绘制标签文字
                cv2.putText(
                    vis_img, label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

        return vis_img

    def draw_trajectories(
            self,
            img: np.ndarray,
            boxes: List[Tuple],
            draw_lines: bool = True,
            draw_points: bool = True
    ) -> np.ndarray:
        """
        绘制运动轨迹

        Args:
            img: 输入图像
            boxes: 边界框列表
            draw_lines: 是否绘制轨迹线
            draw_points: 是否绘制轨迹点

        Returns:
            vis_img: 绘制后的图像
        """
        vis_img = img.copy()

        # 更新轨迹历史
        for box in boxes:
            if len(box) < 5:
                continue

            x1, y1, x2, y2, track_id = map(int, box[:5])

            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 添加到历史
            self.track_history[track_id].append((center_x, center_y))

            # 限制历史长度
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id].pop(0)

        # 绘制轨迹
        for track_id, history in self.track_history.items():
            if len(history) < 2:
                continue

            color = Colors.get_color(track_id)

            # 绘制轨迹线
            if draw_lines:
                points = np.array(history, dtype=np.int32)
                cv2.polylines(
                    vis_img, [points], False, color, 2
                )

            # 绘制轨迹点
            if draw_points:
                for point in history:
                    cv2.circle(vis_img, point, 3, color, -1)

        return vis_img

    def clear_history(self):
        """清除轨迹历史"""
        self.track_history.clear()
        logger.debug("轨迹历史已清除")


def draw_track_on_frame(
        img: np.ndarray,
        boxes: List[Tuple],
        show_id: bool = True,
        show_conf: bool = False,
        show_trajectory: bool = False
) -> np.ndarray:
    """
    在帧上绘制跟踪结果(便捷函数)

    Args:
        img: 输入图像
        boxes: 边界框列表
        show_id: 是否显示ID
        show_conf: 是否显示置信度
        show_trajectory: 是否显示轨迹

    Returns:
        vis_img: 可视化后的图像
    """
    visualizer = TrackVisualizer()

    vis_img = visualizer.draw_boxes(img, boxes, show_id, show_conf)

    if show_trajectory:
        vis_img = visualizer.draw_trajectories(vis_img, boxes)

    return vis_img


def create_video_from_frames(
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
        codec: str = 'mp4v'
) -> bool:
    """
    从帧序列创建视频

    Args:
        frames: 图像帧列表
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码器

    Returns:
        success: 是否成功
    """
    if len(frames) == 0:
        logger.error("帧列表为空")
        return False

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error(f"无法创建视频: {output_path}")
        return False

    for frame in frames:
        writer.write(frame)

    writer.release()
    logger.info(f"视频已保存: {output_path}, 帧数={len(frames)}, FPS={fps}")

    return True


def plot_trajectory_2d(
        trajectories: Dict[int, List[Tuple[int, int]]],
        save_path: Optional[str] = None,
        show: bool = True
):
    """
    绘制2D轨迹图

    Args:
        trajectories: {track_id: [(x, y), ...]}
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for track_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue

        points = np.array(trajectory)

        # 绘制轨迹
        ax.plot(
            points[:, 0], points[:, 1],
            marker='o', markersize=3,
            label=f"ID {track_id}"
        )

        # 标注起点和终点
        ax.scatter(points[0, 0], points[0, 1], s=100, marker='s', c='green')
        ax.scatter(points[-1, 0], points[-1, 1], s=100, marker='*', c='red')

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('2D Trajectories')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # 图像坐标系y轴向下

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"轨迹图已保存: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_tracking_stats(
        results: List[List],
        save_path: Optional[str] = None,
        show: bool = True
):
    """
    绘制跟踪统计图表

    Args:
        results: 跟踪结果 [[frame, id, left, top, w, h, conf, ...], ...]
        save_path: 保存路径
        show: 是否显示
    """
    if len(results) == 0:
        logger.warning("结果为空,无法绘制统计图")
        return

    results = np.array(results)

    # 统计每帧的目标数
    frames = results[:, 0].astype(int)
    unique_frames = np.unique(frames)

    counts_per_frame = []
    for frame in unique_frames:
        count = np.sum(frames == frame)
        counts_per_frame.append(count)

    # 统计每个ID的轨迹长度
    track_ids = results[:, 1].astype(int)
    unique_ids = np.unique(track_ids)

    track_lengths = []
    for track_id in unique_ids:
        length = np.sum(track_ids == track_id)
        track_lengths.append(length)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 每帧目标数
    axes[0, 0].plot(unique_frames, counts_per_frame, linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Number of Targets')
    axes[0, 0].set_title('Targets per Frame')
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 轨迹长度分布
    axes[0, 1].hist(track_lengths, bins=20, edgecolor='black')
    axes[0, 1].set_xlabel('Track Length (frames)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Track Length Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 时空图(轨迹生命周期)
    for track_id in unique_ids:
        mask = track_ids == track_id
        track_frames = frames[mask]
        axes[1, 0].plot(
            [track_frames.min(), track_frames.max()],
            [track_id, track_id],
            linewidth=2
        )
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Track ID')
    axes[1, 0].set_title('Track Lifecycles (Temporal View)')
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 统计摘要
    axes[1, 1].axis('off')
    summary_text = f"""
    Tracking Statistics Summary
    {'=' * 40}
    Total Frames: {len(unique_frames)}
    Total Tracks: {len(unique_ids)}
    Total Detections: {len(results)}

    Avg Targets/Frame: {np.mean(counts_per_frame):.2f}
    Max Targets/Frame: {np.max(counts_per_frame)}
    Min Targets/Frame: {np.min(counts_per_frame)}

    Avg Track Length: {np.mean(track_lengths):.2f}
    Max Track Length: {np.max(track_lengths)}
    Min Track Length: {np.min(track_lengths)}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"统计图已保存: {save_path}")

    if show:
        plt.show()

    plt.close()


def save_visualization_video(
        video_path: str,
        results: List[List],
        output_path: str,
        show_trajectory: bool = True
) -> bool:
    """
    生成可视化视频

    Args:
        video_path: 输入视频路径
        results: 跟踪结果
        output_path: 输出视频路径
        show_trajectory: 是否显示轨迹

    Returns:
        success: 是否成功
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return False

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error(f"无法创建输出视频: {output_path}")
        return False

    # 组织结果(按帧)
    results_by_frame = defaultdict(list)
    for result in results:
        frame_id = int(result[0])
        track_id = int(result[1])
        left, top, width, height = result[2:6]
        conf = result[6] if len(result) > 6 else 1.0

        # 转换为tlbr格式
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)

        results_by_frame[frame_id].append((x1, y1, x2, y2, track_id, conf, 0))

    # 创建可视化器
    visualizer = TrackVisualizer()

    # 处理每一帧
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 获取当前帧的跟踪结果
        boxes = results_by_frame.get(frame_id, [])

        # 绘制
        vis_frame = visualizer.draw_boxes(frame, boxes, show_id=True, show_conf=False)
        if show_trajectory:
            vis_frame = visualizer.draw_trajectories(vis_frame, boxes)

        writer.write(vis_frame)

    cap.release()
    writer.release()

    logger.info(f"可视化视频已保存: {output_path}")
    return True


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("测试可视化工具")
    print("=" * 50)

    # 创建测试图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # 创建测试跟踪结果
    boxes = [
        (100, 100, 150, 200, 1, 0.95, 0),
        (250, 150, 300, 250, 2, 0.88, 0),
        (400, 200, 450, 300, 3, 0.92, 0)
    ]

    # 测试边界框绘制
    print("\n测试边界框绘制...")
    visualizer = TrackVisualizer()
    vis_img = visualizer.draw_boxes(img, boxes, show_id=True, show_conf=True)
    print(f"可视化图像形状: {vis_img.shape}")

    # 测试轨迹绘制
    print("\n测试轨迹绘制...")
    for i in range(10):
        # 模拟移动
        boxes_moving = [
            (100 + i * 5, 100 + i * 3, 150 + i * 5, 200 + i * 3, 1, 0.95, 0),
            (250 - i * 3, 150 + i * 2, 300 - i * 3, 250 + i * 2, 2, 0.88, 0),
            (400 + i * 4, 200 - i * 2, 450 + i * 4, 300 - i * 2, 3, 0.92, 0)
        ]
        vis_img = visualizer.draw_trajectories(img, boxes_moving)

    print(f"轨迹历史长度: {[len(h) for h in visualizer.track_history.values()]}")

    # 测试轨迹图绘制
    print("\n测试2D轨迹图...")
    trajectories = {
        1: [(100 + i * 5, 100 + i * 3) for i in range(20)],
        2: [(250 - i * 3, 150 + i * 2) for i in range(20)],
        3: [(400 + i * 4, 200 - i * 2) for i in range(20)]
    }

    output_dir = "/home/claude/Task2_DeepSORT_Tracking"
    os.makedirs(output_dir, exist_ok=True)

    plot_trajectory_2d(
        trajectories,
        save_path=os.path.join(output_dir, "test_trajectory.png"),
        show=False
    )

    # 测试统计图
    print("\n测试统计图...")
    test_results = []
    for frame in range(1, 51):
        for track_id in [1, 2, 3]:
            test_results.append([
                frame, track_id, 100, 100, 50, 100, 0.9, 0, -1, -1
            ])

    plot_tracking_stats(
        test_results,
        save_path=os.path.join(output_dir, "test_stats.png"),
        show=False
    )

    print("\n测试完成!")