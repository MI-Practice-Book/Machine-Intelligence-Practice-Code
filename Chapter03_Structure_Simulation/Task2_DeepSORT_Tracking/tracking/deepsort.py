"""
DeepSORT主系统模块
整合目标检测、特征提取和多目标跟踪,提供完整的视频跟踪pipeline

主要功能:
- 集成YOLOv5检测器和Re-ID特征提取
- 管理Tracker进行多目标跟踪
- 提供视频处理和结果输出接口
- 支持实时跟踪和离线处理
"""

import numpy as np
import cv2
import logging
from typing import List, Optional, Tuple
import time
import os

from .tracker import Tracker
from models.yolov5_detector import YOLOv5Detector
from models.reid_model import FeatureExtractor
from utils.detection import Detection

# 配置日志
logger = logging.getLogger(__name__)


class DeepSORT:
    """
    DeepSORT多目标跟踪系统

    完整的端到端跟踪系统:
    1. 使用YOLOv5进行目标检测
    2. 使用Re-ID网络提取外观特征
    3. 使用Tracker执行多目标跟踪
    4. 输出带有唯一ID的跟踪结果

    Attributes:
        detector: YOLOv5检测器
        extractor: Re-ID特征提取器
        tracker: 多目标跟踪器
    """

    def __init__(
            self,
            detector_path: str = 'yolov5s',
            reid_model_path: str = None,
            max_dist: float = 0.2,
            max_iou_distance: float = 0.7,
            max_age: int = 30,
            n_init: int = 3,
            nn_budget: int = 100,
            use_cuda: bool = True
    ):
        """
        初始化DeepSORT系统

        Args:
            detector_path: YOLOv5模型路径或名称
            reid_model_path: Re-ID模型路径
            max_dist: 外观匹配的最大余弦距离
            max_iou_distance: IoU匹配的最大距离
            max_age: 轨迹最大允许的未匹配帧数
            n_init: 轨迹确认所需的连续匹配次数
            nn_budget: 每条轨迹的特征库容量
            use_cuda: 是否使用GPU
        """
        logger.info("=" * 50)
        logger.info("初始化DeepSORT系统")
        logger.info("=" * 50)

        # 初始化检测器
        logger.info("加载目标检测器...")
        self.detector = YOLOv5Detector(
            model_path=detector_path,
            conf_thresh=0.5,
            iou_thresh=0.45,
            device='cuda' if use_cuda else 'cpu',
            use_official=True
        )

        # 初始化特征提取器
        logger.info("加载Re-ID特征提取器...")
        self.extractor = FeatureExtractor(
            model_path=reid_model_path,
            device='cuda' if use_cuda else 'cpu',
            feature_dim=128,
            use_cuda=use_cuda
        )

        # 初始化跟踪器
        logger.info("初始化多目标跟踪器...")
        self.tracker = Tracker(
            metric='cosine',
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cosine_distance=max_dist
        )

        logger.info("DeepSORT系统初始化完成")

    def update(self, frame: np.ndarray) -> List[Tuple]:
        """
        处理单帧图像并更新跟踪

        完整流程:
        1. 目标检测
        2. 特征提取
        3. 跟踪更新
        4. 返回结果

        Args:
            frame: 输入图像(BGR格式)

        Returns:
            outputs: 跟踪结果列表 [(x1, y1, x2, y2, track_id, conf, class_id), ...]
        """
        start_time = time.time()

        # 步骤1: 目标检测
        det_start = time.time()
        raw_detections = self.detector.detect(frame)
        det_time = time.time() - det_start

        if len(raw_detections) == 0:
            # 无检测结果,仅执行预测
            self.tracker.predict()
            self.tracker.update([])
            return []

        # 步骤2: 特征提取
        feat_start = time.time()
        bboxes = [det[:4] for det in raw_detections]  # [left, top, width, height]
        features = self.extractor.extract_batch(bboxes, frame)
        feat_time = time.time() - feat_start

        # 步骤3: 构建Detection对象
        detections = []
        for i, det in enumerate(raw_detections):
            left, top, width, height, conf, class_id = det

            detection = Detection(
                tlwh=np.array([left, top, width, height]),
                confidence=conf,
                class_id=int(class_id),
                feature=features[i]
            )
            detections.append(detection)

        # 步骤4: 跟踪更新
        track_start = time.time()
        self.tracker.predict()
        self.tracker.update(detections)
        track_time = time.time() - track_start

        # 步骤5: 提取输出结果
        outputs = []
        for track in self.tracker.get_confirmed_tracks():
            bbox = track.to_tlbr()  # [left, top, right, bottom]
            track_id = track.track_id

            # 获取对应的检测置信度(近似)
            conf = 1.0  # 确认轨迹默认置信度为1.0
            class_id = 0  # 行人类别

            outputs.append((
                bbox[0], bbox[1], bbox[2], bbox[3],
                track_id, conf, class_id
            ))

        total_time = time.time() - start_time
        logger.debug(
            f"帧处理完成: 检测{len(detections)}个, 跟踪{len(outputs)}个, "
            f"总耗时={total_time * 1000:.1f}ms "
            f"(检测:{det_time * 1000:.1f}ms, "
            f"特征:{feat_time * 1000:.1f}ms, "
            f"跟踪:{track_time * 1000:.1f}ms)"
        )

        return outputs

    def track_video(
            self,
            video_path: str,
            output_path: Optional[str] = None,
            show: bool = False,
            save_results: bool = True
    ) -> List:
        """
        处理完整视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径(None则不保存)
            show: 是否显示跟踪结果
            save_results: 是否保存跟踪结果到txt

        Returns:
            results: 所有帧的跟踪结果
        """
        logger.info("=" * 50)
        logger.info(f"开始处理视频: {video_path}")
        logger.info("=" * 50)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        # 准备输出视频
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"保存输出视频到: {output_path}")

        # 处理每一帧
        results = []
        frame_id = 0
        total_time = 0

        logger.info("开始处理帧...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            frame_start = time.time()

            # 跟踪更新
            outputs = self.update(frame)

            frame_time = time.time() - frame_start
            total_time += frame_time

            # 记录结果
            for output in outputs:
                x1, y1, x2, y2, track_id, conf, class_id = output
                results.append([
                    frame_id, track_id,
                    x1, y1, x2 - x1, y2 - y1,  # 转换为tlwh
                    conf, class_id, -1, -1
                ])

            # 可视化
            if show or writer is not None:
                vis_frame = self.visualize(frame, outputs)

                if show:
                    cv2.imshow('DeepSORT Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if writer is not None:
                    writer.write(vis_frame)

            # 进度显示
            if frame_id % 10 == 0:
                avg_fps = frame_id / total_time
                progress = frame_id / total_frames * 100
                logger.info(
                    f"进度: {frame_id}/{total_frames} ({progress:.1f}%), "
                    f"平均FPS: {avg_fps:.2f}, "
                    f"当前跟踪: {len(outputs)}个目标"
                )

        # 释放资源
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        # 保存结果
        if save_results:
            result_path = video_path.replace('.mp4', '_results.txt')
            self.save_results(results, result_path)

        avg_fps = frame_id / total_time
        logger.info("=" * 50)
        logger.info(f"视频处理完成!")
        logger.info(f"总帧数: {frame_id}, 总耗时: {total_time:.2f}s")
        logger.info(f"平均FPS: {avg_fps:.2f}")
        logger.info("=" * 50)

        return results

    def visualize(
            self,
            frame: np.ndarray,
            outputs: List[Tuple],
            show_id: bool = True,
            show_conf: bool = False
    ) -> np.ndarray:
        """
        可视化跟踪结果

        Args:
            frame: 原始图像
            outputs: 跟踪结果
            show_id: 是否显示ID
            show_conf: 是否显示置信度

        Returns:
            vis_frame: 可视化后的图像
        """
        vis_frame = frame.copy()

        # 为不同ID生成不同颜色
        colors = self._get_colors(len(outputs))

        for i, output in enumerate(outputs):
            x1, y1, x2, y2, track_id, conf, class_id = output
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 选择颜色
            color = colors[track_id % len(colors)]

            # 绘制边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"ID:{track_id}"
            if show_conf:
                label += f" {conf:.2f}"

            # 标签背景
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color, -1
            )

            # 标签文字
            cv2.putText(
                vis_frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

        return vis_frame

    def _get_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        生成n种不同的颜色

        Args:
            n: 颜色数量

        Returns:
            colors: BGR颜色列表
        """
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (0, 255, 128), (128, 0, 255), (0, 128, 255)
        ]

        # 如果需要更多颜色,循环使用
        while len(colors) < n:
            colors.extend(colors)

        return colors[:n]

    def save_results(self, results: List, output_path: str):
        """
        保存跟踪结果到文件(MOT格式)

        格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>,
              <conf>, <x>, <y>, <z>

        Args:
            results: 跟踪结果列表
            output_path: 输出文件路径
        """
        logger.info(f"保存跟踪结果到: {output_path}")

        with open(output_path, 'w') as f:
            for result in results:
                line = ','.join([str(x) for x in result]) + '\n'
                f.write(line)

        logger.info(f"保存完成: {len(results)}条记录")

    def reset(self):
        """重置跟踪器"""
        self.tracker.reset()
        logger.info("DeepSORT系统已重置")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 50)
    print("测试DeepSORT系统")
    print("=" * 50)

    # 初始化系统
    print("\n初始化DeepSORT系统...")
    deepsort = DeepSORT(
        detector_path='yolov5s',
        reid_model_path=None,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        use_cuda=False  # 测试使用CPU
    )

    # 创建测试图像序列
    print("\n创建测试图像序列...")
    test_frames = []

    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 绘制移动的矩形(模拟目标)
        x = 100 + i * 50
        cv2.rectangle(frame, (x, 100), (x + 50, 200), (255, 0, 0), -1)

        x2 = 300 - i * 30
        cv2.rectangle(frame, (x2, 150), (x2 + 60, 270), (0, 255, 0), -1)

        test_frames.append(frame)

    # 处理每一帧
    print("\n处理测试帧...")
    all_results = []

    for frame_id, frame in enumerate(test_frames, 1):
        print(f"\n--- 第{frame_id}帧 ---")

        outputs = deepsort.update(frame)

        print(f"检测到目标数: {len(outputs)}")
        for output in outputs:
            x1, y1, x2, y2, track_id, conf, class_id = output
            print(f"  轨迹{track_id}: bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        # 记录结果
        for output in outputs:
            x1, y1, x2, y2, track_id, conf, class_id = output
            all_results.append([
                frame_id, track_id,
                x1, y1, x2 - x1, y2 - y1,
                conf, class_id, -1, -1
            ])

    # 可视化最后一帧
    print("\n可视化最后一帧...")
    last_frame = test_frames[-1]
    outputs = deepsort.update(last_frame)
    vis_frame = deepsort.visualize(last_frame, outputs)

    print(f"可视化帧形状: {vis_frame.shape}")

    # 保存结果
    print("\n保存测试结果...")
    output_path = '/home/claude/Task2_DeepSORT_Tracking/test_results.txt'
    deepsort.save_results(all_results, output_path)

    # 统计信息
    print("\n" + "=" * 50)
    print("测试统计")
    print("=" * 50)

    track_count = deepsort.tracker.get_track_count()
    print(f"总帧数: {len(test_frames)}")
    print(f"总记录数: {len(all_results)}")
    print(f"轨迹统计: {track_count}")

    # 测试重置
    print("\n测试重置功能...")
    print(f"重置前轨迹数: {len(deepsort.tracker.tracks)}")
    deepsort.reset()
    print(f"重置后轨迹数: {len(deepsort.tracker.tracks)}")

    print("\n测试完成!")