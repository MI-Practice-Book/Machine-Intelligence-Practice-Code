"""
DeepSORT跟踪主运行脚本
提供命令行接口运行多目标跟踪

使用方法:
    # 跟踪视频文件
    python run_tracking.py --video path/to/video.mp4 --output output.mp4

    # 跟踪MOT数据集
    python run_tracking.py --mot_root path/to/MOT17 --sequence MOT17-02-DPM

    # 使用自定义配置
    python run_tracking.py --video video.mp4 --config config.yaml
"""

import argparse
import os
import sys
import logging
import time
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking.deepsort import DeepSORT
from data.mot_dataset import MOTDataset, MOTSequence
from utils.visualization import save_visualization_video, plot_tracking_stats
import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepSORT多目标跟踪')

    # 输入源
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='输入视频路径')
    input_group.add_argument('--mot_root', type=str, help='MOT数据集根目录')
    input_group.add_argument('--camera', type=int, help='摄像头ID')

    # MOT数据集相关
    parser.add_argument('--sequence', type=str, help='MOT序列名称')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'], help='数据集划分')

    # 输出
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--save_results', type=str, help='保存跟踪结果路径(.txt)')
    parser.add_argument('--save_stats', type=str, help='保存统计图路径(.png)')

    # 模型配置
    parser.add_argument('--yolo_model', type=str, default='yolov5s',
                        help='YOLOv5模型路径或名称')
    parser.add_argument('--reid_model', type=str,
                        help='Re-ID模型路径')
    parser.add_argument('--config', type=str,
                        help='YAML配置文件路径')

    # 跟踪参数
    parser.add_argument('--max_dist', type=float, default=0.2,
                        help='外观距离阈值')
    parser.add_argument('--max_iou_distance', type=float, default=0.7,
                        help='IoU距离阈值')
    parser.add_argument('--max_age', type=int, default=30,
                        help='轨迹最大年龄')
    parser.add_argument('--n_init', type=int, default=3,
                        help='轨迹确认所需帧数')
    parser.add_argument('--nn_budget', type=int, default=100,
                        help='特征库容量')

    # 显示选项
    parser.add_argument('--show', action='store_true',
                        help='实时显示跟踪结果')
    parser.add_argument('--show_trajectory', action='store_true',
                        help='显示轨迹')

    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    return config


def track_video(args, deepsort: DeepSORT):
    """
    跟踪视频文件

    Args:
        args: 命令行参数
        deepsort: DeepSORT系统
    """
    logger.info("=" * 50)
    logger.info("开始跟踪视频")
    logger.info("=" * 50)

    video_path = args.video

    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return

    # 跟踪处理
    results = deepsort.track_video(
        video_path=video_path,
        output_path=args.output,
        show=args.show,
        save_results=bool(args.save_results)
    )

    # 保存结果
    if args.save_results:
        deepsort.save_results(results, args.save_results)

    # 生成统计图
    if args.save_stats and len(results) > 0:
        logger.info(f"生成统计图: {args.save_stats}")
        plot_tracking_stats(results, save_path=args.save_stats, show=False)

    logger.info("视频跟踪完成!")


def track_mot_sequence(args, deepsort: DeepSORT):
    """
    跟踪MOT数据集序列

    Args:
        args: 命令行参数
        deepsort: DeepSORT系统
    """
    logger.info("=" * 50)
    logger.info("开始跟踪MOT序列")
    logger.info("=" * 50)

    # 加载数据集
    dataset = MOTDataset(args.mot_root, split=args.split)

    # 获取指定序列
    if args.sequence:
        sequence = dataset.get_sequence_by_name(args.sequence)
        if sequence is None:
            logger.error(f"序列不存在: {args.sequence}")
            return
        sequences = [sequence]
    else:
        sequences = dataset.sequences

    # 处理每个序列
    for seq in sequences:
        logger.info(f"\n处理序列: {seq.name}")

        # 重置跟踪器
        deepsort.reset()

        # 准备输出
        results = []
        vis_frames = []

        # 创建可视化器
        if args.show or args.output:
            from utils.visualization import TrackVisualizer
            visualizer = TrackVisualizer()

        # 处理每一帧
        start_time = time.time()

        for frame_id in range(1, len(seq) + 1):
            # 读取图像
            img = seq.get_frame(frame_id)
            if img is None:
                logger.warning(f"无法读取帧: {frame_id}")
                continue

            # 跟踪更新
            outputs = deepsort.update(img)

            # 记录结果
            for output in outputs:
                x1, y1, x2, y2, track_id, conf, class_id = output
                results.append([
                    frame_id, track_id,
                    x1, y1, x2 - x1, y2 - y1,
                    conf, class_id, -1, -1
                ])

            # 可视化
            if args.show or args.output:
                vis_img = visualizer.draw_boxes(
                    img, outputs, show_id=True, show_conf=False
                )
                if args.show_trajectory:
                    vis_img = visualizer.draw_trajectories(vis_img, outputs)

                if args.show:
                    cv2.imshow(f'DeepSORT - {seq.name}', vis_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if args.output:
                    vis_frames.append(vis_img)

            # 进度显示
            if frame_id % 50 == 0:
                progress = frame_id / len(seq) * 100
                logger.info(
                    f"进度: {frame_id}/{len(seq)} ({progress:.1f}%), "
                    f"当前跟踪: {len(outputs)}个目标"
                )

        elapsed_time = time.time() - start_time
        avg_fps = len(seq) / elapsed_time

        if args.show:
            cv2.destroyAllWindows()

        # 保存结果
        if args.save_results:
            result_path = args.save_results.replace(
                '.txt', f'_{seq.name}.txt'
            )
            deepsort.save_results(results, result_path)

        # 保存视频
        if args.output and len(vis_frames) > 0:
            from utils.visualization import create_video_from_frames
            output_path = args.output.replace(
                '.mp4', f'_{seq.name}.mp4'
            )
            create_video_from_frames(vis_frames, output_path, fps=seq.frame_rate)

        # 生成统计图
        if args.save_stats and len(results) > 0:
            stats_path = args.save_stats.replace(
                '.png', f'_{seq.name}.png'
            )
            plot_tracking_stats(results, save_path=stats_path, show=False)

        logger.info(
            f"序列{seq.name}完成: "
            f"总帧数={len(seq)}, "
            f"总耗时={elapsed_time:.2f}s, "
            f"平均FPS={avg_fps:.2f}"
        )

    logger.info("\nMOT序列跟踪完成!")


def track_camera(args, deepsort: DeepSORT):
    """
    跟踪摄像头

    Args:
        args: 命令行参数
        deepsort: DeepSORT系统
    """
    logger.info("=" * 50)
    logger.info(f"开始跟踪摄像头: {args.camera}")
    logger.info("=" * 50)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        logger.error(f"无法打开摄像头: {args.camera}")
        return

    from utils.visualization import TrackVisualizer
    visualizer = TrackVisualizer()

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跟踪更新
            outputs = deepsort.update(frame)

            # 可视化
            vis_frame = visualizer.draw_boxes(
                frame, outputs, show_id=True, show_conf=True
            )
            if args.show_trajectory:
                vis_frame = visualizer.draw_trajectories(vis_frame, outputs)

            # 显示FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                vis_frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )

            # 显示
            cv2.imshow('DeepSORT Camera Tracking', vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                deepsort.reset()
                logger.info("跟踪器已重置")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info(
        f"摄像头跟踪结束: "
        f"总帧数={frame_count}, "
        f"平均FPS={frame_count / (time.time() - start_time):.2f}"
    )


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("DeepSORT多目标跟踪系统")
    logger.info("=" * 50)

    # 加载配置文件
    config = {}
    if args.config:
        config = load_config(args.config)

    # 合并配置(命令行参数优先)
    tracking_params = {
        'detector_path': args.yolo_model,
        'reid_model_path': args.reid_model,
        'max_dist': args.max_dist,
        'max_iou_distance': args.max_iou_distance,
        'max_age': args.max_age,
        'n_init': args.n_init,
        'nn_budget': args.nn_budget,
        'use_cuda': (args.device == 'cuda')
    }

    # 配置文件中的参数作为默认值
    for key in tracking_params:
        if key in config and tracking_params[key] is None:
            tracking_params[key] = config[key]

    # 初始化DeepSORT
    logger.info("初始化DeepSORT系统...")
    deepsort = DeepSORT(**tracking_params)

    # 根据输入源执行不同的跟踪模式
    try:
        if args.video:
            track_video(args, deepsort)
        elif args.mot_root:
            track_mot_sequence(args, deepsort)
        elif args.camera is not None:
            track_camera(args, deepsort)
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)

    logger.info("\n程序结束")


if __name__ == "__main__":
    main()