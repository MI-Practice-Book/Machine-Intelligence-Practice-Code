"""
YOLOv5批量推理脚本
支持图像文件夹、视频文件的批量处理,自动保存检测结果和统计报告

Author: Machine Intelligence Practice
Date: 2025-01
"""

import logging
import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json
from collections import defaultdict, Counter

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger, TaskTimer, log_system_info


class YOLOv5Inferencer:
    """YOLOv5批量推理器"""

    def __init__(
            self,
            yolov5_root: Path,
            weights: Path,
            source: Path,
            project: Path,
            name: str
    ):
        """
        初始化推理器

        Args:
            yolov5_root: YOLOv5仓库根目录
            weights: 模型权重文件路径
            source: 输入源(图像文件夹/视频文件)
            project: 推理输出项目目录
            name: 本次推理的名称
        """
        self.yolov5_root = Path(yolov5_root)
        self.weights = Path(weights)
        self.source = Path(source)
        self.project = Path(project)
        self.name = name

        # 验证路径
        if not self.yolov5_root.exists():
            raise FileNotFoundError(f"YOLOv5目录不存在: {self.yolov5_root}")

        if not self.weights.exists():
            raise FileNotFoundError(f"模型权重不存在: {self.weights}")

        if not self.source.exists():
            raise FileNotFoundError(f"输入源不存在: {self.source}")

        self.detect_script = self.yolov5_root / "detect.py"
        if not self.detect_script.exists():
            raise FileNotFoundError(f"推理脚本不存在: {self.detect_script}")

        # 推理输出目录
        self.output_dir = self.project / self.name

    def build_command(
            self,
            img_size: int = 640,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            device: str = '0',
            max_det: int = 1000,
            save_txt: bool = True,
            save_conf: bool = True,
            save_crop: bool = False,
            nosave: bool = False,
            classes: list = None,
            agnostic_nms: bool = False,
            augment: bool = False,
            visualize: bool = False,
            line_thickness: int = 3,
            hide_labels: bool = False,
            hide_conf: bool = False,
            **kwargs
    ) -> list:
        """
        构建推理命令

        Args:
            img_size: 输入图像尺寸
            conf_thres: 置信度阈值
            iou_thres: NMS IoU阈值
            device: GPU设备ID
            max_det: 每张图最大检测数
            save_txt: 是否保存txt格式结果
            save_conf: 是否在txt中保存置信度
            save_crop: 是否保存裁剪的检测框
            nosave: 不保存图像/视频
            classes: 只检测特定类别(列表)
            agnostic_nms: 类别无关的NMS
            augment: 增强推理
            visualize: 可视化特征
            line_thickness: 边界框线宽
            hide_labels: 隐藏标签
            hide_conf: 隐藏置信度
            **kwargs: 其他参数

        Returns:
            命令列表
        """
        cmd = [
            sys.executable,
            str(self.detect_script),
            '--weights', str(self.weights),
            '--source', str(self.source),
            '--img', str(img_size),
            '--conf-thres', str(conf_thres),
            '--iou-thres', str(iou_thres),
            '--device', device,
            '--max-det', str(max_det),
            '--project', str(self.project),
            '--name', self.name,
            '--line-thickness', str(line_thickness),
        ]

        # 布尔参数
        if save_txt:
            cmd.append('--save-txt')
        if save_conf:
            cmd.append('--save-conf')
        if save_crop:
            cmd.append('--save-crop')
        if nosave:
            cmd.append('--nosave')
        if agnostic_nms:
            cmd.append('--agnostic-nms')
        if augment:
            cmd.append('--augment')
        if visualize:
            cmd.append('--visualize')
        if hide_labels:
            cmd.append('--hide-labels')
        if hide_conf:
            cmd.append('--hide-conf')

        # 类别过滤
        if classes:
            cmd.extend(['--classes', *[str(c) for c in classes]])

        # 其他自定义参数
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        return cmd

    def infer(self, **kwargs) -> bool:
        """
        执行批量推理

        Args:
            **kwargs: 推理参数(传递给build_command)

        Returns:
            推理是否成功
        """
        with TaskTimer(f"批量推理: {self.source.name}"):
            # 构建命令
            cmd = self.build_command(**kwargs)

            # 记录推理配置
            logging.info("=" * 70)
            logging.info("推理配置:")
            logging.info(f"  模型权重: {self.weights}")
            logging.info(f"  输入源: {self.source}")
            logging.info(f"  输出目录: {self.output_dir}")
            logging.info(f"  图像尺寸: {kwargs.get('img_size', 640)}")
            logging.info(f"  置信度阈值: {kwargs.get('conf_thres', 0.25)}")
            logging.info(f"  IoU阈值: {kwargs.get('iou_thres', 0.45)}")
            logging.info(f"  设备: {kwargs.get('device', '0')}")
            logging.info("=" * 70)

            # 记录完整命令
            logging.info("\n执行命令:")
            logging.info(" ".join(cmd))
            logging.info("")

            try:
                # 执行推理
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                # 实时输出日志
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        logging.info(line)

                # 等待进程结束
                return_code = process.wait()

                if return_code == 0:
                    logging.info("=" * 70)
                    logging.info("✅ 推理完成!")
                    logging.info(f"输出目录: {self.output_dir}")
                    logging.info("=" * 70)
                    return True
                else:
                    logging.error(f"❌ 推理失败! 返回码: {return_code}")
                    return False

            except Exception as e:
                logging.error(f"❌ 推理过程出错: {e}")
                return False

    def analyze_results(self, class_names: list = None):
        """
        分析推理结果

        Args:
            class_names: 类别名称列表
        """
        with TaskTimer("分析推理结果"):
            labels_dir = self.output_dir / "labels"

            if not labels_dir.exists():
                logging.warning(f"标注目录不存在: {labels_dir}")
                return

            # 统计信息
            stats = {
                'total_images': 0,
                'total_detections': 0,
                'class_distribution': defaultdict(int),
                'confidence_stats': [],
                'detections_per_image': []
            }

            # 遍历所有txt文件
            label_files = list(labels_dir.glob('*.txt'))
            stats['total_images'] = len(label_files)

            logging.info(f"找到 {len(label_files)} 个结果文件")

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()

                    num_detections = len(lines)
                    stats['detections_per_image'].append(num_detections)
                    stats['total_detections'] += num_detections

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            stats['class_distribution'][class_id] += 1

                            # 如果有置信度
                            if len(parts) == 6:
                                confidence = float(parts[5])
                                stats['confidence_stats'].append(confidence)

                except Exception as e:
                    logging.error(f"分析文件失败 {label_file.name}: {e}")

            # 生成报告
            self._generate_report(stats, class_names)

            # 保存统计结果
            self._save_statistics(stats, class_names)

    def _generate_report(self, stats: dict, class_names: list = None):
        """
        生成统计报告

        Args:
            stats: 统计数据
            class_names: 类别名称列表
        """
        logging.info("=" * 70)
        logging.info("推理统计报告")
        logging.info("=" * 70)

        # 基本统计
        logging.info("\n📊 基本统计:")
        logging.info(f"  处理图像数: {stats['total_images']}")
        logging.info(f"  总检测数: {stats['total_detections']}")

        if stats['total_images'] > 0:
            avg_detections = stats['total_detections'] / stats['total_images']
            logging.info(f"  平均每张检测数: {avg_detections:.2f}")

        # 类别分布
        if len(stats['class_distribution']) > 0:
            logging.info("\n📈 类别分布:")
            sorted_classes = sorted(
                stats['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            for class_id, count in sorted_classes:
                if class_names and class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"类别{class_id}"

                percentage = (count / stats['total_detections']) * 100
                logging.info(f"  {class_name:15s}: {count:6d} ({percentage:5.2f}%)")

        # 置信度统计
        if len(stats['confidence_stats']) > 0:
            import numpy as np
            confidences = stats['confidence_stats']

            logging.info("\n📉 置信度统计:")
            logging.info(f"  平均: {np.mean(confidences):.4f}")
            logging.info(f"  中位数: {np.median(confidences):.4f}")
            logging.info(f"  最小值: {np.min(confidences):.4f}")
            logging.info(f"  最大值: {np.max(confidences):.4f}")
            logging.info(f"  标准差: {np.std(confidences):.4f}")

        logging.info("=" * 70 + "\n")

    def _save_statistics(self, stats: dict, class_names: list = None):
        """
        保存统计信息到JSON

        Args:
            stats: 统计数据
            class_names: 类别名称列表
        """
        # 转换为可序列化格式
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': str(self.source),
            'weights': str(self.weights),
            'total_images': stats['total_images'],
            'total_detections': stats['total_detections'],
            'class_distribution': {}
        }

        # 类别分布
        for class_id, count in stats['class_distribution'].items():
            if class_names and class_id < len(class_names):
                key = class_names[class_id]
            else:
                key = f"class_{class_id}"
            report['class_distribution'][key] = count

        # 置信度统计
        if len(stats['confidence_stats']) > 0:
            import numpy as np
            report['confidence_statistics'] = {
                'mean': float(np.mean(stats['confidence_stats'])),
                'median': float(np.median(stats['confidence_stats'])),
                'min': float(np.min(stats['confidence_stats'])),
                'max': float(np.max(stats['confidence_stats'])),
                'std': float(np.std(stats['confidence_stats']))
            }

        # 保存
        report_file = self.output_dir / "inference_statistics.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info(f"统计报告已保存: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv5批量推理脚本")

    # 必需参数
    parser.add_argument(
        '--yolov5_root',
        type=str,
        required=True,
        help='YOLOv5仓库根目录'
    )

    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='模型权重文件路径'
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='输入源(图像文件夹/视频文件)'
    )

    # 基本参数
    parser.add_argument('--project', type=str, default='runs/detect', help='输出项目目录')
    parser.add_argument('--name', type=str, default='exp', help='本次推理名称')
    parser.add_argument('--img', '--img-size', type=int, default=640, dest='img_size', help='输入图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--max-det', type=int, default=1000, help='每张图最大检测数')

    # 输出参数
    parser.add_argument('--save-txt', action='store_true', default=True, help='保存txt格式结果')
    parser.add_argument('--save-conf', action='store_true', default=True, help='保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的检测框')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')

    # 检测参数
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='只检测特定类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--line-thickness', type=int, default=3, help='边界框线宽')
    parser.add_argument('--hide-labels', action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true', help='隐藏置信度')

    # 分析参数
    parser.add_argument('--analyze', action='store_true', default=True, help='分析推理结果')
    parser.add_argument('--class-names', nargs='+', type=str, default=None, help='类别名称列表')

    # 其他参数
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')

    args = parser.parse_args()

    # 设置日志
    log_file = args.log_file
    if log_file is None:
        log_dir = Path(args.project) / args.name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"infer_{timestamp}.log"

    setup_logger(log_file=str(log_file))

    logging.info("=" * 70)
    logging.info("YOLOv5批量推理工具")
    logging.info("=" * 70)
    logging.info(f"推理名称: {args.name}")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 70 + "\n")

    # 记录系统信息
    log_system_info()

    # 创建推理器
    inferencer = YOLOv5Inferencer(
        yolov5_root=args.yolov5_root,
        weights=args.weights,
        source=args.source,
        project=args.project,
        name=args.name
    )

    # 准备推理参数
    infer_config = {
        'img_size': args.img_size,
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'device': args.device,
        'max_det': args.max_det,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'nosave': args.nosave,
        'classes': args.classes,
        'agnostic_nms': args.agnostic_nms,
        'augment': args.augment,
        'line_thickness': args.line_thickness,
        'hide_labels': args.hide_labels,
        'hide_conf': args.hide_conf,
    }

    # 开始推理
    success = inferencer.infer(**infer_config)

    # 分析结果
    if success and args.analyze:
        inferencer.analyze_results(class_names=args.class_names)

    if success:
        logging.info("✅ 推理成功完成!")
        return 0
    else:
        logging.error("❌ 推理失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())