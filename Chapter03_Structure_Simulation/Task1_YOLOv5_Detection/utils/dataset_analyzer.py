"""
数据集分析工具
提供数据集统计、质量检查和可视化分析功能

Author: Machine Intelligence Practice
Date: 2025-01
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np

from .logger import TaskTimer, ProgressLogger


class DatasetAnalyzer:
    """数据集分析器"""

    def __init__(self, class_names: List[str]):
        """
        初始化分析器

        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

        # 统计数据
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int),
            'bbox_sizes': [],
            'bbox_aspect_ratios': [],
            'annotations_per_image': [],
            'empty_images': 0,
            'problematic_images': []
        }

    def analyze_annotation_file(self, label_file: Path, image_size: Tuple[int, int] = (640, 640)) -> Dict:
        """
        分析单个标注文件

        Args:
            label_file: YOLO格式标注文件路径
            image_size: 图像尺寸 (width, height)

        Returns:
            该文件的分析结果
        """
        img_width, img_height = image_size
        file_stats = {
            'num_annotations': 0,
            'classes': [],
            'bbox_areas': [],
            'has_issue': False,
            'issues': []
        }

        try:
            if not label_file.exists():
                file_stats['has_issue'] = True
                file_stats['issues'].append("文件不存在")
                return file_stats

            with open(label_file, 'r') as f:
                lines = f.readlines()

            if len(lines) == 0:
                file_stats['has_issue'] = True
                file_stats['issues'].append("空文件")
                self.stats['empty_images'] += 1
                return file_stats

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split()
                    if len(parts) != 5:
                        file_stats['has_issue'] = True
                        file_stats['issues'].append(f"第{line_idx + 1}行格式错误")
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 验证类别ID
                    if class_id < 0 or class_id >= self.num_classes:
                        file_stats['has_issue'] = True
                        file_stats['issues'].append(f"第{line_idx + 1}行类别ID越界: {class_id}")
                        continue

                    # 验证坐标范围
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        file_stats['has_issue'] = True
                        file_stats['issues'].append(f"第{line_idx + 1}行中心点坐标越界")
                        continue

                    if not (0 < width <= 1 and 0 < height <= 1):
                        file_stats['has_issue'] = True
                        file_stats['issues'].append(f"第{line_idx + 1}行宽高越界")
                        continue

                    # 记录统计信息
                    file_stats['num_annotations'] += 1
                    file_stats['classes'].append(class_id)

                    # 计算实际像素尺寸
                    actual_width = width * img_width
                    actual_height = height * img_height
                    area = actual_width * actual_height
                    aspect_ratio = actual_width / actual_height if actual_height > 0 else 0

                    file_stats['bbox_areas'].append(area)

                    # 全局统计
                    self.stats['class_distribution'][class_id] += 1
                    self.stats['bbox_sizes'].append((actual_width, actual_height))
                    self.stats['bbox_aspect_ratios'].append(aspect_ratio)

                    # 检测异常尺寸
                    if actual_width < 10 or actual_height < 10:
                        file_stats['has_issue'] = True
                        file_stats['issues'].append(f"第{line_idx + 1}行边界框过小")

                    if aspect_ratio > 10 or aspect_ratio < 0.1:
                        file_stats['has_issue'] = True
                        file_stats['issues'].append(f"第{line_idx + 1}行宽高比异常: {aspect_ratio:.2f}")

                except ValueError as e:
                    file_stats['has_issue'] = True
                    file_stats['issues'].append(f"第{line_idx + 1}行数据解析错误: {e}")
                    continue

            self.stats['annotations_per_image'].append(file_stats['num_annotations'])
            self.stats['total_annotations'] += file_stats['num_annotations']

        except Exception as e:
            file_stats['has_issue'] = True
            file_stats['issues'].append(f"文件读取错误: {e}")
            logging.error(f"分析文件失败 {label_file.name}: {e}")

        return file_stats

    def analyze_dataset(
            self,
            labels_dir: Path,
            image_size: Tuple[int, int] = (640, 640),
            subset: str = 'train'
    ) -> Dict:
        """
        分析整个数据集

        Args:
            labels_dir: 标注文件目录
            image_size: 图像尺寸
            subset: 数据集子集名称

        Returns:
            分析统计结果
        """
        with TaskTimer(f"分析{subset}集"):
            # 获取所有标注文件
            label_files = list(labels_dir.glob('*.txt'))

            if len(label_files) == 0:
                logging.warning(f"未找到标注文件: {labels_dir}")
                return self.stats

            logging.info(f"找到 {len(label_files)} 个标注文件")

            # 重置统计数据
            self.stats['total_images'] = len(label_files)

            # 批量分析
            progress = ProgressLogger(total=len(label_files), task_name=f"{subset}集分析")

            for label_file in label_files:
                file_stats = self.analyze_annotation_file(label_file, image_size)

                if file_stats['has_issue']:
                    self.stats['problematic_images'].append({
                        'filename': label_file.name,
                        'issues': file_stats['issues']
                    })

                progress.update()

            progress.finish()

            # 生成分析报告
            self._generate_report(subset)

            return self.stats.copy()

    def _generate_report(self, subset: str):
        """
        生成分析报告

        Args:
            subset: 数据集子集名称
        """
        logging.info("=" * 70)
        logging.info(f"{subset}集数据分析报告")
        logging.info("=" * 70)

        # 基本统计
        logging.info("\n📊 基本统计:")
        logging.info(f"  总图像数: {self.stats['total_images']}")
        logging.info(f"  总标注数: {self.stats['total_annotations']}")
        logging.info(f"  空标注图像数: {self.stats['empty_images']}")
        logging.info(f"  问题图像数: {len(self.stats['problematic_images'])}")

        if self.stats['total_images'] > 0:
            avg_annotations = self.stats['total_annotations'] / self.stats['total_images']
            logging.info(f"  平均每张图像标注数: {avg_annotations:.2f}")

        # 类别分布
        logging.info("\n📈 类别分布:")
        class_dist = self.stats['class_distribution']

        if len(class_dist) > 0:
            # 按数量排序
            sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)

            for class_id, count in sorted_classes:
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"未知类别{class_id}"
                percentage = (count / self.stats['total_annotations']) * 100
                logging.info(f"  {class_name:15s}: {count:6d} ({percentage:5.2f}%)")

            # 类别不平衡检测
            max_count = sorted_classes[0][1]
            min_count = sorted_classes[-1][1]
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 10:
                logging.warning(f"  ⚠️ 检测到类别不平衡! 最大/最小比例: {imbalance_ratio:.2f}")

        # 边界框尺寸分析
        if len(self.stats['bbox_sizes']) > 0:
            logging.info("\n📐 边界框尺寸分析:")

            widths = [size[0] for size in self.stats['bbox_sizes']]
            heights = [size[1] for size in self.stats['bbox_sizes']]
            areas = [w * h for w, h in self.stats['bbox_sizes']]

            logging.info(f"  宽度 - 平均: {np.mean(widths):.2f}, 中位数: {np.median(widths):.2f}, "
                         f"范围: [{np.min(widths):.2f}, {np.max(widths):.2f}]")
            logging.info(f"  高度 - 平均: {np.mean(heights):.2f}, 中位数: {np.median(heights):.2f}, "
                         f"范围: [{np.min(heights):.2f}, {np.max(heights):.2f}]")
            logging.info(f"  面积 - 平均: {np.mean(areas):.2f}, 中位数: {np.median(areas):.2f}")

            # 尺寸分布
            small_boxes = sum(1 for a in areas if a < 32 * 32)
            medium_boxes = sum(1 for a in areas if 32 * 32 <= a < 96 * 96)
            large_boxes = sum(1 for a in areas if a >= 96 * 96)

            total_boxes = len(areas)
            logging.info(f"  小目标 (<32²): {small_boxes} ({small_boxes / total_boxes * 100:.2f}%)")
            logging.info(f"  中目标 (32²~96²): {medium_boxes} ({medium_boxes / total_boxes * 100:.2f}%)")
            logging.info(f"  大目标 (≥96²): {large_boxes} ({large_boxes / total_boxes * 100:.2f}%)")

        # 宽高比分析
        if len(self.stats['bbox_aspect_ratios']) > 0:
            logging.info("\n📏 宽高比分析:")
            aspect_ratios = self.stats['bbox_aspect_ratios']
            logging.info(f"  平均: {np.mean(aspect_ratios):.2f}")
            logging.info(f"  中位数: {np.median(aspect_ratios):.2f}")
            logging.info(f"  范围: [{np.min(aspect_ratios):.2f}, {np.max(aspect_ratios):.2f}]")

            extreme_ratios = sum(1 for r in aspect_ratios if r > 5 or r < 0.2)
            if extreme_ratios > 0:
                logging.warning(
                    f"  ⚠️ 极端宽高比数量: {extreme_ratios} ({extreme_ratios / len(aspect_ratios) * 100:.2f}%)")

        # 问题图像列表
        if len(self.stats['problematic_images']) > 0:
            logging.info(f"\n⚠️ 问题图像详情 (前10个):")
            for item in self.stats['problematic_images'][:10]:
                logging.info(f"  {item['filename']}:")
                for issue in item['issues']:
                    logging.info(f"    - {issue}")

        logging.info("=" * 70 + "\n")

    def save_report(self, output_file: Path):
        """
        保存分析报告到JSON文件

        Args:
            output_file: 输出文件路径
        """
        # 转换numpy类型为Python原生类型
        report = {
            'total_images': self.stats['total_images'],
            'total_annotations': self.stats['total_annotations'],
            'empty_images': self.stats['empty_images'],
            'class_distribution': dict(self.stats['class_distribution']),
            'problematic_images_count': len(self.stats['problematic_images']),
            'problematic_images': self.stats['problematic_images'][:50]  # 只保存前50个
        }

        # 统计信息
        if len(self.stats['bbox_sizes']) > 0:
            widths = [size[0] for size in self.stats['bbox_sizes']]
            heights = [size[1] for size in self.stats['bbox_sizes']]

            report['bbox_statistics'] = {
                'width': {
                    'mean': float(np.mean(widths)),
                    'median': float(np.median(widths)),
                    'min': float(np.min(widths)),
                    'max': float(np.max(widths))
                },
                'height': {
                    'mean': float(np.mean(heights)),
                    'median': float(np.median(heights)),
                    'min': float(np.min(heights)),
                    'max': float(np.max(heights))
                }
            }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info(f"分析报告已保存: {output_file}")


if __name__ == "__main__":
    from .logger import setup_logger

    # 测试代码
    setup_logger(log_file="logs/dataset_analyzer_test.log")

    # VOC类别
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    analyzer = DatasetAnalyzer(class_names=voc_classes)

    logging.info("数据集分析器初始化完成")