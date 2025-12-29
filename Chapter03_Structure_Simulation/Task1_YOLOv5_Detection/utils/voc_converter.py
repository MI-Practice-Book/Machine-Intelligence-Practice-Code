"""
VOC格式到YOLO格式转换工具
解析VOC XML标注文件并转换为YOLO格式(归一化坐标)

Author: Machine Intelligence Practice
Date: 2025-01
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .logger import TaskTimer, ProgressLogger


@dataclass
class BoundingBox:
    """边界框数据类"""
    class_name: str
    class_id: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
        """
        转换为YOLO格式 (归一化坐标)

        Args:
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            (class_id, x_center, y_center, width, height) 归一化后的值
        """
        # 计算中心点坐标
        x_center = (self.xmin + self.xmax) / 2.0
        y_center = (self.ymin + self.ymax) / 2.0

        # 计算宽高
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        # 归一化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        # 确保坐标在 [0, 1] 范围内
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))

        return self.class_id, x_center_norm, y_center_norm, width_norm, height_norm

    def is_valid(self, img_width: int, img_height: int) -> bool:
        """
        检查边界框是否有效

        Args:
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            是否有效
        """
        # 检查坐标顺序
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            return False

        # 检查坐标范围
        if self.xmin < 0 or self.ymin < 0:
            return False
        if self.xmax > img_width or self.ymax > img_height:
            return False

        # 检查边界框大小(至少1个像素)
        if (self.xmax - self.xmin) < 1 or (self.ymax - self.ymin) < 1:
            return False

        return True


class VOCConverter:
    """VOC到YOLO格式转换器"""

    def __init__(self, class_names: List[str]):
        """
        初始化转换器

        Args:
            class_names: 类别名称列表,顺序对应类别ID
        """
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}

        # 统计信息
        self.stats = {
            'total_annotations': 0,
            'total_images': 0,
            'invalid_boxes': 0,
            'unknown_classes': 0,
            'successful_conversions': 0
        }

    def parse_voc_xml(self, xml_file: Path) -> Optional[Dict]:
        """
        解析VOC格式的XML文件

        Args:
            xml_file: XML文件路径

        Returns:
            包含图像信息和边界框的字典,解析失败返回None
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 获取图像尺寸
            size = root.find('size')
            if size is None:
                logging.warning(f"{xml_file.name}: 缺少size标签")
                return None

            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

            # 获取图像文件名
            filename = root.find('filename')
            if filename is None:
                logging.warning(f"{xml_file.name}: 缺少filename标签")
                return None
            img_filename = filename.text

            # 解析所有目标
            boxes = []
            for obj in root.findall('object'):
                # 获取类别名称
                class_name = obj.find('name').text

                # 检查类别是否在预定义列表中
                if class_name not in self.class_to_id:
                    logging.warning(f"{xml_file.name}: 未知类别 '{class_name}'")
                    self.stats['unknown_classes'] += 1
                    continue

                class_id = self.class_to_id[class_name]

                # 获取边界框坐标
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    logging.warning(f"{xml_file.name}: 目标缺少bndbox标签")
                    continue

                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))

                # 创建边界框对象
                bbox = BoundingBox(
                    class_name=class_name,
                    class_id=class_id,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax
                )

                # 验证边界框
                if not bbox.is_valid(img_width, img_height):
                    logging.warning(
                        f"{xml_file.name}: 无效边界框 "
                        f"[{xmin},{ymin},{xmax},{ymax}] "
                        f"图像尺寸:[{img_width},{img_height}]"
                    )
                    self.stats['invalid_boxes'] += 1
                    continue

                boxes.append(bbox)
                self.stats['total_annotations'] += 1

            if len(boxes) == 0:
                logging.warning(f"{xml_file.name}: 没有有效的标注")
                return None

            return {
                'filename': img_filename,
                'width': img_width,
                'height': img_height,
                'boxes': boxes
            }

        except Exception as e:
            logging.error(f"解析XML文件失败 {xml_file.name}: {e}")
            return None

    def convert_to_yolo(self, annotation: Dict) -> List[str]:
        """
        将标注转换为YOLO格式

        Args:
            annotation: 标注字典(来自parse_voc_xml)

        Returns:
            YOLO格式的标注行列表
        """
        yolo_lines = []
        img_width = annotation['width']
        img_height = annotation['height']

        for bbox in annotation['boxes']:
            class_id, x_center, y_center, width, height = bbox.to_yolo_format(
                img_width, img_height
            )

            # YOLO格式: class_id x_center y_center width height
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(line)

        return yolo_lines

    def convert_file(self, xml_file: Path, output_file: Path) -> bool:
        """
        转换单个XML文件到YOLO格式

        Args:
            xml_file: 输入的VOC XML文件
            output_file: 输出的YOLO txt文件

        Returns:
            转换是否成功
        """
        # 解析XML
        annotation = self.parse_voc_xml(xml_file)
        if annotation is None:
            return False

        # 转换为YOLO格式
        yolo_lines = self.convert_to_yolo(annotation)

        # 写入文件
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_lines))

            self.stats['successful_conversions'] += 1
            self.stats['total_images'] += 1
            return True

        except Exception as e:
            logging.error(f"写入YOLO文件失败 {output_file}: {e}")
            return False

    def convert_dataset(
            self,
            annotations_dir: Path,
            output_dir: Path,
            subset: str = 'train'
    ) -> Dict:
        """
        批量转换数据集

        Args:
            annotations_dir: VOC标注文件目录
            output_dir: YOLO标注文件输出目录
            subset: 数据集子集名称(train/val/test)

        Returns:
            转换统计信息
        """
        with TaskTimer(f"转换{subset}集VOC标注"):
            # 获取所有XML文件
            xml_files = list(annotations_dir.glob('*.xml'))

            if len(xml_files) == 0:
                logging.warning(f"未找到XML文件: {annotations_dir}")
                return self.stats

            logging.info(f"找到 {len(xml_files)} 个XML文件")

            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)

            # 批量转换
            progress = ProgressLogger(total=len(xml_files), task_name=f"{subset}集转换")

            for xml_file in xml_files:
                # 生成对应的输出文件名
                output_file = output_dir / f"{xml_file.stem}.txt"

                # 转换
                self.convert_file(xml_file, output_file)

                progress.update()

            progress.finish()

            # 输出统计信息
            self._log_statistics()

            return self.stats.copy()

    def _log_statistics(self):
        """记录转换统计信息"""
        logging.info("=" * 60)
        logging.info("转换统计:")
        logging.info(f"  处理图像数: {self.stats['total_images']}")
        logging.info(f"  成功转换数: {self.stats['successful_conversions']}")
        logging.info(f"  总标注数: {self.stats['total_annotations']}")
        logging.info(f"  无效边界框: {self.stats['invalid_boxes']}")
        logging.info(f"  未知类别: {self.stats['unknown_classes']}")

        if self.stats['total_images'] > 0:
            success_rate = (self.stats['successful_conversions'] / self.stats['total_images']) * 100
            logging.info(f"  成功率: {success_rate:.2f}%")

            if self.stats['successful_conversions'] > 0:
                avg_annotations = self.stats['total_annotations'] / self.stats['successful_conversions']
                logging.info(f"  平均每张图像标注数: {avg_annotations:.2f}")

        logging.info("=" * 60)


if __name__ == "__main__":
    from .logger import setup_logger

    # 测试代码
    setup_logger(log_file="logs/voc_converter_test.log")

    # VOC类别
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    converter = VOCConverter(class_names=voc_classes)

    # 示例:转换单个文件
    # converter.convert_file(
    #     xml_file=Path("data/VOC2012/Annotations/2007_000027.xml"),
    #     output_file=Path("data/labels/train/2007_000027.txt")
    # )

    logging.info("VOC转换器初始化完成")