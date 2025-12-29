"""
MOT数据集加载模块
实现MOT Challenge数据集(MOT17等)的加载和解析

数据集结构:
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── img1/          # 视频帧
│   │   ├── gt/            # 真值标注
│   │   │   └── gt.txt
│   │   ├── det/           # 检测结果
│   │   │   └── det.txt
│   │   └── seqinfo.ini    # 序列信息
│   └── ...
└── test/
    └── ...
"""

import os
import numpy as np
import cv2
import configparser
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)


class MOTSequence:
    """
    单个MOT序列类

    封装一个视频序列的所有信息:
    - 图像帧
    - 真值标注
    - 检测结果
    - 序列元数据

    Attributes:
        name: 序列名称
        img_dir: 图像目录
        seq_length: 序列长度(帧数)
        img_width: 图像宽度
        img_height: 图像高度
        frame_rate: 帧率
    """

    def __init__(self, seq_path: str):
        """
        初始化序列

        Args:
            seq_path: 序列目录路径
        """
        self.seq_path = seq_path
        self.name = os.path.basename(seq_path)

        # 读取序列配置
        self._load_seqinfo()

        # 设置路径
        self.img_dir = os.path.join(seq_path, 'img1')
        self.gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        self.det_file = os.path.join(seq_path, 'det', 'det.txt')

        # 加载标注
        self.gt_data = self._load_annotations(self.gt_file) if os.path.exists(self.gt_file) else None
        self.det_data = self._load_detections(self.det_file) if os.path.exists(self.det_file) else None

        logger.info(
            f"加载序列: {self.name}, "
            f"帧数={self.seq_length}, "
            f"尺寸={self.img_width}x{self.img_height}, "
            f"FPS={self.frame_rate}"
        )

    def _load_seqinfo(self):
        """读取seqinfo.ini配置文件"""
        seqinfo_path = os.path.join(self.seq_path, 'seqinfo.ini')

        if not os.path.exists(seqinfo_path):
            raise FileNotFoundError(f"未找到seqinfo.ini: {seqinfo_path}")

        config = configparser.ConfigParser()
        config.read(seqinfo_path)

        seq_info = config['Sequence']
        self.seq_length = int(seq_info['seqLength'])
        self.img_width = int(seq_info['imWidth'])
        self.img_height = int(seq_info['imHeight'])
        self.frame_rate = int(seq_info['frameRate'])
        self.img_ext = seq_info.get('imExt', '.jpg')

    def _load_annotations(self, file_path: str) -> Dict[int, np.ndarray]:
        """
        加载真值标注文件

        格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>,
              <conf>, <class>, <visibility>

        Args:
            file_path: gt.txt文件路径

        Returns:
            annotations: {frame_id: array([[id, left, top, w, h, conf, class, vis], ...])}
        """
        if not os.path.exists(file_path):
            return {}

        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)

        # 按帧组织数据
        annotations = {}
        for row in data:
            frame_id = int(row[0])
            if frame_id not in annotations:
                annotations[frame_id] = []

            # [id, left, top, width, height, conf, class, visibility]
            annotations[frame_id].append(row[1:9])

        # 转换为numpy数组
        for frame_id in annotations:
            annotations[frame_id] = np.array(annotations[frame_id])

        logger.debug(f"加载真值标注: {len(annotations)}帧")
        return annotations

    def _load_detections(self, file_path: str) -> Dict[int, np.ndarray]:
        """
        加载检测结果文件

        格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>

        Args:
            file_path: det.txt文件路径

        Returns:
            detections: {frame_id: array([[left, top, w, h, conf], ...])}
        """
        if not os.path.exists(file_path):
            return {}

        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)

        # 按帧组织数据
        detections = {}
        for row in data:
            frame_id = int(row[0])
            if frame_id not in detections:
                detections[frame_id] = []

            # [left, top, width, height, conf]
            detections[frame_id].append(row[2:7])

        # 转换为numpy数组
        for frame_id in detections:
            detections[frame_id] = np.array(detections[frame_id])

        logger.debug(f"加载检测结果: {len(detections)}帧")
        return detections

    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        获取指定帧的图像

        Args:
            frame_id: 帧ID (从1开始)

        Returns:
            img: BGR格式图像,如果不存在返回None
        """
        img_path = os.path.join(self.img_dir, f'{frame_id:06d}{self.img_ext}')

        if not os.path.exists(img_path):
            logger.warning(f"图像不存在: {img_path}")
            return None

        img = cv2.imread(img_path)
        return img

    def get_gt(self, frame_id: int) -> Optional[np.ndarray]:
        """
        获取指定帧的真值标注

        Args:
            frame_id: 帧ID

        Returns:
            gt: 标注数组 [[id, left, top, w, h, conf, class, vis], ...]
        """
        if self.gt_data is None or frame_id not in self.gt_data:
            return None
        return self.gt_data[frame_id]

    def get_detections(self, frame_id: int) -> Optional[np.ndarray]:
        """
        获取指定帧的检测结果

        Args:
            frame_id: 帧ID

        Returns:
            dets: 检测数组 [[left, top, w, h, conf], ...]
        """
        if self.det_data is None or frame_id not in self.det_data:
            return None
        return self.det_data[frame_id]

    def __len__(self) -> int:
        """返回序列长度"""
        return self.seq_length

    def __iter__(self):
        """迭代器:遍历所有帧"""
        for frame_id in range(1, self.seq_length + 1):
            img = self.get_frame(frame_id)
            gt = self.get_gt(frame_id)
            dets = self.get_detections(frame_id)
            yield frame_id, img, gt, dets

    def get_info(self) -> Dict:
        """
        获取序列详细信息

        Returns:
            info: 信息字典
        """
        return {
            'name': self.name,
            'path': self.seq_path,
            'length': self.seq_length,
            'width': self.img_width,
            'height': self.img_height,
            'fps': self.frame_rate,
            'has_gt': self.gt_data is not None,
            'has_det': self.det_data is not None
        }


class MOTDataset:
    """
    MOT数据集类

    管理整个MOT数据集(包含多个序列)
    支持train/test划分

    Attributes:
        root: 数据集根目录
        sequences: 序列列表
    """

    def __init__(self, root: str, split: str = 'train'):
        """
        初始化数据集

        Args:
            root: 数据集根目录(如 'MOT17')
            split: 数据划分 ('train' or 'test')
        """
        self.root = root
        self.split = split
        self.split_dir = os.path.join(root, split)

        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"数据集路径不存在: {self.split_dir}")

        # 加载所有序列
        self.sequences = self._load_sequences()

        logger.info(
            f"加载MOT数据集: {root}, split={split}, "
            f"序列数={len(self.sequences)}"
        )

    def _load_sequences(self) -> List[MOTSequence]:
        """
        加载所有序列

        Returns:
            sequences: MOTSequence对象列表
        """
        sequences = []

        # 遍历split目录
        for seq_name in sorted(os.listdir(self.split_dir)):
            seq_path = os.path.join(self.split_dir, seq_name)

            # 检查是否为有效序列目录
            if not os.path.isdir(seq_path):
                continue

            seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
            if not os.path.exists(seqinfo_path):
                logger.warning(f"跳过无效序列: {seq_name} (缺少seqinfo.ini)")
                continue

            try:
                seq = MOTSequence(seq_path)
                sequences.append(seq)
            except Exception as e:
                logger.error(f"加载序列失败: {seq_name}, 错误: {e}")

        return sequences

    def __len__(self) -> int:
        """返回序列数量"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> MOTSequence:
        """获取指定序列"""
        return self.sequences[idx]

    def __iter__(self):
        """迭代器:遍历所有序列"""
        return iter(self.sequences)

    def get_sequence_by_name(self, name: str) -> Optional[MOTSequence]:
        """
        根据名称获取序列

        Args:
            name: 序列名称

        Returns:
            sequence: MOTSequence对象,不存在返回None
        """
        for seq in self.sequences:
            if seq.name == name:
                return seq
        return None

    def get_summary(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            summary: 统计信息字典
        """
        total_frames = sum(len(seq) for seq in self.sequences)
        total_annotations = 0

        for seq in self.sequences:
            if seq.gt_data is not None:
                for frame_data in seq.gt_data.values():
                    total_annotations += len(frame_data)

        return {
            'root': self.root,
            'split': self.split,
            'n_sequences': len(self.sequences),
            'total_frames': total_frames,
            'total_annotations': total_annotations,
            'sequences': [seq.name for seq in self.sequences]
        }


def visualize_annotations(
        img: np.ndarray,
        annotations: np.ndarray,
        show_id: bool = True
) -> np.ndarray:
    """
    可视化真值标注

    Args:
        img: 原始图像
        annotations: 标注数组 [[id, left, top, w, h, conf, class, vis], ...]
        show_id: 是否显示ID

    Returns:
        vis_img: 可视化后的图像
    """
    vis_img = img.copy()

    if annotations is None or len(annotations) == 0:
        return vis_img

    for ann in annotations:
        track_id, left, top, width, height = map(int, ann[:5])

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(
            vis_img,
            (left, top),
            (left + width, top + height),
            color, 2
        )

        # 显示ID
        if show_id:
            label = f"ID:{track_id}"
            cv2.putText(
                vis_img, label,
                (left, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )

    return vis_img


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("测试MOT数据集加载")
    print("=" * 50)

    # 注意:需要先下载MOT17数据集
    dataset_root = "/path/to/MOT17"  # 修改为实际路径

    # 检查路径是否存在
    if not os.path.exists(dataset_root):
        print(f"\n数据集路径不存在: {dataset_root}")
        print("请下载MOT17数据集并修改dataset_root路径")
        print("\n创建模拟数据集结构进行测试...")

        # 创建模拟结构
        mock_root = "/tmp/MOT17_mock"
        mock_seq = os.path.join(mock_root, "train", "MOT17-02-DPM")
        os.makedirs(os.path.join(mock_seq, "img1"), exist_ok=True)
        os.makedirs(os.path.join(mock_seq, "gt"), exist_ok=True)
        os.makedirs(os.path.join(mock_seq, "det"), exist_ok=True)

        # 创建seqinfo.ini
        seqinfo = """[Sequence]
name=MOT17-02-DPM
imDir=img1
frameRate=30
seqLength=10
imWidth=1920
imHeight=1080
imExt=.jpg
"""
        with open(os.path.join(mock_seq, "seqinfo.ini"), 'w') as f:
            f.write(seqinfo)

        # 创建模拟图像
        for i in range(1, 11):
            img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(
                os.path.join(mock_seq, "img1", f"{i:06d}.jpg"),
                img
            )

        # 创建模拟标注
        gt_data = []
        for frame in range(1, 11):
            for obj_id in range(1, 4):
                x = 100 + obj_id * 200 + frame * 10
                y = 200 + obj_id * 50
                gt_data.append([frame, obj_id, x, y, 50, 100, 1, 1, 1])

        np.savetxt(
            os.path.join(mock_seq, "gt", "gt.txt"),
            gt_data, delimiter=',', fmt='%d,%d,%d,%d,%d,%d,%d,%d,%d'
        )

        dataset_root = mock_root
        print(f"模拟数据集创建完成: {dataset_root}")

    # 加载数据集
    print(f"\n加载数据集: {dataset_root}")
    dataset = MOTDataset(dataset_root, split='train')

    # 显示数据集信息
    print("\n" + "=" * 50)
    print("数据集统计")
    print("=" * 50)

    summary = dataset.get_summary()
    for key, value in summary.items():
        if key != 'sequences':
            print(f"{key}: {value}")

    # 测试第一个序列
    if len(dataset) > 0:
        print("\n" + "=" * 50)
        print("测试第一个序列")
        print("=" * 50)

        seq = dataset[0]
        print(f"\n序列名称: {seq.name}")
        print(f"序列长度: {len(seq)}帧")
        print(f"图像尺寸: {seq.img_width}x{seq.img_height}")
        print(f"帧率: {seq.frame_rate}fps")

        # 测试读取第一帧
        print("\n读取第1帧...")
        frame_id = 1
        img = seq.get_frame(frame_id)
        gt = seq.get_gt(frame_id)

        if img is not None:
            print(f"图像形状: {img.shape}")

        if gt is not None:
            print(f"真值标注数: {len(gt)}")
            print(f"标注示例:\n{gt[:3]}")

            # 可视化
            if img is not None:
                vis_img = visualize_annotations(img, gt)
                print(f"可视化图像形状: {vis_img.shape}")

        # 测试迭代器
        print("\n测试迭代器(前3帧)...")
        for i, (frame_id, img, gt, dets) in enumerate(seq):
            if i >= 3:
                break
            print(f"帧{frame_id}: img={img.shape if img is not None else None}, "
                  f"gt={len(gt) if gt is not None else 0}, "
                  f"dets={len(dets) if dets is not None else 0}")

    print("\n测试完成!")