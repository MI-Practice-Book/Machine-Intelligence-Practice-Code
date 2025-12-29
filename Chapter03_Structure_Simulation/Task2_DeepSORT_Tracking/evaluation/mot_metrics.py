"""
MOT评估指标计算模块
实现MOT Challenge标准评估指标:
- MOTA (Multiple Object Tracking Accuracy): 多目标跟踪准确度
- MOTP (Multiple Object Tracking Precision): 多目标跟踪精度
- IDF1: ID F1分数
- MT/ML/FP/FN/IDS等统计指标
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import os

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入py-motmetrics库
try:
    import motmetrics as mm

    MOTMETRICS_AVAILABLE = True
    logger.info("motmetrics库已加载,使用官方实现")
except ImportError:
    MOTMETRICS_AVAILABLE = False
    logger.warning("未安装motmetrics库,使用简化实现")


class MOTEvaluator:
    """
    MOT评估器

    支持两种模式:
    1. 使用py-motmetrics官方库(推荐)
    2. 简化自实现版本
    """

    def __init__(self):
        """初始化评估器"""
        self.use_motmetrics = MOTMETRICS_AVAILABLE

        if self.use_motmetrics:
            self.accumulator = mm.MOTAccumulator(auto_id=True)
        else:
            self.reset()

    def reset(self):
        """重置累加器"""
        if self.use_motmetrics:
            self.accumulator = mm.MOTAccumulator(auto_id=True)
        else:
            self.frame_data = []
            logger.debug("评估器已重置")

    def update(
            self,
            frame_id: int,
            gt_boxes: np.ndarray,
            pred_boxes: np.ndarray,
            iou_threshold: float = 0.5
    ):
        """
        更新单帧评估数据

        Args:
            frame_id: 帧ID
            gt_boxes: 真值框 [[id, left, top, w, h, ...], ...]
            pred_boxes: 预测框 [[id, left, top, w, h, ...], ...]
            iou_threshold: IoU匹配阈值
        """
        if self.use_motmetrics:
            self._update_motmetrics(frame_id, gt_boxes, pred_boxes, iou_threshold)
        else:
            self._update_simple(frame_id, gt_boxes, pred_boxes, iou_threshold)

    def _update_motmetrics(
            self,
            frame_id: int,
            gt_boxes: np.ndarray,
            pred_boxes: np.ndarray,
            iou_threshold: float
    ):
        """使用motmetrics库更新"""
        # 提取ID和边界框
        gt_ids = gt_boxes[:, 0].astype(int) if len(gt_boxes) > 0 else []
        pred_ids = pred_boxes[:, 0].astype(int) if len(pred_boxes) > 0 else []

        gt_tlwh = gt_boxes[:, 1:5] if len(gt_boxes) > 0 else np.array([])
        pred_tlwh = pred_boxes[:, 1:5] if len(pred_boxes) > 0 else np.array([])

        # 计算距离矩阵(1 - IoU)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            distances = compute_iou_distance_matrix(gt_tlwh, pred_tlwh)
        else:
            distances = np.empty((len(gt_ids), len(pred_ids)))

        # 更新累加器
        self.accumulator.update(
            gt_ids, pred_ids, distances,
            frameid=frame_id
        )

    def _update_simple(
            self,
            frame_id: int,
            gt_boxes: np.ndarray,
            pred_boxes: np.ndarray,
            iou_threshold: float
    ):
        """简化版本更新"""
        self.frame_data.append({
            'frame_id': frame_id,
            'gt_boxes': gt_boxes.copy() if len(gt_boxes) > 0 else np.array([]),
            'pred_boxes': pred_boxes.copy() if len(pred_boxes) > 0 else np.array([]),
            'iou_threshold': iou_threshold
        })

    def compute_metrics(self) -> Dict:
        """
        计算所有评估指标

        Returns:
            metrics: 指标字典
        """
        if self.use_motmetrics:
            return self._compute_motmetrics()
        else:
            return self._compute_simple()

    def _compute_motmetrics(self) -> Dict:
        """使用motmetrics库计算指标"""
        # 创建metrics host
        mh = mm.metrics.create()

        # 计算所有指标
        summary = mh.compute(
            self.accumulator,
            metrics=[
                'num_frames', 'mota', 'motp', 'idf1',
                'num_switches', 'num_false_positives',
                'num_misses', 'num_detections',
                'num_objects', 'num_predictions',
                'mostly_tracked', 'partially_tracked',
                'mostly_lost', 'precision', 'recall'
            ],
            name='acc'
        )

        # 转换为字典
        metrics = {
            'MOTA': summary['mota'][0] * 100,  # 转换为百分比
            'MOTP': summary['motp'][0] * 100,
            'IDF1': summary['idf1'][0] * 100,
            'IDS': int(summary['num_switches'][0]),
            'FP': int(summary['num_false_positives'][0]),
            'FN': int(summary['num_misses'][0]),
            'Precision': summary['precision'][0] * 100,
            'Recall': summary['recall'][0] * 100,
            'MT': int(summary['mostly_tracked'][0]),
            'PT': int(summary['partially_tracked'][0]),
            'ML': int(summary['mostly_lost'][0]),
            'num_frames': int(summary['num_frames'][0]),
            'num_objects': int(summary['num_objects'][0]),
            'num_predictions': int(summary['num_predictions'][0])
        }

        return metrics

    def _compute_simple(self) -> Dict:
        """简化版本计算指标"""
        logger.info("使用简化版本计算MOT指标...")

        total_gt = 0
        total_pred = 0
        total_matched = 0
        total_fp = 0
        total_fn = 0
        total_iou = 0.0
        matched_count = 0

        # 统计ID切换
        prev_matches = {}  # {gt_id: pred_id}
        id_switches = 0

        for frame_data in self.frame_data:
            gt_boxes = frame_data['gt_boxes']
            pred_boxes = frame_data['pred_boxes']
            iou_threshold = frame_data['iou_threshold']

            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)

            total_gt += n_gt
            total_pred += n_pred

            if n_gt == 0 or n_pred == 0:
                total_fp += n_pred
                total_fn += n_gt
                continue

            # 计算IoU矩阵
            gt_tlwh = gt_boxes[:, 1:5]
            pred_tlwh = pred_boxes[:, 1:5]
            iou_matrix = compute_iou_matrix(gt_tlwh, pred_tlwh)

            # 贪心匹配(简化版)
            matched_gt = set()
            matched_pred = set()
            curr_matches = {}

            # 按IoU从大到小匹配
            while True:
                max_iou = 0
                max_pos = None

                for i in range(n_gt):
                    if i in matched_gt:
                        continue
                    for j in range(n_pred):
                        if j in matched_pred:
                            continue
                        if iou_matrix[i, j] > max_iou and iou_matrix[i, j] >= iou_threshold:
                            max_iou = iou_matrix[i, j]
                            max_pos = (i, j)

                if max_pos is None:
                    break

                i, j = max_pos
                matched_gt.add(i)
                matched_pred.add(j)
                total_matched += 1
                total_iou += max_iou
                matched_count += 1

                gt_id = int(gt_boxes[i, 0])
                pred_id = int(pred_boxes[j, 0])
                curr_matches[gt_id] = pred_id

                # 检测ID切换
                if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                    id_switches += 1

            prev_matches = curr_matches

            # 统计FP和FN
            total_fp += (n_pred - len(matched_pred))
            total_fn += (n_gt - len(matched_gt))

        # 计算指标
        mota = 1.0 - (total_fp + total_fn + id_switches) / max(total_gt, 1)
        motp = total_iou / max(matched_count, 1)
        precision = total_matched / max(total_pred, 1)
        recall = total_matched / max(total_gt, 1)

        metrics = {
            'MOTA': mota * 100,
            'MOTP': motp * 100,
            'IDF1': -1,  # 简化版本不计算
            'IDS': id_switches,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'MT': -1,  # 简化版本不计算
            'PT': -1,
            'ML': -1,
            'num_frames': len(self.frame_data),
            'num_objects': total_gt,
            'num_predictions': total_pred
        }

        return metrics

    def print_metrics(self, metrics: Dict):
        """
        打印评估指标

        Args:
            metrics: 指标字典
        """
        logger.info("=" * 50)
        logger.info("MOT评估结果")
        logger.info("=" * 50)

        logger.info(f"MOTA: {metrics['MOTA']:.2f}%")
        logger.info(f"MOTP: {metrics['MOTP']:.2f}%")
        if metrics['IDF1'] >= 0:
            logger.info(f"IDF1: {metrics['IDF1']:.2f}%")
        logger.info(f"Precision: {metrics['Precision']:.2f}%")
        logger.info(f"Recall: {metrics['Recall']:.2f}%")

        logger.info("-" * 50)
        logger.info(f"ID Switches: {metrics['IDS']}")
        logger.info(f"False Positives: {metrics['FP']}")
        logger.info(f"False Negatives: {metrics['FN']}")

        if metrics['MT'] >= 0:
            logger.info(f"Mostly Tracked: {metrics['MT']}")
            logger.info(f"Partially Tracked: {metrics['PT']}")
            logger.info(f"Mostly Lost: {metrics['ML']}")

        logger.info("-" * 50)
        logger.info(f"Total Frames: {metrics['num_frames']}")
        logger.info(f"Total Objects: {metrics['num_objects']}")
        logger.info(f"Total Predictions: {metrics['num_predictions']}")
        logger.info("=" * 50)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    计算两个框的IoU

    Args:
        box1: [left, top, width, height]
        box2: [left, top, width, height]

    Returns:
        iou: IoU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-6)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算IoU矩阵

    Args:
        boxes1: (N, 4) [[left, top, width, height], ...]
        boxes2: (M, 4) [[left, top, width, height], ...]

    Returns:
        iou_matrix: (N, M)
    """
    n = len(boxes1)
    m = len(boxes2)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])

    return iou_matrix


def compute_iou_distance_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    计算IoU距离矩阵 (1 - IoU)

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        distance_matrix: (N, M)
    """
    iou_matrix = compute_iou_matrix(boxes1, boxes2)
    return 1.0 - iou_matrix


def load_mot_results(file_path: str) -> Dict[int, np.ndarray]:
    """
    加载MOT结果文件

    格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, ...

    Args:
        file_path: 文件路径

    Returns:
        results: {frame_id: array([[id, left, top, w, h, conf], ...])}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"结果文件不存在: {file_path}")

    data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)

    results = defaultdict(list)
    for row in data:
        frame_id = int(row[0])
        track_id = int(row[1])
        bbox = row[2:6]  # [left, top, width, height]
        conf = row[6] if len(row) > 6 else 1.0

        results[frame_id].append([track_id, *bbox, conf])

    # 转换为numpy数组
    for frame_id in results:
        results[frame_id] = np.array(results[frame_id])

    logger.info(f"加载结果: {len(results)}帧")
    return dict(results)


def evaluate_mot(
        gt_file: str,
        pred_file: str,
        iou_threshold: float = 0.5
) -> Dict:
    """
    评估MOT跟踪结果

    Args:
        gt_file: 真值文件路径
        pred_file: 预测结果文件路径
        iou_threshold: IoU匹配阈值

    Returns:
        metrics: 评估指标
    """
    logger.info("=" * 50)
    logger.info("开始MOT评估")
    logger.info("=" * 50)
    logger.info(f"真值文件: {gt_file}")
    logger.info(f"预测文件: {pred_file}")
    logger.info(f"IoU阈值: {iou_threshold}")

    # 加载数据
    gt_data = load_mot_results(gt_file)
    pred_data = load_mot_results(pred_file)

    # 创建评估器
    evaluator = MOTEvaluator()

    # 获取所有帧
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))

    # 逐帧评估
    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, np.array([]))
        pred_boxes = pred_data.get(frame_id, np.array([]))

        evaluator.update(frame_id, gt_boxes, pred_boxes, iou_threshold)

    # 计算指标
    metrics = evaluator.compute_metrics()

    # 打印结果
    evaluator.print_metrics(metrics)

    return metrics


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("测试MOT评估指标")
    print("=" * 50)

    # 创建模拟数据
    evaluator = MOTEvaluator()

    # 模拟10帧数据
    for frame_id in range(1, 11):
        # 真值: 3个目标
        gt_boxes = np.array([
            [1, 100 + frame_id * 5, 200, 50, 100],
            [2, 250 + frame_id * 3, 150, 55, 110],
            [3, 400 + frame_id * 4, 200, 50, 105]
        ])

        # 预测: 3个目标(略有偏差)
        pred_boxes = np.array([
            [1, 105 + frame_id * 5, 205, 48, 98],  # 匹配ID1
            [2, 255 + frame_id * 3, 155, 53, 108],  # 匹配ID2
            [3, 405 + frame_id * 4, 205, 52, 103]  # 匹配ID3
        ])

        evaluator.update(frame_id, gt_boxes, pred_boxes, iou_threshold=0.5)

    # 计算指标
    print("\n计算评估指标...")
    metrics = evaluator.compute_metrics()

    # 打印结果
    evaluator.print_metrics(metrics)

    print("\n测试完成!")