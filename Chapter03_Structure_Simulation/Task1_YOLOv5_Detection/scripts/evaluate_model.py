"""
YOLOv5模型评估脚本
提供mAP计算、混淆矩阵生成、PR曲线绘制和详细评估报告

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

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger, TaskTimer, log_system_info


class YOLOv5Evaluator:
    """YOLOv5模型评估器"""

    def __init__(
            self,
            yolov5_root: Path,
            weights: Path,
            data_yaml: Path,
            project: Path,
            name: str
    ):
        """
        初始化评估器

        Args:
            yolov5_root: YOLOv5仓库根目录
            weights: 模型权重文件路径
            data_yaml: 数据集配置文件路径
            project: 评估输出项目目录
            name: 本次评估的名称
        """
        self.yolov5_root = Path(yolov5_root)
        self.weights = Path(weights)
        self.data_yaml = Path(data_yaml)
        self.project = Path(project)
        self.name = name

        # 验证路径
        if not self.yolov5_root.exists():
            raise FileNotFoundError(f"YOLOv5目录不存在: {self.yolov5_root}")

        if not self.weights.exists():
            raise FileNotFoundError(f"模型权重不存在: {self.weights}")

        if not self.data_yaml.exists():
            raise FileNotFoundError(f"数据配置文件不存在: {self.data_yaml}")

        self.val_script = self.yolov5_root / "val.py"
        if not self.val_script.exists():
            raise FileNotFoundError(f"评估脚本不存在: {self.val_script}")

        # 评估输出目录
        self.output_dir = self.project / self.name

    def build_command(
            self,
            img_size: int = 640,
            batch_size: int = 32,
            conf_thres: float = 0.001,
            iou_thres: float = 0.6,
            device: str = '0',
            workers: int = 8,
            task: str = 'val',
            save_txt: bool = True,
            save_conf: bool = True,
            save_json: bool = False,
            plots: bool = True,
            verbose: bool = False,
            **kwargs
    ) -> list:
        """
        构建评估命令

        Args:
            img_size: 输入图像尺寸
            batch_size: 批次大小
            conf_thres: 置信度阈值
            iou_thres: NMS的IoU阈值
            device: GPU设备ID
            workers: 数据加载线程数
            task: 评估任务(val/test/speed)
            save_txt: 是否保存txt格式预测结果
            save_conf: 是否在txt中保存置信度
            save_json: 是否保存COCO格式的json结果
            plots: 是否生成可视化图表
            verbose: 是否显示详细信息
            **kwargs: 其他参数

        Returns:
            命令列表
        """
        cmd = [
            sys.executable,
            str(self.val_script),
            '--data', str(self.data_yaml),
            '--weights', str(self.weights),
            '--img', str(img_size),
            '--batch', str(batch_size),
            '--conf-thres', str(conf_thres),
            '--iou-thres', str(iou_thres),
            '--device', device,
            '--workers', str(workers),
            '--task', task,
            '--project', str(self.project),
            '--name', self.name,
        ]

        # 布尔参数
        if save_txt:
            cmd.append('--save-txt')
        if save_conf:
            cmd.append('--save-conf')
        if save_json:
            cmd.append('--save-json')
        if plots:
            cmd.append('--plots')
        if verbose:
            cmd.append('--verbose')

        # 其他自定义参数
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        return cmd

    def evaluate(self, **kwargs) -> dict:
        """
        执行模型评估

        Args:
            **kwargs: 评估参数(传递给build_command)

        Returns:
            评估结果字典
        """
        with TaskTimer(f"评估模型: {self.weights.name}"):
            # 构建命令
            cmd = self.build_command(**kwargs)

            # 记录评估配置
            logging.info("=" * 70)
            logging.info("评估配置:")
            logging.info(f"  模型权重: {self.weights}")
            logging.info(f"  数据配置: {self.data_yaml}")
            logging.info(f"  输出目录: {self.output_dir}")
            logging.info(f"  图像尺寸: {kwargs.get('img_size', 640)}")
            logging.info(f"  批次大小: {kwargs.get('batch_size', 32)}")
            logging.info(f"  置信度阈值: {kwargs.get('conf_thres', 0.001)}")
            logging.info(f"  IoU阈值: {kwargs.get('iou_thres', 0.6)}")
            logging.info(f"  设备: {kwargs.get('device', '0')}")
            logging.info("=" * 70)

            # 记录完整命令
            logging.info("\n执行命令:")
            logging.info(" ".join(cmd))
            logging.info("")

            results = {}

            try:
                # 执行评估
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                # 实时输出日志并解析结果
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        logging.info(line)

                        # 解析关键指标
                        if 'all' in line and 'mAP' in line:
                            # 尝试解析mAP结果行
                            try:
                                parts = line.split()
                                if len(parts) >= 6:
                                    results['precision'] = float(parts[4])
                                    results['recall'] = float(parts[5])
                                    results['mAP50'] = float(parts[6])
                                    if len(parts) >= 7:
                                        results['mAP50-95'] = float(parts[7])
                            except:
                                pass

                # 等待进程结束
                return_code = process.wait()

                if return_code == 0:
                    logging.info("=" * 70)
                    logging.info("✅ 评估完成!")

                    # 输出评估结果
                    if results:
                        logging.info("\n评估结果:")
                        if 'precision' in results:
                            logging.info(f"  Precision: {results['precision']:.4f}")
                        if 'recall' in results:
                            logging.info(f"  Recall: {results['recall']:.4f}")
                        if 'mAP50' in results:
                            logging.info(f"  mAP@0.5: {results['mAP50']:.4f}")
                        if 'mAP50-95' in results:
                            logging.info(f"  mAP@0.5:0.95: {results['mAP50-95']:.4f}")

                    logging.info(f"\n输出目录: {self.output_dir}")
                    logging.info("生成文件:")

                    # 列出生成的文件
                    output_files = [
                        'confusion_matrix.png',
                        'F1_curve.png',
                        'P_curve.png',
                        'R_curve.png',
                        'PR_curve.png',
                    ]

                    for file in output_files:
                        file_path = self.output_dir / file
                        if file_path.exists():
                            logging.info(f"  ✓ {file}")

                    logging.info("=" * 70)

                else:
                    logging.error(f"❌ 评估失败! 返回码: {return_code}")

            except Exception as e:
                logging.error(f"❌ 评估过程出错: {e}")

            return results

    def save_results(self, results: dict):
        """
        保存评估结果

        Args:
            results: 评估结果字典
        """
        if not results:
            logging.warning("没有评估结果可保存")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.output_dir / "evaluation_results.json"

        # 添加元数据
        full_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weights': str(self.weights),
            'data_yaml': str(self.data_yaml),
            'metrics': results
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)

        logging.info(f"评估结果已保存: {results_file}")

    def compare_models(self, other_results: dict, other_name: str):
        """
        对比两个模型的评估结果

        Args:
            other_results: 另一个模型的评估结果
            other_name: 另一个模型的名称
        """
        logging.info("=" * 70)
        logging.info(f"模型对比: {self.name} vs {other_name}")
        logging.info("=" * 70)

        metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']

        for metric in metrics:
            if metric in other_results:
                # 这里可以加载当前模型的结果进行对比
                logging.info(f"{metric}: {other_results[metric]:.4f}")

        logging.info("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv5模型评估脚本")

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
        '--data',
        type=str,
        required=True,
        help='数据集配置文件(yaml)'
    )

    # 基本参数
    parser.add_argument('--project', type=str, default='runs/val', help='输出项目目录')
    parser.add_argument('--name', type=str, default='exp', help='本次评估名称')
    parser.add_argument('--img', '--img-size', type=int, default=640, dest='img_size', help='输入图像尺寸')
    parser.add_argument('--batch', '--batch-size', type=int, default=32, dest='batch_size', help='批次大小')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--task', type=str, default='val', choices=['val', 'test', 'speed'], help='评估任务')

    # 输出参数
    parser.add_argument('--save-txt', action='store_true', help='保存txt格式结果')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度')
    parser.add_argument('--save-json', action='store_true', help='保存COCO json格式')
    parser.add_argument('--plots', action='store_true', default=True, help='生成可视化图表')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    # 其他参数
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')

    args = parser.parse_args()

    # 设置日志
    log_file = args.log_file
    if log_file is None:
        log_dir = Path(args.project) / args.name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"eval_{timestamp}.log"

    setup_logger(log_file=str(log_file))

    logging.info("=" * 70)
    logging.info("YOLOv5模型评估工具")
    logging.info("=" * 70)
    logging.info(f"评估名称: {args.name}")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 70 + "\n")

    # 记录系统信息
    log_system_info()

    # 创建评估器
    evaluator = YOLOv5Evaluator(
        yolov5_root=args.yolov5_root,
        weights=args.weights,
        data_yaml=args.data,
        project=args.project,
        name=args.name
    )

    # 准备评估参数
    eval_config = {
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'device': args.device,
        'workers': args.workers,
        'task': args.task,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_json': args.save_json,
        'plots': args.plots,
        'verbose': args.verbose,
    }

    # 开始评估
    results = evaluator.evaluate(**eval_config)

    # 保存结果
    if results:
        evaluator.save_results(results)
        logging.info("✅ 评估成功完成!")
        return 0
    else:
        logging.error("❌ 评估失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())