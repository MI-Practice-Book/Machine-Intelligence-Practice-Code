"""
YOLOv5训练封装脚本
提供参数化配置、断点续训、学习率调度和训练日志管理功能

Author: Machine Intelligence Practice
Date: 2025-01
"""

import logging
import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import yaml
import json

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger, TaskTimer, log_system_info


class YOLOv5Trainer:
    """YOLOv5训练器封装类"""

    def __init__(
            self,
            yolov5_root: Path,
            data_yaml: Path,
            project: Path,
            name: str
    ):
        """
        初始化训练器

        Args:
            yolov5_root: YOLOv5仓库根目录
            data_yaml: 数据集配置文件路径
            project: 训练输出项目目录
            name: 本次训练的名称
        """
        self.yolov5_root = Path(yolov5_root)
        self.data_yaml = Path(data_yaml)
        self.project = Path(project)
        self.name = name

        # 验证路径
        if not self.yolov5_root.exists():
            raise FileNotFoundError(f"YOLOv5目录不存在: {self.yolov5_root}")

        if not self.data_yaml.exists():
            raise FileNotFoundError(f"数据配置文件不存在: {self.data_yaml}")

        self.train_script = self.yolov5_root / "train.py"
        if not self.train_script.exists():
            raise FileNotFoundError(f"训练脚本不存在: {self.train_script}")

        # 训练输出目录
        self.output_dir = self.project / self.name

    def build_command(
            self,
            img_size: int = 640,
            batch_size: int = 16,
            epochs: int = 100,
            weights: str = 'yolov5s.pt',
            device: str = '0',
            workers: int = 8,
            optimizer: str = 'SGD',
            lr0: float = 0.01,
            lrf: float = 0.1,
            momentum: float = 0.937,
            weight_decay: float = 0.0005,
            warmup_epochs: int = 3,
            warmup_momentum: float = 0.8,
            warmup_bias_lr: float = 0.1,
            resume: bool = False,
            cache: str = None,
            multi_scale: bool = False,
            single_cls: bool = False,
            image_weights: bool = False,
            rect: bool = False,
            cos_lr: bool = False,
            label_smoothing: float = 0.0,
            patience: int = 100,
            freeze: list = None,
            save_period: int = -1,
            **kwargs
    ) -> list:
        """
        构建训练命令

        Args:
            img_size: 输入图像尺寸
            batch_size: 批次大小
            epochs: 训练轮数
            weights: 预训练权重路径
            device: GPU设备ID(多GPU用逗号分隔,如'0,1')
            workers: 数据加载线程数
            optimizer: 优化器类型(SGD/Adam/AdamW)
            lr0: 初始学习率
            lrf: 最终学习率(相对于lr0的比例)
            momentum: SGD动量
            weight_decay: 权重衰减
            warmup_epochs: 预热轮数
            warmup_momentum: 预热阶段动量
            warmup_bias_lr: 预热阶段偏置学习率
            resume: 是否从上次中断处继续训练
            cache: 缓存图像(ram/disk)
            multi_scale: 是否使用多尺度训练
            single_cls: 是否单类别训练
            image_weights: 是否使用图像加权采样
            rect: 是否使用矩形训练
            cos_lr: 是否使用余弦学习率调度
            label_smoothing: 标签平滑系数
            patience: 早停耐心值
            freeze: 冻结层列表
            save_period: 保存间隔(轮数)
            **kwargs: 其他参数

        Returns:
            命令列表
        """
        cmd = [
            sys.executable,
            str(self.train_script),
            '--data', str(self.data_yaml),
            '--weights', weights,
            '--cfg', '',  # 使用weights中的配置
            '--img', str(img_size),
            '--batch', str(batch_size),
            '--epochs', str(epochs),
            '--device', device,
            '--workers', str(workers),
            '--project', str(self.project),
            '--name', self.name,
            '--optimizer', optimizer,
            '--lr0', str(lr0),
            '--lrf', str(lrf),
            '--momentum', str(momentum),
            '--weight-decay', str(weight_decay),
            '--warmup-epochs', str(warmup_epochs),
            '--warmup-momentum', str(warmup_momentum),
            '--warmup-bias-lr', str(warmup_bias_lr),
            '--label-smoothing', str(label_smoothing),
            '--patience', str(patience),
            '--save-period', str(save_period),
        ]

        # 布尔参数
        if resume:
            cmd.append('--resume')
        if multi_scale:
            cmd.append('--multi-scale')
        if single_cls:
            cmd.append('--single-cls')
        if image_weights:
            cmd.append('--image-weights')
        if rect:
            cmd.append('--rect')
        if cos_lr:
            cmd.append('--cos-lr')

        # 可选参数
        if cache:
            cmd.extend(['--cache', cache])

        if freeze:
            cmd.extend(['--freeze', *[str(x) for x in freeze]])

        # 其他自定义参数
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        return cmd

    def train(self, **kwargs) -> bool:
        """
        执行训练

        Args:
            **kwargs: 训练参数(传递给build_command)

        Returns:
            训练是否成功
        """
        with TaskTimer(f"训练模型: {self.name}"):
            # 构建命令
            cmd = self.build_command(**kwargs)

            # 记录训练配置
            logging.info("=" * 70)
            logging.info("训练配置:")
            logging.info(f"  数据配置: {self.data_yaml}")
            logging.info(f"  输出目录: {self.output_dir}")
            logging.info(f"  图像尺寸: {kwargs.get('img_size', 640)}")
            logging.info(f"  批次大小: {kwargs.get('batch_size', 16)}")
            logging.info(f"  训练轮数: {kwargs.get('epochs', 100)}")
            logging.info(f"  初始学习率: {kwargs.get('lr0', 0.01)}")
            logging.info(f"  优化器: {kwargs.get('optimizer', 'SGD')}")
            logging.info(f"  设备: {kwargs.get('device', '0')}")
            logging.info("=" * 70)

            # 记录完整命令
            logging.info("\n执行命令:")
            logging.info(" ".join(cmd))
            logging.info("")

            try:
                # 执行训练
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
                    logging.info("✅ 训练完成!")
                    logging.info(f"输出目录: {self.output_dir}")
                    logging.info(f"最佳权重: {self.output_dir / 'weights' / 'best.pt'}")
                    logging.info(f"最终权重: {self.output_dir / 'weights' / 'last.pt'}")
                    logging.info("=" * 70)
                    return True
                else:
                    logging.error(f"❌ 训练失败! 返回码: {return_code}")
                    return False

            except Exception as e:
                logging.error(f"❌ 训练过程出错: {e}")
                return False

    def save_config(self, config: dict):
        """
        保存训练配置

        Args:
            config: 配置字典
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_file = self.output_dir / "train_config.json"

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logging.info(f"训练配置已保存: {config_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv5训练封装脚本")

    # 必需参数
    parser.add_argument(
        '--yolov5_root',
        type=str,
        required=True,
        help='YOLOv5仓库根目录'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='数据集配置文件(yaml)'
    )

    # 基本参数
    parser.add_argument('--project', type=str, default='runs/train', help='输出项目目录')
    parser.add_argument('--name', type=str, default='exp', help='本次训练名称')
    parser.add_argument('--img', '--img-size', type=int, default=640, dest='img_size', help='输入图像尺寸')
    parser.add_argument('--batch', '--batch-size', type=int, default=16, dest='batch_size', help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='预训练权重')
    parser.add_argument('--device', type=str, default='0', help='GPU设备ID')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')

    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='优化器')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.1, help='最终学习率比例')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')

    # 训练策略
    parser.add_argument('--warmup-epochs', type=int, default=3, help='预热轮数')
    parser.add_argument('--cos-lr', action='store_true', help='使用余弦学习率')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='标签平滑')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值')
    parser.add_argument('--multi-scale', action='store_true', help='多尺度训练')
    parser.add_argument('--cache', type=str, default=None, choices=['ram', 'disk'], help='缓存图像')

    # 其他参数
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续')
    parser.add_argument('--save-period', type=int, default=-1, help='保存间隔')
    parser.add_argument('--log-file', type=str, default=None, help='日志文件路径')

    args = parser.parse_args()

    # 设置日志
    log_file = args.log_file
    if log_file is None:
        log_dir = Path(args.project) / args.name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{timestamp}.log"

    setup_logger(log_file=str(log_file))

    logging.info("=" * 70)
    logging.info("YOLOv5训练工具")
    logging.info("=" * 70)
    logging.info(f"训练名称: {args.name}")
    logging.info(f"日志文件: {log_file}")
    logging.info("=" * 70 + "\n")

    # 记录系统信息
    log_system_info()

    # 创建训练器
    trainer = YOLOv5Trainer(
        yolov5_root=args.yolov5_root,
        data_yaml=args.data,
        project=args.project,
        name=args.name
    )

    # 准备训练参数
    train_config = {
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weights': args.weights,
        'device': args.device,
        'workers': args.workers,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'cos_lr': args.cos_lr,
        'label_smoothing': args.label_smoothing,
        'patience': args.patience,
        'multi_scale': args.multi_scale,
        'cache': args.cache,
        'resume': args.resume,
        'save_period': args.save_period,
    }

    # 保存配置
    trainer.save_config(train_config)

    # 开始训练
    success = trainer.train(**train_config)

    if success:
        logging.info("✅ 训练成功完成!")
        return 0
    else:
        logging.error("❌ 训练失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())