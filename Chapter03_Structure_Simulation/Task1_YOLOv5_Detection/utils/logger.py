"""
日志管理工具模块
提供统一的日志记录、时间统计和进度显示功能

Author: Machine Intelligence Practice
Date: 2025-01
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime


class TaskTimer:
    """任务计时器,用于统计各阶段执行时间"""

    def __init__(self, task_name: str):
        """
        初始化计时器

        Args:
            task_name: 任务名称
        """
        self.task_name = task_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """上下文管理器入口"""
        self.start_time = time.time()
        logging.info(f"{'=' * 50}")
        logging.info(f"开始任务: {self.task_name}")
        logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"{'=' * 50}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time

        logging.info(f"{'=' * 50}")
        logging.info(f"完成任务: {self.task_name}")
        logging.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"耗时: {self._format_time(elapsed_time)}")
        logging.info(f"{'=' * 50}\n")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        格式化时间显示

        Args:
            seconds: 秒数

        Returns:
            格式化的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}分钟 ({seconds:.2f}秒)"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}小时 ({seconds:.2f}秒)"


class ProgressLogger:
    """进度记录器,用于显示批处理任务的进度"""

    def __init__(self, total: int, task_name: str = "处理"):
        """
        初始化进度记录器

        Args:
            total: 总任务数
            task_name: 任务名称
        """
        self.total = total
        self.task_name = task_name
        self.current = 0
        self.start_time = time.time()

    def update(self, step: int = 1):
        """
        更新进度

        Args:
            step: 增加的步数
        """
        self.current += step
        elapsed_time = time.time() - self.start_time

        # 计算进度百分比
        progress = (self.current / self.total) * 100

        # 估算剩余时间
        if self.current > 0:
            avg_time_per_item = elapsed_time / self.current
            remaining_items = self.total - self.current
            estimated_remaining = avg_time_per_item * remaining_items

            logging.info(
                f"{self.task_name}进度: {self.current}/{self.total} "
                f"({progress:.1f}%) | "
                f"已用时: {TaskTimer._format_time(elapsed_time)} | "
                f"预计剩余: {TaskTimer._format_time(estimated_remaining)}"
            )
        else:
            logging.info(f"{self.task_name}进度: {self.current}/{self.total} ({progress:.1f}%)")

    def finish(self):
        """完成进度记录"""
        elapsed_time = time.time() - self.start_time
        logging.info(
            f"{self.task_name}完成! 总计: {self.total} | "
            f"总耗时: {TaskTimer._format_time(elapsed_time)}"
        )


def setup_logger(
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        console_output: bool = True
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_file: 日志文件路径,如果为None则只输出到控制台
        log_level: 日志级别
        console_output: 是否输出到控制台

    Returns:
        配置好的logger对象
    """
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除已有的handlers
    logger.handlers.clear()

    # 设置日志格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logging.info(f"日志文件: {log_file}")

    return logger


def log_system_info():
    """记录系统信息"""
    import platform
    import torch

    logging.info("=" * 60)
    logging.info("系统信息:")
    logging.info(f"  Python版本: {platform.python_version()}")
    logging.info(f"  操作系统: {platform.system()} {platform.release()}")
    logging.info(f"  PyTorch版本: {torch.__version__}")
    logging.info(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"  CUDA版本: {torch.version.cuda}")
        logging.info(f"  GPU数量: {torch.cuda.device_count()}")
        logging.info(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    # 测试代码
    setup_logger(log_file="test_log.txt")
    log_system_info()

    # 测试TaskTimer
    with TaskTimer("测试任务"):
        time.sleep(2)

    # 测试ProgressLogger
    progress = ProgressLogger(total=10, task_name="测试处理")
    for i in range(10):
        time.sleep(0.5)
        progress.update()
    progress.finish()