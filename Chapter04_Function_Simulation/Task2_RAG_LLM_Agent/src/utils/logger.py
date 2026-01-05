"""
日志配置模块
"""
import logging
import os
from datetime import datetime


def setup_logger(log_dir='logs', log_level='INFO'):
    """
    配置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件名
    log_filename = os.path.join(
        log_dir,
        f'xiyouji_qa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化，日志文件: {log_filename}")