"""
文本加载模块
"""
import logging

logger = logging.getLogger(__name__)


class TextLoader:
    """文本加载器，自动检测文件编码"""
    
    def __init__(self):
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    
    def load(self, filepath):
        """
        加载文本文件，自动检测编码
        
        Args:
            filepath: 文件路径
            
        Returns:
            text: 文本内容
        """
        for encoding in self.supported_encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    text = f.read()
                logger.info(f"成功使用 {encoding} 编码加载文件: {filepath}")
                return text
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"无法解码文件 {filepath}，请检查文件编码")