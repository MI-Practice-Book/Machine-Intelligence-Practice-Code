"""
BM25索引模块
"""
import jieba
import pickle
import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25索引：管理BM25检索索引"""
    
    def __init__(self, bm25, chunks, tokenized_chunks):
        """
        Args:
            bm25: BM25Okapi实例
            chunks: List[str], 原始文本块
            tokenized_chunks: List[List[str]], 分词后的文本块
        """
        self.bm25 = bm25
        self.chunks = chunks
        self.tokenized_chunks = tokenized_chunks
        
        logger.info(f"BM25索引初始化完成: {len(chunks)} 个文档")
    
    def save(self, filepath):
        """保存BM25索引"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks': self.chunks,
                'tokenized_chunks': self.tokenized_chunks
            }, f)
        logger.info(f"BM25索引已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载BM25索引"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"从 {filepath} 加载BM25索引")
        return cls(
            bm25=data['bm25'],
            chunks=data['chunks'],
            tokenized_chunks=data['tokenized_chunks']
        )
    
    @classmethod
    def build(cls, chunks, custom_dict_path=None):
        """
        构建BM25索引
        
        Args:
            chunks: List[str], 文本块
            custom_dict_path: 自定义词典路径（可选）
            
        Returns:
            BM25Index实例
        """
        # 加载自定义词典
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
            logger.info(f"已加载自定义词典: {custom_dict_path}")
        
        # 分词
        logger.info("正在对文本块进行分词...")
        tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]
        
        # 构建BM25索引
        logger.info("构建BM25索引...")
        bm25 = BM25Okapi(tokenized_chunks)
        
        logger.info(f"BM25索引构建完成")
        
        return cls(bm25, chunks, tokenized_chunks)