"""
BM25检索器
"""
import jieba
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self, bm25_index):
        """
        Args:
            bm25_index: BM25Index实例
        """
        self.bm25 = bm25_index.bm25
        self.chunks = bm25_index.chunks
        self.tokenized_chunks = bm25_index.tokenized_chunks
    
    def retrieve(self, query, top_k=5):
        """
        执行BM25检索
        
        Args:
            query: 查询字符串
            top_k: 返回top-k结果
            
        Returns:
            indices: List[int], top-k文档索引
        """
        # 对查询进行分词
        query_tokens = list(jieba.cut(query))
        
        # 计算BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top-k索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return top_indices.tolist()
    
    def retrieve_with_scores(self, query, top_k=5):
        """返回带分数的检索结果"""
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return top_indices.tolist(), scores[top_indices].tolist()