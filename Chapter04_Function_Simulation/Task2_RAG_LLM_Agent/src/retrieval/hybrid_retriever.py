"""
混合检索器：整合BM25和向量检索
"""
import logging
from .fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, bm25_retriever, vector_retriever, chunks):
        """
        Args:
            bm25_retriever: BM25Retriever实例
            vector_retriever: VectorRetriever实例
            chunks: List[str], 文本块列表
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.chunks = chunks
    
    def retrieve(self, query, top_k=3, retrieve_k=10, fusion_k=60):
        """
        执行混合检索
        
        Args:
            query: 查询字符串
            top_k: 最终返回结果数
            retrieve_k: 每个检索器的初始候选数
            fusion_k: RRF参数
            
        Returns:
            contexts: List[str], top-k文本块
            indices: List[int], 对应的索引
        """
        # 执行双重检索
        bm25_indices = self.bm25_retriever.retrieve(query, top_k=retrieve_k)
        vector_indices = self.vector_retriever.retrieve(query, top_k=retrieve_k)
        
        # RRF融合
        fused_indices = reciprocal_rank_fusion(
            [bm25_indices, vector_indices],
            k=fusion_k
        )
        
        # 返回top-k
        final_indices = fused_indices[:top_k]
        contexts = [self.chunks[i] for i in final_indices]
        
        return contexts, final_indices
    
    def retrieve_with_scores(self, query, top_k=3, retrieve_k=10):
        """返回带详细分数的检索结果"""
        # 获取带分数的检索结果
        bm25_indices, bm25_scores = self.bm25_retriever.retrieve_with_scores(
            query, top_k=retrieve_k
        )
        vector_indices, vector_scores = self.vector_retriever.retrieve_with_scores(
            query, top_k=retrieve_k
        )
        
        # 构建分数字典
        bm25_score_dict = dict(zip(bm25_indices, bm25_scores))
        vector_score_dict = dict(zip(vector_indices, vector_scores))
        
        # RRF融合
        fused_indices = reciprocal_rank_fusion(
            [bm25_indices, vector_indices],
            k=60
        )[:top_k]
        
        # 组装详细结果
        results = []
        for idx in fused_indices:
            results.append({
                'index': idx,
                'text': self.chunks[idx],
                'bm25_score': bm25_score_dict.get(idx, 0.0),
                'vector_score': vector_score_dict.get(idx, 0.0)
            })
        
        return results