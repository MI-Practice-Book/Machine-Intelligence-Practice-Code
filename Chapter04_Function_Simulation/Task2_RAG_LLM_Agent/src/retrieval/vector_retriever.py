"""
向量检索器
"""
import logging

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, vector_store, embedding_model):
        """
        Args:
            vector_store: VectorStore实例
            embedding_model: SentenceTransformer模型
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query, top_k=5):
        """
        执行向量检索
        
        Args:
            query: 查询字符串
            top_k: 返回top-k结果
            
        Returns:
            indices: List[int], top-k文档索引
        """
        # 对查询进行向量化
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )
        
        # 执行相似度搜索
        indices, similarities = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
        
        return indices.tolist()
    
    def retrieve_with_scores(self, query, top_k=5):
        """返回带分数的检索结果"""
        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )
        
        indices, similarities = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
        
        return indices.tolist(), similarities.tolist()