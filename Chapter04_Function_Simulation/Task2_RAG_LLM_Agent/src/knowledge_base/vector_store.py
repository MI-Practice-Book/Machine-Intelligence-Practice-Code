"""
向量存储模块
"""
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储：管理文本块的向量表示"""
    
    def __init__(self, embeddings, chunks, metadata):
        """
        Args:
            embeddings: numpy array, shape (n_chunks, embedding_dim)
            chunks: List[str], 文本块列表
            metadata: List[Dict], 元数据列表
        """
        self.embeddings = np.array(embeddings)
        self.chunks = chunks
        self.metadata = metadata
        
        logger.info(f"向量存储初始化完成: {len(chunks)} 个文本块, 向量维度 {self.embeddings.shape[1]}")
    
    def search(self, query_embedding, top_k=5):
        """
        向量相似度搜索
        
        Args:
            query_embedding: numpy array, shape (embedding_dim,)
            top_k: 返回top-k个结果
            
        Returns:
            indices: numpy array, top-k索引
            similarities: numpy array, 对应的相似度分数
        """
        # 计算余弦相似度
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def save(self, filepath):
        """保存向量存储到文件"""
        np.savez(
            filepath,
            embeddings=self.embeddings,
            chunks=self.chunks,
            metadata=self.metadata
        )
        logger.info(f"向量存储已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """从文件加载向量存储"""
        data = np.load(filepath, allow_pickle=True)
        logger.info(f"从 {filepath} 加载向量存储")
        return cls(
            embeddings=data['embeddings'],
            chunks=data['chunks'].tolist(),
            metadata=data['metadata'].tolist()
        )
    
    @classmethod
    def build(cls, chunks, embedding_model, metadata, batch_size=32):
        """
        构建向量存储
        
        Args:
            chunks: List[str], 文本块
            embedding_model: SentenceTransformer模型
            metadata: List[Dict], 元数据
            batch_size: 批处理大小
            
        Returns:
            VectorStore实例
        """
        logger.info(f"开始向量化 {len(chunks)} 个文本块...")
        
        embeddings = embedding_model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # 归一化
        )
        
        logger.info(f"向量化完成，向量维度: {embeddings.shape}")
        
        return cls(embeddings, chunks, metadata)