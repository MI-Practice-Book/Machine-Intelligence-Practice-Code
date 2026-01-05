"""
知识库构建器：整合所有构建步骤
"""
import json
import logging
from sentence_transformers import SentenceTransformer

from ..data_processing import TextLoader, TextPreprocessor, TextSplitter
from .vector_store import VectorStore
from .bm25_index import BM25Index

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化组件
        self.text_loader = TextLoader()
        self.preprocessor = TextPreprocessor()
        self.text_splitter = TextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        
        # 加载嵌入模型
        logger.info(f"加载嵌入模型: {config['embedding_model_name']}")
        self.embedding_model = SentenceTransformer(config['embedding_model_name'])
    
    def build(self, text_path, custom_dict_path=None, save_indexes=True):
        """
        构建完整的知识库
        
        Args:
            text_path: 原始文本路径
            custom_dict_path: 自定义词典路径
            save_indexes: 是否保存索引
            
        Returns:
            knowledge_base: dict包含所有组件
        """
        logger.info("="*60)
        logger.info("开始构建知识库")
        logger.info("="*60)
        
        # Step 1: 加载文本
        logger.info("\n[Step 1/5] 加载文本")
        text = self.text_loader.load(text_path)
        logger.info(f"文本长度: {len(text)} 字符")
        
        # Step 2: 预处理
        logger.info("\n[Step 2/5] 文本预处理")
        text = self.preprocessor.preprocess(text)
        chapters = self.preprocessor.extract_chapters(text)
        
        # Step 3: 文本切分
        logger.info("\n[Step 3/5] 文本切分")
        chunks, metadata = self.text_splitter.split(text, chapters)
        
        # Step 4: 构建向量索引
        logger.info("\n[Step 4/5] 构建向量索引")
        vector_store = VectorStore.build(chunks, self.embedding_model, metadata)
        
        # Step 5: 构建BM25索引
        logger.info("\n[Step 5/5] 构建BM25索引")
        bm25_index = BM25Index.build(chunks, custom_dict_path)
        
        # 保存索引
        if save_indexes:
            logger.info("\n保存索引文件...")
            vector_store.save(self.config['vector_store_path'])
            bm25_index.save(self.config['bm25_index_path'])
            
            # 保存chunks和metadata
            with open(self.config['chunks_path'], 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            with open(self.config['metadata_path'], 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("知识库构建完成！")
        logger.info("="*60)
        
        return {
            'chunks': chunks,
            'metadata': metadata,
            'vector_store': vector_store,
            'bm25_index': bm25_index,
            'embedding_model': self.embedding_model
        }