"""
构建知识库脚本
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.knowledge_base import KnowledgeBaseBuilder
from config import DATA_CONFIG, MODEL_CONFIG, SPLIT_CONFIG, INDEX_CONFIG, LOG_CONFIG


def main():
    # 配置日志
    setup_logger(log_dir=LOG_CONFIG['log_dir'], log_level=LOG_CONFIG['log_level'])
    
    # 合并配置
    build_config = {
        **MODEL_CONFIG,
        **SPLIT_CONFIG,
        **INDEX_CONFIG
    }
    
    # 创建构建器
    builder = KnowledgeBaseBuilder(build_config)
    
    # 构建知识库
    knowledge_base = builder.build(
        text_path=DATA_CONFIG['raw_text_path'],
        custom_dict_path=DATA_CONFIG.get('custom_dict_path'),
        save_indexes=True
    )
    
    print("\n知识库构建成功！")
    print(f"  文本块数量: {len(knowledge_base['chunks'])}")
    print(f"  向量索引: {INDEX_CONFIG['vector_store_path']}")
    print(f"  BM25索引: {INDEX_CONFIG['bm25_index_path']}")


if __name__ == '__main__':
    main()