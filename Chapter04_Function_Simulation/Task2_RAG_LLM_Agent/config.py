"""
配置文件：集中管理所有配置参数
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_CONFIG = {
    'raw_text_path': os.path.join(PROJECT_ROOT, 'data/raw/xiyouji.txt'),
    'custom_dict_path': os.path.join(PROJECT_ROOT, 'data/custom_dict.txt'),
}

# 模型配置
MODEL_CONFIG = {
    # 嵌入模型
    'embedding_model_name': 'BAAI/bge-small-zh-v1.5',
    'embedding_dim': 512,
    
    # LLM
    'llm_model_name': 'Qwen/Qwen2.5-3B-Instruct',
    'use_quantization': False,  # 是否使用4-bit量化
    'device_map': 'auto',
    'torch_dtype': 'bfloat16',  # 'float32', 'float16', 'bfloat16'
}

# 文本切分配置
SPLIT_CONFIG = {
    'chunk_size': 500,
    'chunk_overlap': 100,
}

# 生成配置（可能需要调整）
GENERATION_CONFIG = {
    'max_new_tokens': 256,        # 减少最大长度，避免冗长
    'temperature': 0.15,            # 降低温度，提高确定性
    'top_p': 0.9,                 # 降低top_p，减少随机性
    'repetition_penalty': 1.2,     # 添加重复惩罚
}

# 检索配置（可能需要调整）
RETRIEVAL_CONFIG = {
    'top_k': 3,                    # 保持3个结果
    'retrieve_k': 10,
    'fusion_k': 60,
    'similarity_threshold': 0.3,   # 降低阈值，允许更多候选
}

# 索引路径配置
INDEX_CONFIG = {
    'vector_store_path': os.path.join(PROJECT_ROOT, 'indexes/vector_store.npz'),
    'bm25_index_path': os.path.join(PROJECT_ROOT, 'indexes/bm25_index.pkl'),
    'chunks_path': os.path.join(PROJECT_ROOT, 'indexes/chunks.json'),
    'metadata_path': os.path.join(PROJECT_ROOT, 'indexes/metadata.json'),
}

# 日志配置
LOG_CONFIG = {
    'log_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'log_level': 'INFO',
}

# 创建必要的目录
os.makedirs(os.path.join(PROJECT_ROOT, 'indexes'), exist_ok=True)
os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'data/raw'), exist_ok=True)