"""
运行问答系统脚本
"""
import sys
import os
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.knowledge_base import VectorStore, BM25Index
from src.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from src.generation import LLMLoader, QAGenerator
from config import MODEL_CONFIG, INDEX_CONFIG, RETRIEVAL_CONFIG, GENERATION_CONFIG, LOG_CONFIG
from sentence_transformers import SentenceTransformer


class XiYouJiQA:
    """《西游记》问答系统"""
    
    def __init__(self, retriever, generator, chunks, metadata):
        self.retriever = retriever
        self.generator = generator
        self.chunks = chunks
        self.metadata = metadata
    
    def ask(self, question, top_k=3, verbose=False):
        """回答问题"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"问题: {question}")
            print(f"{'='*60}\n")
            print("正在检索相关内容...")
        
        # 检索
        contexts, indices = self.retriever.retrieve(question, top_k=top_k)
        
        if verbose:
            print(f"检索到 {len(contexts)} 个相关片段\n")
            for i, (ctx, idx) in enumerate(zip(contexts, indices), 1):
                meta = self.metadata[idx]
                chapter = meta.get('chapter_num', '?')
                print(f"[片段{i}] 第{chapter}回")
                print(f"{ctx[:100]}...\n")
            print("正在生成答案...\n")
        
        # 生成答案
        answer = self.generator.answer_question(
            question,
            contexts,
            [self.metadata[i] for i in indices]
        )
        
        if verbose:
            print(f"{'='*60}")
            print(f"答案:\n{answer}")
            print(f"{'='*60}\n")
        
        return answer


def main():
    parser = argparse.ArgumentParser(description='西游记RAG问答系统')
    parser.add_argument('--question', type=str, help='要提问的问题')
    parser.add_argument('--top_k', type=int, default=3, help='检索文档数量')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logger(log_dir=LOG_CONFIG['log_dir'], log_level=LOG_CONFIG['log_level'])
    
    print("正在加载系统...")
    
    # 加载索引
    print("  加载向量索引...")
    vector_store = VectorStore.load(INDEX_CONFIG['vector_store_path'])
    
    print("  加载BM25索引...")
    bm25_index = BM25Index.load(INDEX_CONFIG['bm25_index_path'])
    
    # 加载嵌入模型
    print("  加载嵌入模型...")
    embedding_model = SentenceTransformer(MODEL_CONFIG['embedding_model_name'])
    
    # 初始化检索器
    print("  初始化检索器...")
    bm25_retriever = BM25Retriever(bm25_index)
    vector_retriever = VectorRetriever(vector_store, embedding_model)
    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        vector_retriever,
        vector_store.chunks
    )
    
    # 加载LLM
    print("  加载大语言模型...")
    model, tokenizer = LLMLoader.load(MODEL_CONFIG)
    
    # 初始化生成器
    generator = QAGenerator(model, tokenizer, GENERATION_CONFIG)
    
    # 创建问答系统
    qa_system = XiYouJiQA(
        retriever=hybrid_retriever,
        generator=generator,
        chunks=vector_store.chunks,
        metadata=vector_store.metadata
    )
    
    print("系统加载完成！\n")
    
    if args.interactive:
        # 交互式模式
        print("=" * 60)
        print("《西游记》问答系统 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60 + "\n")
        
        while True:
            question = input("请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not question:
                continue
            
            answer = qa_system.ask(question, top_k=args.top_k, verbose=args.verbose)
            
            if not args.verbose:
                print(f"\n答案: {answer}\n")
                print("-" * 60 + "\n")
    
    else:
        # 单次问答模式
        if not args.question:
            print("错误: 请提供问题 (--question) 或使用交互模式 (--interactive)")
            return
        
        answer = qa_system.ask(args.question, top_k=args.top_k, verbose=args.verbose)
        
        if not args.verbose:
            print(f"问题: {args.question}")
            print(f"答案: {answer}")


if __name__ == '__main__':
    main()