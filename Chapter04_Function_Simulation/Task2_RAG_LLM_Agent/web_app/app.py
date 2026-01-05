"""
Streamlit Web应用
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from sentence_transformers import SentenceTransformer

from src.knowledge_base import VectorStore, BM25Index
from src.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from src.generation import LLMLoader, QAGenerator
from config import MODEL_CONFIG, INDEX_CONFIG, RETRIEVAL_CONFIG, GENERATION_CONFIG


@st.cache_resource
def load_system():
    """加载系统（使用缓存）"""
    # 加载索引
    vector_store = VectorStore.load(INDEX_CONFIG['vector_store_path'])
    bm25_index = BM25Index.load(INDEX_CONFIG['bm25_index_path'])
    
    # 加载嵌入模型
    embedding_model = SentenceTransformer(MODEL_CONFIG['embedding_model_name'])
    
    # 初始化检索器
    bm25_retriever = BM25Retriever(bm25_index)
    vector_retriever = VectorRetriever(vector_store, embedding_model)
    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        vector_retriever,
        vector_store.chunks
    )
    
    # 加载LLM
    model, tokenizer = LLMLoader.load(MODEL_CONFIG)
    generator = QAGenerator(model, tokenizer, GENERATION_CONFIG)
    
    return hybrid_retriever, generator, vector_store


def main():
    st.set_page_config(
        page_title="《西游记》问答系统",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 《西游记》智能问答系统")
    st.markdown("基于RAG技术的名著问答Agent")
    
    # 加载系统
    with st.spinner("正在加载系统..."):
        retriever, generator, vector_store = load_system()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        top_k = st.slider("检索文档数量", 1, 10, 3)
        show_contexts = st.checkbox("显示检索的原文片段", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 系统信息")
        st.info(f"""
        - 知识库大小: {len(vector_store.chunks)} 个文本块
        - 嵌入模型: {MODEL_CONFIG['embedding_model_name']}
        - LLM: {MODEL_CONFIG['llm_model_name']}
        """)
    
    # 主界面
    question = st.text_input(
        "请输入您的问题：",
        placeholder="例如：孙悟空的师傅是谁？"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 提问", type="primary", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("正在思考..."):
            # 检索
            contexts, indices = retriever.retrieve(question, top_k=top_k)
            
            # 生成答案
            answer = generator.answer_question(
                question,
                contexts,
                [vector_store.metadata[i] for i in indices]
            )
        
        # 显示答案
        st.success("💡 答案")
        st.write(answer)
        
        # 显示检索的原文
        if show_contexts:
            st.markdown("---")
            st.subheader("📖 参考原文片段")
            
            for i, (ctx, idx) in enumerate(zip(contexts, indices), 1):
                meta = vector_store.metadata[idx]
                chapter = meta.get('chapter_num', '?')
                chapter_title = meta.get('chapter_title', '')
                
                with st.expander(f"片段 {i} - 第{chapter}回: {chapter_title}"):
                    st.write(ctx)
    
    elif ask_button and not question:
        st.warning("请输入问题")
    
    # 示例问题
    st.markdown("---")
    st.subheader("💡 示例问题")
    
    examples = [
        "孙悟空的师傅是谁？",
        "金箍棒有多重？",
        "师徒四人经历了多少难？",
        "猪八戒的前世是什么？",
        "唐僧在哪里收的沙僧？",
        "孙悟空是怎么被压在五行山下的？"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.question = example
                st.rerun()


if __name__ == '__main__':
    main()