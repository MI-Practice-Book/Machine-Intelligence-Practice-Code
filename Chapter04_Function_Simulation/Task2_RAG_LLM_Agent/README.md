# 《西游记》RAG问答系统

基于检索增强生成(RAG)技术的《西游记》智能问答系统。

## 功能特点

- 🔍 混合检索策略（BM25 + 向量检索）
- 🤖 基于大语言模型的智能问答
- 📚 完整的《西游记》知识库
- 🎯 高准确率的答案生成
- 📖 可追溯的原文引用

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据

将《西游记》文本文件放置在 `data/raw/xiyouji.txt`

### 3. 构建知识库
```bash
python scripts/build_knowledge_base.py
```

### 4. 运行问答系统

**命令行模式：**
```bash
# 单次问答
python scripts/run_qa_system.py --question "孙悟空最终被封为什么？"

# 交互模式
python scripts/run_qa_system.py --interactive --verbose
```

**Web应用模式：**
```bash
streamlit run web_app/app.py
```

## 项目结构
```
xiyouji-rag-qa/
├── config.py              # 配置文件
├── src/                   # 源代码
│   ├── data_processing/   # 数据处理
│   ├── knowledge_base/    # 知识库
│   ├── retrieval/         # 检索
│   └── generation/        # 生成
├── scripts/               # 可执行脚本
└── web_app/               # Web应用
```

## 技术栈

- 嵌入模型: BAAI/bge-small-zh-v1.5
- LLM: Qwen/Qwen2.5-3B-Instruct
- 检索: BM25 + 向量检索 + RRF融合
- 框架: PyTorch, Transformers, Streamlit

## 配置说明

在 `config.py` 中可以调整以下参数：

- `chunk_size`: 文本块大小
- `chunk_overlap`: 重叠窗口大小
- `top_k`: 检索文档数量
- `temperature`: 生成温度
- `use_quantization`: 是否使用量化

