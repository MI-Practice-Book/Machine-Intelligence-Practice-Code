# src/__init__.py
"""
西游记RAG问答系统
"""
__version__ = '1.0.0'

# 确保子模块可以被导入
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))