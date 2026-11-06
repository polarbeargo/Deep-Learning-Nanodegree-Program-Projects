"""
Semantic Retrieval Project - Core Modules

This package contains the core modules for building a semantic retrieval system:
- data_loader: Load and preprocess datasets
- bm25_retriever: Keyword-based retrieval
- word2vec_retriever: Static embedding retrieval
- transformer_retriever: Pretrained semantic retrieval
- evaluator: Evaluate retrieval performance
"""

from .data_loader import DataLoader
from .bm25_retriever import BM25Retriever
from .word2vec_retriever import Word2VecRetriever
from .transformer_retriever import TransformerRetriever
from .evaluator import IRMetrics

__version__ = "1.0.0"
__author__ = "Udacity Deep Learning Nanodegree"

__all__ = [
    "DataLoader",
    "BM25Retriever",
    "Word2VecRetriever",
    "TransformerRetriever",
    "IRMetrics",
]
