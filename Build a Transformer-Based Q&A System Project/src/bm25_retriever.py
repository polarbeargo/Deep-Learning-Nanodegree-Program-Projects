"""
BM25 Retrieval Implementation for Baseline Comparison
"""
from typing import List, Dict
from rank_bm25 import BM25Okapi
from tqdm import tqdm

if __package__:
    from .retrieval_utils import top_k_indices, validate_k, validate_texts
else:
    from retrieval_utils import top_k_indices, validate_k, validate_texts


class BM25Retriever:
    """Traditional keyword-based retrieval using BM25 algorithm."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever with tunable parameters.
        
        Args:
            k1 (float): Controls term frequency saturation (default: 1.2)
            b (float): Controls document length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = None
        
    def build_index(self, corpus_texts: List[str]):
        """Build BM25 index from corpus texts."""
        validate_texts(corpus_texts, corpus=True)
        tokenized_corpus = [text.lower().split() for text in corpus_texts]
        if not any(tokenized_corpus):
            raise ValueError("Corpus must contain at least one token.")
        print("🔧 Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.corpus = list(corpus_texts)
        
        print(f"✅ BM25 index built for {len(corpus_texts):,} documents")
        
    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents for each query."""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        validate_texts(query_texts)
        validate_k(k)
            
        print(f"🔍 Running BM25 retrieval for {len(query_texts)} queries...")
        results = {}
        
        for q_idx, query in enumerate(tqdm(query_texts, desc="Retrieving")):
            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            results[q_idx] = top_k_indices(doc_scores, k).tolist()
            
        print(f"✅ Retrieved top-{k} documents for {len(query_texts)} queries")
        return results
