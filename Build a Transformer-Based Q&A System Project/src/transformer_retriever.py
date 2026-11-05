"""
Transformer-based Retrieval using Sentence Embeddings
"""
from typing import List, Dict
from sentence_transformers import SentenceTransformer

if __package__:
    from .retrieval_utils import cosine_retrieve, normalize_rows, validate_k, validate_texts
else:
    from retrieval_utils import cosine_retrieve, normalize_rows, validate_k, validate_texts


class TransformerRetriever:
    """Semantic retrieval using transformer-based sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            batch_size: Maximum number of texts per model encoding batch
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = batch_size
        print(f"🤖 Loading transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
        print(f"✅ Model loaded: {model_name}")
        
    def build_index(self, corpus_texts: List[str]):
        """Build semantic index by encoding corpus texts."""
        validate_texts(corpus_texts, corpus=True)
        print("🧠 Building semantic index...")
        embeddings = self.model.encode(
            corpus_texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        self.corpus_embeddings = normalize_rows(embeddings)
        self.corpus = list(corpus_texts)
        
        print(f"✅ Semantic index built for {len(corpus_texts):,} documents")
        if len(corpus_texts) > 0 and self.corpus_embeddings is not None:
            print(f"   Embedding dimension: {self.corpus_embeddings.shape[1]}")
        
    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents using semantic similarity."""
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        validate_texts(query_texts)
        validate_k(k)
        if not query_texts or k == 0:
            return {i: [] for i in range(len(query_texts))}
            
        print(f"🔍 Running semantic retrieval for {len(query_texts)} queries...")
        
        query_embeddings = self.model.encode(
            query_texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        results = cosine_retrieve(query_embeddings, self.corpus_embeddings, k)
        
        print(f"✅ Retrieved top-{k} documents using semantic similarity")
        return results
