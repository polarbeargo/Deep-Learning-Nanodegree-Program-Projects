"""Static word-embedding retrieval with validation-set parameter search."""

import logging
import re
import zlib
from typing import Dict, List

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

if __package__:
    from .retrieval_utils import cosine_retrieve, normalize_rows, validate_k, validate_texts
else:
    from retrieval_utils import cosine_retrieve, normalize_rows, validate_k, validate_texts

logger = logging.getLogger(__name__)


def _stable_hash(word):
    return zlib.adler32(word.encode("utf-8"))


class Word2VecRetriever:
    """Average corpus-trained word vectors and rank by cosine similarity."""

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1,
                 sg: int = 0, epochs: int = 10, seed: int = 42, workers: int = 1):
        for name, value in (("vector_size", vector_size), ("window", window),
                            ("min_count", min_count), ("epochs", epochs),
                            ("workers", workers)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if sg not in (0, 1):
            raise ValueError("sg must be 0 (CBOW) or 1 (skip-gram).")
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.seed = seed
        self.workers = workers
        self.model = None
        self.corpus_vectors = None
        self.corpus_texts = None
        self._normalized_vectors = None

    def _preprocess_text(self, text: str) -> List[str]:
        words = re.sub(r"[^a-zA-Z\s]", " ", text.lower()).split()
        return [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]

    def _text_to_vector(self, text: str) -> np.ndarray:
        if self.model is None:
            raise ValueError("Index not built. Call build_index() first.")
        vectors = [self.model.wv[word] for word in self._preprocess_text(text)
                   if word in self.model.wv]
        if vectors:
            return np.mean(vectors, axis=0, dtype=np.float32)
        return np.zeros(self.vector_size, dtype=np.float32)

    def build_index(self, corpus_texts: List[str]) -> None:
        validate_texts(corpus_texts, corpus=True)
        processed_docs = [self._preprocess_text(text) for text in corpus_texts]
        model = Word2Vec(
            vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, sg=self.sg, seed=self.seed,
            workers=self.workers, hashfxn=_stable_hash,
        )
        model.build_vocab(processed_docs)
        if not model.wv:
            raise ValueError("No vocabulary survives preprocessing and min_count.")
        model.train(processed_docs, total_examples=model.corpus_count, epochs=self.epochs)
        self.model = model
        self.corpus_vectors = np.stack([self._text_to_vector(text) for text in corpus_texts])
        self._normalized_vectors = normalize_rows(self.corpus_vectors)
        self.corpus_texts = list(corpus_texts)
        logger.info("Indexed %d documents with %d vocabulary terms.",
                    len(corpus_texts), len(model.wv))

    def retrieve(self, queries: List[str], k: int = 10) -> Dict[int, List[int]]:
        if self.model is None:
            raise ValueError("Index not built. Call build_index() first.")
        validate_texts(queries)
        validate_k(k)
        if not queries or k == 0:
            return {i: [] for i in range(len(queries))}
        vectors = np.stack([self._text_to_vector(query) for query in queries])
        unknown = np.flatnonzero(~np.any(vectors, axis=1))
        if len(unknown):
            logger.warning(
                "Queries %s have no known words; all cosine scores are zero, "
                "so ties are returned in document-index order.", unknown.tolist(),
            )
        return cosine_retrieve(vectors, self._normalized_vectors, k)

    def get_vocabulary_stats(self) -> dict:
        if self.model is None:
            raise ValueError("Index not built. Call build_index() first.")
        return {
            "vocabulary_size": len(self.model.wv),
            "vector_size": self.vector_size,
            "total_words_trained": self.model.corpus_total_words,
            "most_similar_to_search": (
                self.model.wv.most_similar("search", topn=5)
                if "search" in self.model.wv else None
            ),
        }

    @staticmethod
    def optimize_parameters(param_combinations, corpus_texts, query_texts,
                            qrels_dict, evaluator_class):
        """Select by validation MRR, then Recall@5; retain the trained winner."""
        if not param_combinations:
            raise ValueError("Provide at least one parameter configuration.")
        if not query_texts or not qrels_dict:
            raise ValueError("Parameter search requires validation queries and qrels.")
        best_config = None
        best_metrics = None
        best_retriever = None
        best_score = -np.inf
        best_recall = -np.inf
        results_log = []
        for params in param_combinations:
            retriever = Word2VecRetriever(**params)
            retriever.build_index(corpus_texts)
            results = retriever.retrieve(query_texts, k=20)
            metrics = evaluator_class.evaluate_retrieval(results, qrels_dict)
            score, recall = metrics["MRR"], metrics["Recall@5"]
            if not np.isfinite(score) or not np.isfinite(recall):
                raise ValueError(f"Non-finite validation metrics for {params}.")
            results_log.append({
                "config": dict(params), "mrr": score,
                "recall_5": recall, "metrics": metrics,
            })
            print(f"{params}: validation MRR={score:.4f}, Recall@5={recall:.4f}")
            if (score, recall) > (best_score, best_recall):
                best_score, best_recall = score, recall
                best_config, best_metrics = dict(params), metrics
                best_retriever = retriever
        return {
            "best_config": best_config, "best_score": best_score,
            "best_metrics": best_metrics, "results_log": results_log,
            "best_retriever": best_retriever,
        }
