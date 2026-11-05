"""Shared validation and exact, bounded-memory vector ranking."""

from numbers import Integral

import numpy as np


def validate_texts(texts, *, corpus=False):
    if isinstance(texts, str) or not all(isinstance(text, str) for text in texts):
        raise TypeError("Expected a sequence of text strings.")
    if corpus and not texts:
        raise ValueError("Corpus must contain at least one document.")


def validate_k(k):
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
        raise ValueError("k must be a non-negative integer.")


def top_k_indices(scores, k):
    """Rank descending; break equal-score ties by ascending document index."""
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == len(scores):
        return np.argsort(-scores, kind="stable")
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k - len(higher)]
    candidates = np.concatenate((higher, tied))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


def cosine_retrieve(query_vectors, corpus_vectors, k):
    """Exact cosine search on normalized rows, including zero-vector ties."""
    validate_k(k)
    count = min(k, len(corpus_vectors))
    results = {i: [] for i in range(len(query_vectors))}
    if count == 0:
        return results
    queries = normalize_rows(query_vectors)
    for query_start in range(0, len(queries), 64):
        batch = queries[query_start:query_start + 64]
        best_scores = [np.empty(0, dtype=np.float32) for _ in batch]
        best_ids = [np.empty(0, dtype=np.int64) for _ in batch]
        for doc_start in range(0, len(corpus_vectors), 16384):
            scores = batch @ corpus_vectors[doc_start:doc_start + 16384].T
            for row, values in enumerate(scores):
                local = top_k_indices(values, count)
                ids = np.concatenate((best_ids[row], local + doc_start))
                values = np.concatenate((best_scores[row], values[local]))
                # Sorting IDs first preserves the global tie-breaking rule.
                order = np.argsort(ids)
                selected = top_k_indices(values[order], count)
                best_ids[row] = ids[order][selected]
                best_scores[row] = values[order][selected]
        for row, ids in enumerate(best_ids):
            results[query_start + row] = ids.tolist()
    return results
