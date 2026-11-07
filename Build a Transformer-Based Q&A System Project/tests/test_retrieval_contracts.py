"""Numerical, edge-case and application contracts beyond the starter tests."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit"))

from bm25_retriever import BM25Retriever
from evaluator import IRMetrics
from retrieval_utils import cosine_retrieve, normalize_rows, top_k_indices
from transformer_retriever import TransformerRetriever
from word2vec_retriever import Word2VecRetriever
from data_loader import DataLoader


class TestExactRanking(unittest.TestCase):
    def test_partial_selection_matches_stable_sort(self):
        rng = np.random.default_rng(42)
        for size in (1, 10, 100):
            scores = rng.integers(-2, 3, size).astype(np.float32)
            for k in (0, 1, 5, size, size + 10):
                expected = np.argsort(-scores, kind="stable")[:k]
                np.testing.assert_array_equal(top_k_indices(scores, k), expected)

    def test_cosine_across_query_and_document_blocks(self):
        rng = np.random.default_rng(42)
        corpus = rng.normal(size=(16405, 8)).astype(np.float32)
        queries = rng.normal(size=(65, 8)).astype(np.float32)
        corpus[0] = 0
        queries[0] = 0
        expected = np.argsort(-cosine_similarity(queries, corpus), axis=1,
                              kind="stable")[:, :7]
        actual = cosine_retrieve(queries, normalize_rows(corpus), 7)
        for query_id in range(len(queries)):
            np.testing.assert_array_equal(actual[query_id], expected[query_id])

    def test_zero_and_negative_scores(self):
        corpus = normalize_rows(np.array([[0, 0], [-1, 0], [1, 0], [1, 0]]))
        results = cosine_retrieve(np.array([[1, 0], [0, 0]]), corpus, 20)
        self.assertEqual(results, {0: [2, 3, 0, 1], 1: [0, 1, 2, 3]})


class TestMetricEdgeCases(unittest.TestCase):
    def test_missing_queries_nonpositive_judgments_and_duplicates(self):
        results = {0: [1, 1, 2], 99: [1]}
        qrels = {0: {1: 1, 2: 0, 3: -1}, 1: {4: 2}, 2: {}}
        self.assertAlmostEqual(IRMetrics.recall_at_k(results, qrels, 3), 1 / 3)
        self.assertAlmostEqual(IRMetrics.precision_at_k(results, qrels, 3), 1 / 9)
        self.assertAlmostEqual(IRMetrics.mrr(results, qrels), 1 / 3)

    def test_precision_denominator_and_reciprocal_rank(self):
        self.assertEqual(IRMetrics.precision_at_k({0: [3]}, {0: {3: 1}}, 5), 0.2)
        self.assertEqual(IRMetrics.mrr({0: [0, 1, 2, 3]}, {0: {3: 1}}), 0.25)
        self.assertEqual(IRMetrics.mrr({0: []}, {0: {3: 1}}), 0.0)

    def test_empty_and_invalid_cutoffs(self):
        self.assertEqual(IRMetrics.evaluate_retrieval({}, {})["MRR"], 0.0)
        for metric in (IRMetrics.recall_at_k, IRMetrics.precision_at_k):
            self.assertEqual(metric({0: [1]}, {0: {1: 1}}, 0), 0.0)
            for k in (-1, 1.5, True):
                with self.assertRaises(ValueError):
                    metric({}, {}, k)


class TestDatasetSampling(unittest.TestCase):
    def test_sampling_is_stable_and_preserves_global_random_state(self):
        corpus = {str(i): {"text": f"document {i}"} for i in range(8)}
        queries = {f"q{i}": f"question {i}" for i in range(8)}
        qrels = {f"q{i}": {str(i): 1} for i in range(8)}
        np.random.seed(123)
        expected = np.random.random(3)
        np.random.seed(123)
        with patch("data_loader.load_beir_nq", return_value=(corpus, queries, qrels)):
            first = DataLoader().load_dataset(query_sample_size=4)
            second = DataLoader().load_dataset(query_sample_size=4)
        np.testing.assert_array_equal(np.random.random(3), expected)
        self.assertEqual(list(first["corpus"]), sorted(first["corpus"]))
        self.assertEqual(first, second)


class TestRetrieverContracts(unittest.TestCase):
    corpus = ["machine learning search", "neural vision images", "pets animals dogs"]

    def make_transformer(self):
        embeddings = {
            "positive": [1.0, 0.0], "negative": [-1.0, 0.0],
            "orthogonal": [0.0, 1.0], "query": [1.0, 0.0],
        }
        with patch("transformer_retriever.SentenceTransformer") as model:
            model.return_value.encode.side_effect = lambda texts, **kwargs: np.array(
                [embeddings[text] for text in texts], dtype=np.float32,
            )
            retriever = TransformerRetriever(batch_size=2)
        return retriever

    def test_transformer_encodes_only_queries_after_build(self):
        retriever = self.make_transformer()
        retriever.build_index(["positive", "negative", "orthogonal"])
        self.assertEqual(retriever.corpus_embeddings.shape, (3, 2))
        self.assertEqual(retriever.retrieve(["query"], 3), {0: [0, 2, 1]})
        self.assertEqual(retriever.model.encode.call_count, 2)
        self.assertTrue(retriever.model.encode.call_args.kwargs["convert_to_numpy"])
        self.assertEqual(retriever.model.encode.call_args.args[0], ["query"])

    def test_shared_empty_queries_invalid_k_and_clamping(self):
        transformer = self.make_transformer()
        transformer.build_index(["positive", "negative", "orthogonal"])
        for retriever in (BM25Retriever(), Word2VecRetriever(epochs=5)):
            retriever.build_index(self.corpus)
            for k in (-1, 2.5, True):
                with self.assertRaises(ValueError):
                    retriever.retrieve(["query"], k)
            self.assertEqual(retriever.retrieve([], 3), {})
            self.assertEqual(retriever.retrieve(["query"], 0), {0: []})
            self.assertEqual(len(retriever.retrieve(["learning"], 100)[0]), 3)
        self.assertEqual(transformer.retrieve([], 3), {})
        self.assertEqual(transformer.retrieve(["query"], 0), {0: []})
        self.assertEqual(transformer.retrieve(["query"], 100), {0: [0, 2, 1]})
        with self.assertRaises(ValueError):
            transformer.retrieve(["query"], -1)

    def test_unbuilt_and_empty_corpora_raise(self):
        for retriever in (BM25Retriever(), Word2VecRetriever(), self.make_transformer()):
            with self.assertRaises(ValueError):
                retriever.retrieve(["query"])
            with self.assertRaises(ValueError):
                retriever.build_index([])
        with self.assertRaises(ValueError):
            BM25Retriever().build_index([" ", ""])
        with self.assertRaises(ValueError):
            Word2VecRetriever().build_index(["the and of", ""])

    def test_bm25_matches_library_scores_and_lowercases(self):
        retriever = BM25Retriever()
        retriever.build_index(self.corpus)
        expected = np.argsort(-retriever.bm25.get_scores(["machine", "learning"]),
                              kind="stable")[:2].tolist()
        self.assertEqual(retriever.retrieve(["MACHINE LEARNING"], 2), {0: expected})

    def test_word2vec_averaging_oov_and_reproducibility(self):
        first = Word2VecRetriever(vector_size=8, epochs=5)
        first.build_index(self.corpus)
        expected = np.mean([first.model.wv["machine"], first.model.wv["learning"]], axis=0)
        np.testing.assert_allclose(first._text_to_vector("MACHINE learning"), expected)
        with self.assertLogs("word2vec_retriever", level="WARNING"):
            self.assertEqual(first.retrieve(["zzzzunknown"], 3), {0: [0, 1, 2]})
        second = Word2VecRetriever(vector_size=8, epochs=5)
        second.build_index(self.corpus)
        np.testing.assert_array_equal(first.corpus_vectors, second.corpus_vectors)

    def test_parameter_search_returns_measured_winner(self):
        configs = [
            dict(vector_size=8, epochs=5),
            dict(vector_size=12, epochs=5, sg=1),
        ]
        queries, qrels = ["machine learning", "pets dogs"], {0: {0: 1}, 1: {2: 1}}
        result = Word2VecRetriever.optimize_parameters(
            configs, self.corpus, queries, qrels, IRMetrics,
        )
        self.assertEqual(len(result["results_log"]), 2)
        best = max(result["results_log"], key=lambda row: (row["mrr"], row["recall_5"]))
        self.assertEqual(result["best_config"], best["config"])
        metrics = IRMetrics.evaluate_retrieval(
            result["best_retriever"].retrieve(queries, 20), qrels,
        )
        self.assertEqual(metrics, result["best_metrics"])
        with self.assertRaises(ValueError):
            Word2VecRetriever.optimize_parameters([], self.corpus, queries, qrels, IRMetrics)
        with self.assertRaises(ValueError):
            Word2VecRetriever.optimize_parameters(
                [dict(min_count=100)], self.corpus, queries, qrels, IRMetrics,
            )


if __name__ == "__main__":
    unittest.main()
