# Transformer-Based Semantic Retrieval

Compare BM25, corpus-trained Word2Vec and pretrained SentenceTransformers on
identical queries and relevance judgments, then use them in a Streamlit RAG demo.

## Run

Use your existing Python environment; a new virtual environment is not required.
From this project directory:

```bash
pip install -r requirements.txt
# Run all tests (initially many will fail - this is expected!)
python -m pytest tests/ -v

# Test individual components as you implement them
python tests/test_bm25_retriever.py        # BM25 keyword search implementation
python tests/test_evaluator.py             # IR metrics calculation
python tests/test_word2vec_retriever.py     # Word2Vec embedding implementation  
python tests/test_transformer_retriever.py  # Transformer semantic search

# Final validation - all tests should pass when complete
python -m pytest tests/ -v

```

### **Try the Interactive Demo**
```bash
# Navigate to streamlit directory
cd streamlit

# Configure OpenAI API (optional)
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your_openai_api_key_here

# Launch demo
streamlit run app.py
```

The transformer downloads Hugging Face model weights on first use and reuses
the local model cache afterward. Tests for the real transformer and Streamlit
require these weights; the numerical contract tests mock the encoder to verify
ranking independently of pretrained model quality.

The notebook defaults to the bundled fictional TechFlow AI dataset for a fast,
end-to-end run. Change `DATASET` to `"nq"` for a 200-query BEIR Natural Questions
experiment (a large initial dataset download). Run all cells in order and save
the executed notebook to retain the measured table, rankings and PCA plot.
Larger transformer comparisons are opt-in via `RUN_EXTRA_MODELS`.

## Implementation

| Module | Responsibility |
|---|---|
| [bm25_retriever.py](src/bm25_retriever.py) | Lowercase/whitespace tokenization, BM25Okapi index, ranked document IDs |
| [word2vec_retriever.py](src/word2vec_retriever.py) | Corpus-only Word2Vec training, averaged vectors, cosine search, validation parameter search |
| [transformer_retriever.py](src/transformer_retriever.py) | Batched sentence-transformer document/query encoding and cosine search |
| [retrieval_utils.py](src/retrieval_utils.py) | Shared exact, bounded-memory cosine ranking and deterministic top-k selection |
| [evaluator.py](src/evaluator.py) | Recall@k, Precision@k and MRR implemented directly from relevance judgments |
| [unified_retrieval_comparison.ipynb](notebooks/unified_retrieval_comparison.ipynb) | Query-disjoint tuning/evaluation, quality/runtime comparison, PCA, RAG explanation |
| [app.py](streamlit/app.py) | Cached retrievers, common evaluation metrics, optional generation |

### Retrieval API and efficiency

```python
from src.transformer_retriever import TransformerRetriever

retriever = TransformerRetriever("all-MiniLM-L6-v2", batch_size=64)
retriever.build_index(["A document about neural networks.", "A guide to gardening."])
results = retriever.retrieve(["How do neural networks work?"], k=1)
# {0: [document_id]} -- IDs are zero-based positions in the original corpus.
```

All three retrievers return `{query_index: [document_index, ...]}` in descending
score order. Equal scores break ties by ascending document index. `k` must be a
non-negative integer: zero yields empty rankings; oversized values are capped
at corpus size. Empty query lists return `{}`. Unbuilt indexes, empty corpora
and invalid inputs raise explicit errors. BM25 rejects all-blank corpora;
Word2Vec rejects corpora with no surviving vocabulary.

SentenceTransformers returns float32 NumPy embeddings with shape
`(n_samples, embedding_dim)` (384 dimensions for MiniLM). Corpus embeddings are
computed once per index; queries are batch-encoded with the same encoder.
Normalized dot products implement cosine similarity. Search uses blocks of
64 queries and 16,384 documents, limiting the score block to about 4 MiB, and
partial selection rather than sorting the entire corpus for small k. This is
exact search, not an approximate nearest-neighbor index; computation still
scales with the number of query/document pairs. For much larger corpora,
consider an ANN index after measuring the recall/latency trade-off.

Word2Vec averages in-vocabulary vectors and caches normalized document vectors.
Queries with no known words produce zero vectors; a warning explains that all
scores tie and the returned IDs carry no semantic evidence. Defaults use a
fixed seed, stable vocabulary hash and one worker for reproducible training.
You may increase `workers` for large corpora at the cost of reproducibility.
Parameter search selects validation MRR, then Recall@5, and returns both its log
and the trained `best_retriever`, avoiding a redundant training run. Failures are
not suppressed or replaced with synthetic metrics.

### Evaluation conventions

- Only relevance scores **greater than zero** count as relevant.
- Recall@k counts distinct relevant IDs among the first k ranks, divided by
  the number of relevant documents.
- Precision@k divides that count by **k**, even for shorter rankings.
- MRR averages `1 / first_relevant_rank`, or zero if none was returned.
- All queries in qrels contribute to the macro-average. Missing results and
  queries without positive judgments score zero; extra unjudged query IDs are
  ignored. Duplicate IDs cannot inflate recall or precision. Empty evaluations
  and k=0 cutoffs yield zero.
- Notebook rankings are capped at 20, so its MRR is effectively **MRR@20**.
  The demo displays reciprocal rank at the selected cutoff for a single query.

The notebook separates tuning from held-out queries before Word2Vec search;
all final methods use the same held-out queries. The demo has only five
documents, so Recall@5/@10 saturate, and paraphrases can cross the query split.
Treat it as an integration demonstration, not a generalization benchmark.
The existing NQ sampler retains documents judged for sampled queries, rather
than the entire NQ corpus; reported scores apply to that reduced candidate pool.
Transformer pretraining uses external data whereas Word2Vec trains only on the
provided corpus. Those are different training budgets, not a controlled
architecture-only comparison.

The notebook prints actual metrics, setup/query timings, differences from BM25,
and example rankings. No method is assumed to win. Setup time includes model
loading/download or Word2Vec tuning; it is not a pure index-building benchmark.
PCA jointly projects document/query vectors into 2D with an explained-variance
label. Its geometry illustrates associations, not a replacement for IR metrics.
Long documents are truncated to the sentence model's token limit: production
applications should index overlapping text chunks with source metadata.

## Retrieval-Augmented Generation

RAG performs **retrieval followed by generation**. A retriever selects relevant
documents or chunks from an indexed knowledge base. A generator, usually an
LLM, receives the question and retrieved passages and synthesizes a grounded
answer with source citations. Retrieval does not change generator weights.
Incomplete or irrelevant evidence can still produce incorrect answers, so
retrieval metrics and grounded-answer evaluation are both needed.

Search-enabled ChatGPT, Perplexity and code-context features in Copilot
illustrate this pattern; their production systems are more complex and need
not use these particular retrieval models. BM25 excels at exact terms,
Word2Vec adds static word associations, and transformers capture contextual
sentence meaning and paraphrases. Which performs best depends on the corpus,
queries and deployment budget; hybrid search can combine their strengths.

### Optional generation in the demo

Retrieval requires **no API key**. Generation is **off by default**, even when a
key exists. To use it, configure `OPENAI_API_KEY` via your environment or a local
`.env` file using [the example](streamlit/.env.example), then opt in in the UI.
Never commit the key. Generation sends the query and selected passages to an
external provider: use only non-sensitive information you are authorized to
share. Do not send private code, personal data or credentials. The app warns
before generation and reuses already retrieved passages instead of repeating
all searches. No external generation call is needed for tests or evaluation.
