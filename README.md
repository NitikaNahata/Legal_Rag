# ASK-LEGALBOT — Legal RAG Pipeline

A hybrid retrieval-augmented generation system over legal contract datasets (CUAD, MAUD, ContractNLI, PrivacyQA).

---

## Architecture

```
.txt files (698)
      │
      ▼
  Clause-aware chunker  →  ≤500-token chunks (50 overlap)
      │
      ├──► SentenceTransformer BGE  →  768-dim L2-norm vectors  →  ChromaDB (cosine)
      │    (PyTorch + CUDA GPU)
      │
      └──► BM25Okapi tokenizer      →  BM25 index               →  bm25_index/
```

**Retrieval:**  BM25 top-20 + Dense top-20 → Weighted RRF (1:3) → pool-15 → MiniLM-L6 reranker → top-5 → Ollama LLM

---

## What We Fixed (Iteration History)

### Problem 1 — BGE token limit exceeded (silent truncation)
**What was wrong:** `CHUNK_SIZE = 800` tiktoken tokens. BAAI/bge-base-en-v1.5 has a **512 WordPiece token limit**. Since tiktoken (cl100k_base) produces fewer tokens than BERT WordPiece for the same text (~1.2–1.5× ratio), 800 tiktoken tokens ≈ 960–1200 WordPiece tokens. The model silently truncated every chunk to 512, meaning the last ~300–500 tokens of every chunk were **never embedded**.

**Fix:** Reduced `CHUNK_SIZE = 500` tiktoken tokens, which safely maps to ≤512 WordPiece tokens with margin for `[CLS]`/`[SEP]`.

---

### Problem 2 — Blind fixed-size chunking split legal clauses mid-sentence
**What was wrong:** The original chunker used a pure token-window slide (stride 700), cutting through legal clauses at arbitrary points. This split clauses like `2.1 INDEMNIFICATION` mid-paragraph, losing the semantic unit.

**Fix:** Implemented a **3-tier clause-aware chunker**:
1. **Tier 1 (clause split):** Regex splits on `1.`, `2.1`, `(a)`, `(iv)` boundaries — keeps each clause intact
2. **Tier 2 (sentence split):** Fallback for prose-style files (e.g. PrivacyQA) with no clause markers
3. **Tier 3 (token window):** Fallback for any single clause that exceeds 500 tokens

Chunks went from 21,959 → **38,576** (more chunks, but each is a complete legal unit).

---

### Problem 3 — ONNX/fastembed on CPU, no GPU utilization
**What was wrong:** The original pipeline used `fastembed` (ONNX runtime, CPU only). With an RTX 4070 Laptop GPU available, this wasted hardware.

**Fix:** Switched to `sentence-transformers` (PyTorch backend) with `device="cuda"`. Added `normalize_embeddings=True` since BGE is trained with L2 normalization.

---

### Problem 4 — Per-file embedding caused GPU starvation
**What was wrong:** Even after switching to GPU, embedding was done file-by-file. Each file had only ~10–30 chunks, so GPU batches were tiny and the GPU spent most of its time idle between files. Result: ~29s per batch of 256, ~73 minutes estimated.

**Fix:** Restructured `main()` into two phases:
- **Phase 1 (CPU):** Chunk all 698 files → accumulate ~38k chunks in memory
- **Phase 2 (GPU):** Embed all 38k chunks in one continuous pass (`batch_size=64`)

Result: ~1.16 batches/sec → **~10 minutes total** (25× faster).

---

### Problem 5 — Batch size 256 caused VRAM saturation
**What was wrong:** `EMBED_BATCH = 256` with 500-token chunks used 7.9/8.2GB VRAM, running the GPU at memory bandwidth limit (not compute limit). This caused memory management overhead and slow throughput.

**Fix:** Reduced `EMBED_BATCH = 64`. Less VRAM pressure → better memory throughput → 25× speedup.

---

### Problem 6 — retrieve.py used wrong embedder and collection name
**What was wrong:** `retrieve.py` still used `fastembed.TextEmbedding` (ONNX/CPU) for query encoding, and `COLLECTION_NAME = "privacy_qa"` instead of `"legal_docs"`. This would produce embedding vectors from a different backend than what was stored in ChromaDB (mismatch in normalization), and query the wrong collection.

**Fix:** Updated `retrieve.py` to use `SentenceTransformer` with `normalize_embeddings=True` and correct collection name `"legal_docs"`.

---

## Files

| File | Description |
|---|---|
| `ingest.py` | Chunking + embedding + BM25 indexing pipeline |
| `retrieve.py` | Hybrid retrieval (BM25 + Dense + RRF + reranker) + Ollama LLM |
| `chroma_db/` | ChromaDB persistent store (38,576 vectors, 768-dim, cosine) |
| `bm25_index/` | BM25 index + texts + metadata JSON |
| `cuad/` | 462 commercial contract .txt files |
| `maud/` | 134 M&A agreement .txt files |
| `contractnli/` | 95 NDA .txt files |
| `privacy_qa/` | 7 privacy policy .txt files |

---

## Running

```bash
# Re-ingest (only needed if corpus changes)
python ingest.py

# Start the QA chatbot
python retrieve.py
```

Requires Ollama running locally with `llama3.1:8b` pulled.
