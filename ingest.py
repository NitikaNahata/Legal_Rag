"""
Phase 1 + 2 RAG Ingestion Pipeline
- Clause-aware chunking (500 tokens max, 50 overlap) — respects legal clause boundaries
- Dense embeddings: BAAI/bge-base-en-v1.5 via sentence-transformers (PyTorch + GPU)
- Sparse index:     BM25 via rank-bm25             (exact keyword matching)
- Stores dense vectors in ChromaDB
- Stores BM25 index + corpus metadata to bm25_index/
"""

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import tiktoken
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CORPUS_DIRS     = ["cuad", "maud", "contractnli", "privacy_qa"]
CHROMA_DIR      = BASE_DIR / "chroma_db"
BM25_DIR        = BASE_DIR / "bm25_index"
COLLECTION_NAME = "legal_docs"

CHUNK_SIZE    = 500   # fits within BGE's 512 WordPiece limit (with margin)
CHUNK_OVERLAP = 50

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim
EMBED_BATCH = 64    # keeps VRAM usage low for faster throughput on 8GB GPU

enc = tiktoken.get_encoding("cl100k_base")

# Regex: lines that start a new legal clause/section
_CLAUSE_BOUNDARY = re.compile(
    r"(?m)^(?="
    r"\d+\.\d+\s+[A-Z]"      # 2.1 SUBSECTION
    r"|\d+\.\s+[A-Z]"         # 1. SECTION
    r"|\([a-z]\)\s+"           # (a) lettered item
    r"|\([ivxlcdm]+\)\s+"      # (iv) roman numeral item
    r")"
)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _tok(text: str) -> int:
    return len(enc.encode(text, disallowed_special=()))


def _split_into_units(text: str) -> list[str]:
    """
    Tier 1: split on clause boundaries (numbered sections, lettered items).
    Tier 2: if no clause boundaries found, split on sentence endings.
    Returns a list of raw text units (may still be over CHUNK_SIZE individually).
    """
    parts = _CLAUSE_BOUNDARY.split(text)
    parts = [p for p in parts if p.strip()]

    # If clause splitting produced nothing useful, fall back to sentence splitting
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p for p in parts if p.strip()]

    return parts


def _pack_units(units: list[str], full_text: str) -> list[dict]:
    """
    Greedily pack units into chunks of ≤ CHUNK_SIZE tokens.
    Oversized single units are split by token window with CHUNK_OVERLAP.
    char_start / char_end are computed by scanning full_text.
    """
    chunks = []
    search_offset = 0  # cursor into full_text for char position lookup

    def find_char_pos(snippet: str) -> tuple[int, int]:
        nonlocal search_offset
        idx = full_text.find(snippet[:80], search_offset)
        if idx == -1:
            idx = full_text.find(snippet[:40])  # fallback wider search
        if idx == -1:
            idx = search_offset
        search_offset = idx + len(snippet)
        return idx, idx + len(snippet)

    def flush(text: str):
        text = text.strip()
        if not text:
            return
        toks = _tok(text)
        if toks <= CHUNK_SIZE:
            cs, ce = find_char_pos(text)
            chunks.append({"text": text, "char_start": cs, "char_end": ce, "token_count": toks})
        else:
            # Token-window fallback for oversized units
            token_ids = enc.encode(text, disallowed_special=())
            stride = CHUNK_SIZE - CHUNK_OVERLAP
            start = 0
            while start < len(token_ids):
                end = min(start + CHUNK_SIZE, len(token_ids))
                chunk_text = enc.decode(token_ids[start:end])
                cs, ce = find_char_pos(chunk_text)
                chunks.append({"text": chunk_text, "char_start": cs, "char_end": ce,
                                "token_count": end - start})
                start += stride

    current = ""
    for unit in units:
        candidate = (current + "\n" + unit).strip() if current else unit
        if _tok(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            flush(current)
            # If the unit itself is already too big, flush it directly (will be split)
            if _tok(unit) > CHUNK_SIZE:
                flush(unit)
                current = ""
            else:
                current = unit

    flush(current)

    # Add overlap: prepend tail of previous chunk to each chunk
    if CHUNK_OVERLAP > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = enc.encode(chunks[i - 1]["text"], disallowed_special=())
            overlap_text = enc.decode(prev_tokens[-CHUNK_OVERLAP:])
            merged = (overlap_text + " " + chunks[i]["text"]).strip()
            if _tok(merged) <= CHUNK_SIZE:
                overlapped.append({**chunks[i], "text": merged,
                                    "token_count": _tok(merged)})
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return chunks


def chunk_document(text: str) -> list[dict]:
    if not text.strip():
        return []
    units = _split_into_units(text)
    return _pack_units(units, text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def iter_corpus_files() -> Generator[tuple[str, Path], None, None]:
    for dataset in CORPUS_DIRS:
        corpus_path = BASE_DIR / dataset
        if not corpus_path.exists():
            print(f"  [WARN] Not found: {corpus_path}")
            continue
        txt_files = sorted(corpus_path.glob("*.txt"))
        print(f"  {dataset}: {len(txt_files)} files")
        for fp in txt_files:
            yield dataset, fp


def make_chunk_id(rel_path: str, idx: int) -> str:
    return hashlib.md5(f"{rel_path}::{idx}".encode()).hexdigest()


def bm25_tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal RAG — Ingestion  (Dense + BM25)")
    print("=" * 55)
    print(f"  Embed model : {EMBED_MODEL}  (sentence-transformers / {DEVICE.upper()})")
    print(f"  Chunk size  : {CHUNK_SIZE} tokens max  ({CHUNK_OVERLAP} overlap, clause-aware)")
    print()

    # 1. Load embedding model onto GPU
    print(f"Loading {EMBED_MODEL} via sentence-transformers on {DEVICE.upper()}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"  dim = {dim}")
    print()

    # 2. Set up ChromaDB
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped old collection '{COLLECTION_NAME}' (re-ingesting with new chunker)")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Created collection '{COLLECTION_NAME}'")
    print()

    # 3. Chunk all files first (CPU), accumulate everything
    print("Scanning corpus:")
    all_files = list(iter_corpus_files())
    print()

    skipped_files = 0
    all_ids, all_documents, all_metadatas = [], [], []

    print("Chunking all files...")
    for dataset, fp in tqdm(all_files, desc="Chunking", unit="file"):
        rel_path = f"{dataset}/{fp.name}"
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            tqdm.write(f"[WARN] {fp.name}: read error — {e}")
            skipped_files += 1
            continue

        if not text.strip():
            skipped_files += 1
            continue

        chunks = chunk_document(text)
        if not chunks:
            skipped_files += 1
            continue

        ids       = [make_chunk_id(rel_path, i) for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "file_path":   rel_path,
                "dataset":     dataset,
                "chunk_index": i,
                "char_start":  c["char_start"],
                "char_end":    c["char_end"],
                "token_count": c["token_count"],
                "chunk_id":    ids[i],
            }
            for i, c in enumerate(chunks)
        ]
        all_ids.extend(ids)
        all_documents.extend(documents)
        all_metadatas.extend(metadatas)

    total_chunks = len(all_documents)
    print(f"  Total chunks to embed: {total_chunks:,}")
    print()

    # 4. Embed ALL chunks in one big GPU pass
    print(f"Embedding {total_chunks:,} chunks on {DEVICE.upper()}...")
    all_embeddings = embedder.encode(
        all_documents,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()
    print()

    # 5. Upsert to ChromaDB in batches (ChromaDB has a 5461-item limit per call)
    UPSERT_BATCH = 2000
    print(f"Upserting to ChromaDB...")
    for b in tqdm(range(0, total_chunks, UPSERT_BATCH), desc="Upserting", unit="batch"):
        collection.upsert(
            ids=all_ids[b:b+UPSERT_BATCH],
            embeddings=all_embeddings[b:b+UPSERT_BATCH],
            documents=all_documents[b:b+UPSERT_BATCH],
            metadatas=all_metadatas[b:b+UPSERT_BATCH],
        )

    bm25_texts = all_documents
    bm25_metas = all_metadatas

    # 6. Build and save BM25 index
    print()
    print(f"Building BM25 index over {total_chunks:,} chunks...")
    tokenized_corpus = [bm25_tokenize(t) for t in bm25_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    BM25_DIR.mkdir(exist_ok=True)
    with open(BM25_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_DIR / "texts.json", "w", encoding="utf-8") as f:
        json.dump(bm25_texts, f, ensure_ascii=False)
    with open(BM25_DIR / "metas.json", "w", encoding="utf-8") as f:
        json.dump(bm25_metas, f, ensure_ascii=False)
    print(f"BM25 index saved to {BM25_DIR}")

    # 5. Summary
    print()
    print("=" * 55)
    print("  Ingestion Complete")
    print("=" * 55)
    print(f"  Files processed : {len(all_files) - skipped_files:,}")
    print(f"  Files skipped   : {skipped_files}")
    print(f"  Total chunks    : {total_chunks:,}")
    print(f"  ChromaDB count  : {collection.count():,}")
    print(f"  BM25 corpus     : {len(bm25_texts):,} docs")
    print(f"  Dense store     : {CHROMA_DIR}")
    print(f"  BM25 store      : {BM25_DIR}")


if __name__ == "__main__":
    main()
