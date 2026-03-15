"""
Retrieval Evaluation — Recall@10, Precision@10, MRR
Conditions: BM25 | Dense | Hybrid (Dense:BM25 = 2:1 RRF) | Reranked
Dataset: 50 questions sampled consistently from cuad.json, privacy_qa.json, contractnli.json
"""

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = BASE_DIR / "chroma_db"
BM25_DIR        = BASE_DIR / "bm25_index"
COLLECTION_NAME = "legal_docs"

EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K        = 10    # Recall@10, Precision@10, MRR@10
RERANK_POOL  = 20    # candidates fed to reranker before top-10 cut
TOTAL_EVAL   = 50    # total questions across all datasets
SEED         = 42

RRF_K        = 60
RRF_W_BM25   = 1
RRF_W_DENSE  = 2    # dense gets 2x weight per spec

# ── Helpers ───────────────────────────────────────────────────────────────────

def bm25_tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_bm25():
    with open(BM25_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DIR / "texts.json", encoding="utf-8") as f:
        texts = json.load(f)
    with open(BM25_DIR / "metas.json", encoding="utf-8") as f:
        metas = json.load(f)
    return bm25, texts, metas


def load_eval_questions(n_total: int, seed: int) -> list[dict]:
    """
    Sample n_total questions proportionally from cuad, privacy_qa, contractnli.
    Each question: {query, relevant_chunks: [{file_path, span_start, span_end}]}
    A retrieved chunk is a hit if same file AND its [char_start, char_end] overlaps the gold span.
    """
    sources = {
        "cuad":         "cuad.json",
        "privacy_qa":   "privacy_qa.json",
        "contractnli":  "contractnli.json",
    }
    all_qs = []
    for dataset, fname in sources.items():
        raw = json.load(open(BASE_DIR / fname, encoding="utf-8"))["tests"]
        for item in raw:
            # Only keep questions that have at least one snippet in our corpus
            relevant = [
                {
                    "file_path":   s["file_path"],
                    "span_start":  s["span"][0],
                    "span_end":    s["span"][1],
                }
                for s in item["snippets"]
                if "span" in s and len(s["span"]) == 2
            ]
            if relevant:
                all_qs.append({
                    "query":    item["query"],
                    "relevant": relevant,
                    "dataset":  dataset,
                })

    rng = random.Random(seed)
    rng.shuffle(all_qs)
    sampled = all_qs[:n_total]
    print(f"Sampled {len(sampled)} questions (seed={seed})")
    counts = {}
    for q in sampled:
        counts[q["dataset"]] = counts.get(q["dataset"], 0) + 1
    for ds, c in sorted(counts.items()):
        print(f"  {ds}: {c}")
    return sampled


def is_hit(retrieved_meta: dict, relevant: list[dict]) -> bool:
    """
    A retrieved chunk is a hit if:
      - same file_path AND
      - the chunk's char span overlaps with any gold span
    """
    r_file  = retrieved_meta["file_path"]
    r_start = retrieved_meta["char_start"]
    r_end   = retrieved_meta["char_end"]
    for gold in relevant:
        if gold["file_path"] != r_file:
            continue
        # Overlap: not (r_end <= g_start or r_start >= g_end)
        if not (r_end <= gold["span_start"] or r_start >= gold["span_end"]):
            return True
    return False


def compute_metrics(results_per_query: list[list[dict]],
                    relevants: list[list[dict]]) -> dict:
    """
    results_per_query: list of lists of metadata dicts (ranked)
    relevants:         list of gold relevant lists
    Returns Recall@K, Precision@K, MRR@K
    """
    recalls, precisions, rrs = [], [], []

    for ranked_metas, relevant in zip(results_per_query, relevants):
        top_k = ranked_metas[:TOP_K]
        hits  = [is_hit(m, relevant) for m in top_k]

        # Recall@K = hits found / total gold (capped at 1)
        n_gold   = len(relevant)
        n_hits   = sum(hits)
        recalls.append(min(n_hits / n_gold, 1.0) if n_gold > 0 else 0.0)

        # Precision@K = hits / K
        precisions.append(n_hits / TOP_K)

        # MRR@K = 1/rank of first hit
        rr = 0.0
        for rank, h in enumerate(hits, 1):
            if h:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    return {
        f"Recall@{TOP_K}":    round(np.mean(recalls), 4),
        f"Precision@{TOP_K}": round(np.mean(precisions), 4),
        f"MRR@{TOP_K}":       round(np.mean(rrs), 4),
    }


def weighted_rrf(b_ids, d_ids, wb=RRF_W_BM25, wd=RRF_W_DENSE, k=RRF_K):
    sc = {}
    for r, cid in enumerate(b_ids, 1):
        sc[cid] = sc.get(cid, 0.0) + wb / (k + r)
    for r, cid in enumerate(d_ids, 1):
        sc[cid] = sc.get(cid, 0.0) + wd / (k + r)
    return sorted(sc, key=sc.__getitem__, reverse=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Legal RAG — Retrieval Evaluation")
    print("=" * 60)
    print(f"  Top-K={TOP_K}  |  Rerank pool={RERANK_POOL}  |  N={TOTAL_EVAL}  |  seed={SEED}")
    print(f"  RRF weights: BM25={RRF_W_BM25}  Dense={RRF_W_DENSE}")
    print()

    # ── Load resources ────────────────────────────────────────────────────────
    print(f"Loading embedder ({EMBED_MODEL}) on {DEVICE.upper()}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    print("Loading BM25 index...")
    bm25, bm25_texts, bm25_metas = load_bm25()
    id_to_meta = {m["chunk_id"]: m for m in bm25_metas}
    print(f"  {len(bm25_texts):,} docs")

    print("Connecting to ChromaDB...")
    db         = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = db.get_collection(COLLECTION_NAME)
    print(f"  {collection.count():,} vectors")

    print(f"Loading reranker ({RERANK_MODEL})...")
    reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
    print()

    # ── Sample questions ──────────────────────────────────────────────────────
    print("Loading eval questions...")
    questions = load_eval_questions(TOTAL_EVAL, SEED)
    queries   = [q["query"]    for q in questions]
    relevants = [q["relevant"] for q in questions]
    print()

    # ── Pre-encode all queries (one GPU pass) ─────────────────────────────────
    print(f"Encoding {len(queries)} queries on {DEVICE.upper()}...")
    t0 = time.time()
    q_embs = embedder.encode(queries, normalize_embeddings=True,
                             batch_size=64, show_progress_bar=False)
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    # ── Per-condition result lists ─────────────────────────────────────────────
    bm25_results    = []
    dense_results   = []
    hybrid_results  = []
    rerank_results  = []

    print("Running retrieval for all queries...")
    for i, (query, q_emb) in enumerate(tqdm(zip(queries, q_embs),
                                             total=len(queries), desc="Queries")):
        # ── BM25 ──────────────────────────────────────────────────────────────
        scores = bm25.get_scores(bm25_tokenize(query))
        b_top  = np.argsort(scores)[::-1][:RERANK_POOL]
        b_ids  = [bm25_metas[j]["chunk_id"] for j in b_top]
        bm25_results.append([bm25_metas[j] for j in b_top[:TOP_K]])

        # ── Dense ─────────────────────────────────────────────────────────────
        dense_res = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=RERANK_POOL,
            include=["metadatas"],
        )
        d_metas = dense_res["metadatas"][0]
        d_ids   = [m["chunk_id"] for m in d_metas]
        dense_results.append(d_metas[:TOP_K])

        # ── Hybrid (weighted RRF 2:1 dense:bm25) ─────────────────────────────
        rrf_ids   = weighted_rrf(b_ids, d_ids)[:RERANK_POOL]
        rrf_metas = [id_to_meta[c] for c in rrf_ids if c in id_to_meta]
        hybrid_results.append(rrf_metas[:TOP_K])

        # ── Reranked ──────────────────────────────────────────────────────────
        rrf_texts = [bm25_texts[bm25_metas.index(m)]
                     if m in bm25_metas else "" for m in rrf_metas]
        # faster: use id_to_text
        id_to_text = {m["chunk_id"]: bm25_texts[j] for j, m in enumerate(bm25_metas)}
        rrf_texts  = [id_to_text.get(m["chunk_id"], "") for m in rrf_metas]

        rerank_scores = reranker.predict([[query, t] for t in rrf_texts])
        ranked = sorted(zip(rerank_scores, rrf_metas),
                        key=lambda x: x[0], reverse=True)
        rerank_results.append([m for _, m in ranked[:TOP_K]])

    # ── Compute metrics ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  RESULTS  (N={TOTAL_EVAL}, K={TOP_K})")
    print("=" * 60)

    conditions = [
        ("BM25 only",           bm25_results),
        ("Dense only",          dense_results),
        ("Hybrid (2:1 D:B RRF)", hybrid_results),
        ("Hybrid + Reranker",   rerank_results),
    ]

    all_metrics = {}
    header = f"{'Condition':<25}  {'Recall@'+str(TOP_K):<12}  {'Precision@'+str(TOP_K):<14}  {'MRR@'+str(TOP_K)}"
    print(header)
    print("-" * len(header))

    for name, res in conditions:
        m = compute_metrics(res, relevants)
        all_metrics[name] = m
        print(f"  {name:<23}  {m[f'Recall@{TOP_K}']:<12.4f}  {m[f'Precision@{TOP_K}']:<14.4f}  {m[f'MRR@{TOP_K}']:.4f}")

    print()

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    datasets = list(set(q["dataset"] for q in questions))
    for ds in sorted(datasets):
        idxs = [i for i, q in enumerate(questions) if q["dataset"] == ds]
        print(f"  -- {ds} ({len(idxs)} questions) --")
        hdr2 = f"  {'Condition':<25}  {'Recall@'+str(TOP_K):<12}  {'Precision@'+str(TOP_K):<14}  {'MRR@'+str(TOP_K)}"
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for name, res in conditions:
            sub_res = [res[i] for i in idxs]
            sub_rel = [relevants[i] for i in idxs]
            m = compute_metrics(sub_res, sub_rel)
            print(f"  {name:<25}  {m[f'Recall@{TOP_K}']:<12.4f}  {m[f'Precision@{TOP_K}']:<14.4f}  {m[f'MRR@{TOP_K}']:.4f}")
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "config": {
            "top_k": TOP_K, "rerank_pool": RERANK_POOL,
            "n_questions": TOTAL_EVAL, "seed": SEED,
            "rrf_w_bm25": RRF_W_BM25, "rrf_w_dense": RRF_W_DENSE,
        },
        "overall": all_metrics,
        "per_dataset": {},
    }
    for ds in sorted(datasets):
        idxs = [i for i, q in enumerate(questions) if q["dataset"] == ds]
        out["per_dataset"][ds] = {}
        for name, res in conditions:
            sub_res = [res[i] for i in idxs]
            sub_rel = [relevants[i] for i in idxs]
            out["per_dataset"][ds][name] = compute_metrics(sub_res, sub_rel)

    with open(BASE_DIR / "eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Results saved to eval_results.json")


if __name__ == "__main__":
    main()
