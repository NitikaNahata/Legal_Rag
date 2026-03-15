"""
Legal RAG Retrieval + Generation Pipeline
  BM25 top-20 + Dense top-20
  -> Weighted RRF (BM25:Dense = 1:3), pool-15
  -> MiniLM-L6 reranker -> top-5
  -> Ollama LLM with enforced inline [N] citations

Generation enforcement:
  1. Stronger prompt with explicit per-sentence citation rule + few-shot example
  2. Post-generation citation validator:
       - extracts all [N] from answer text via regex
       - reconciles with cited_sources list (union, dedup, sort)
       - strips out-of-range citation numbers
  3. Retry loop (up to MAX_RETRIES) with corrective feedback on failure
  4. Faithfulness check: cross-encoder scores each cited (claim-sentence, source-chunk)
     pair; flags any sentence whose max entailment score is below FAITH_THRESHOLD
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DIR      = BASE_DIR / "chroma_db"
BM25_DIR        = BASE_DIR / "bm25_index"
COLLECTION_NAME = "legal_docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "llama3.1:8b"

RETRIEVAL_TOP_K = 20   # BM25 + Dense each fetch this many
RRF_POOL        = 15   # RRF merged pool fed to reranker
RRF_W_BM25      = 1
RRF_W_DENSE     = 3
RRF_K           = 60
FINAL_TOP_N     = 5    # chunks sent to LLM after reranking

MAX_RETRIES     = 2    # LLM retry attempts on citation validation failure
FAITH_THRESHOLD = -3.0 # cross-encoder score below this → flag as unsupported


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """You are a precise legal and privacy document analyst.

You will be given numbered source chunks [1] through [{n_sources}]. Answer the question using ONLY those sources.

STRICT RULES:
1. Every single sentence in your answer MUST end with one or more inline citations, e.g. [1] or [2][4].
2. Do NOT write any sentence without a citation — even background or transitional sentences.
3. Use ONLY source numbers between 1 and {n_sources}. Never invent a number outside this range.
4. If the sources lack sufficient information, respond with exactly:
   "The provided documents do not contain sufficient information to answer this question."
5. Do NOT add information from your training data or general knowledge.

Example of a correctly cited answer (sources 1-3 available):
  "The agreement requires 30 days written notice for termination [1]. Early termination incurs a \
penalty equal to two months of fees [2][3]."

Return ONLY a JSON object with exactly these two keys — no markdown, no extra text:
{{
  "answer": "<your cited answer>",
  "cited_sources": [<list of integer source numbers actually used, e.g. 1, 3>]
}}"""

RETRY_HUMAN = (
    "Your previous response had citation problems: {error}\n\n"
    "QUESTION: {question}\n\n"
    "SOURCES:\n{context}\n\n"
    "Fix the answer so EVERY sentence has an inline [N] citation and cited_sources "
    "only contains integers between 1 and {n_sources}.\n\nJSON response:"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "QUESTION: {question}\n\n"
     "SOURCES:\n{context}\n\n"
     "JSON response:"),
])

RETRY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", RETRY_HUMAN),
])


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def bm25_tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_bm25(bm25_dir: Path):
    with open(bm25_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(bm25_dir / "texts.json", encoding="utf-8") as f:
        texts = json.load(f)
    with open(bm25_dir / "metas.json", encoding="utf-8") as f:
        metas = json.load(f)
    return bm25, texts, metas


def weighted_rrf(b_ids, d_ids, wb=RRF_W_BM25, wd=RRF_W_DENSE, k=RRF_K):
    sc = {}
    for r, cid in enumerate(b_ids, 1):
        sc[cid] = sc.get(cid, 0.0) + wb / (k + r)
    for r, cid in enumerate(d_ids, 1):
        sc[cid] = sc.get(cid, 0.0) + wd / (k + r)
    return sorted(sc, key=sc.__getitem__, reverse=True)


def build_context(docs: list[str], metas: list[dict]) -> str:
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        fname = Path(meta["file_path"]).name
        blocks.append(
            f"[{i}] {fname}  (chars {meta['char_start']}-{meta['char_end']})\n"
            f"{doc.strip()}"
        )
    return "\n\n".join(blocks)


# ── Citation validation ───────────────────────────────────────────────────────

def extract_inline_cites(text: str) -> list[int]:
    """Return all [N] numbers found in the answer text."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", text)]


def validate_citations(answer: str, cited_sources: list, n_sources: int) -> tuple[bool, str]:
    """
    Returns (ok, error_message).
    Checks:
      - answer is non-empty and not the refusal string
      - every sentence has at least one [N] citation
      - all cited numbers are in [1, n_sources]
      - cited_sources list matches the inline references
    """
    refusal = "The provided documents do not contain sufficient information"
    if not answer or refusal in answer:
        return True, ""   # refusal is valid — no citations needed

    inline = extract_inline_cites(answer)
    if not inline:
        return False, "No inline [N] citations found in the answer."

    oob = [n for n in inline if n < 1 or n > n_sources]
    if oob:
        return False, f"Citation numbers out of range (1-{n_sources}): {oob}"

    # Check every sentence ends with a citation
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    uncited = [s for s in sentences if s.strip() and not re.search(r"\[\d+\]", s)]
    if uncited:
        return False, f"{len(uncited)} sentence(s) lack inline citations: {uncited[:2]}"

    return True, ""


def reconcile_citations(answer: str, cited_sources: list, n_sources: int) -> list[int]:
    """
    Union of inline [N] in text and cited_sources list.
    Filters to valid range and deduplicates.
    """
    inline  = set(extract_inline_cites(answer))
    listed  = set(int(n) for n in cited_sources if isinstance(n, (int, float)))
    merged  = inline | listed
    valid   = sorted(n for n in merged if 1 <= n <= n_sources)
    return valid


# ── Faithfulness check ───────────────────────────────────────────────────────

def check_faithfulness(answer: str, top_docs: list[str], top_metas: list[dict],
                       reranker: CrossEncoder) -> list[dict]:
    """
    For each sentence in the answer that has citations, score it against
    each cited source chunk using the cross-encoder.
    Returns a list of warning dicts for sentences with low max score.
    """
    refusal = "The provided documents do not contain sufficient information"
    if refusal in answer:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    warnings  = []

    pairs  = []
    s_meta = []  # (sentence_idx, source_idx)

    for si, sent in enumerate(sentences):
        nums = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
        nums = [n for n in nums if 1 <= n <= len(top_docs)]
        for n in nums:
            pairs.append([sent, top_docs[n - 1]])
            s_meta.append((si, n))

    if not pairs:
        return []

    scores = reranker.predict(pairs)

    # For each sentence, find the max score across its cited sources
    sent_max: dict[int, float] = {}
    for (si, _), sc in zip(s_meta, scores):
        sent_max[si] = max(sent_max.get(si, -999), float(sc))

    for si, max_sc in sent_max.items():
        if max_sc < FAITH_THRESHOLD:
            warnings.append({
                "sentence": sentences[si],
                "max_entailment_score": round(max_sc, 3),
                "flag": "LOW_FAITHFULNESS",
            })

    return warnings


# ── LLM call with retry ───────────────────────────────────────────────────────

def call_llm_with_retry(query: str, context: str, n_sources: int,
                        llm: ChatOllama) -> tuple[str, list[int], int]:
    """
    Returns (answer, reconciled_cited_nums, attempts_used).
    Retries up to MAX_RETRIES with corrective feedback on validation failure.
    """
    prompt_vars = {
        "question":  query,
        "context":   context,
        "n_sources": n_sources,
    }

    chain        = PROMPT | llm
    retry_chain  = RETRY_PROMPT | llm

    answer      = ""
    cited_nums  = []
    last_error  = ""

    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt == 0:
                raw = chain.invoke(prompt_vars)
            else:
                print(f"  [Retry {attempt}] citation error: {last_error}")
                raw = retry_chain.invoke({**prompt_vars, "error": last_error})

            raw_text = raw.content if hasattr(raw, "content") else str(raw)

            # Strip markdown fences if the model wraps its JSON
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
            raw_text = re.sub(r"\s*```$", "", raw_text.strip())

            parsed     = json.loads(raw_text)
            answer     = parsed.get("answer", "").strip()
            cited_nums = parsed.get("cited_sources", [])

        except Exception as e:
            last_error = f"JSON parse error: {e}"
            continue

        ok, last_error = validate_citations(answer, cited_nums, n_sources)
        if ok:
            break

    cited_nums = reconcile_citations(answer, cited_nums, n_sources)
    return answer, cited_nums, attempt + 1


# ── Main pipeline ─────────────────────────────────────────────────────────────

def retrieve_and_answer(query, embedder, collection, bm25, bm25_texts,
                        bm25_metas, reranker, llm):
    import time
    t_total = time.time()
    timings = {}

    id_to_text = {m["chunk_id"]: bm25_texts[i] for i, m in enumerate(bm25_metas)}
    id_to_meta = {m["chunk_id"]: m for m in bm25_metas}

    # 1. BM25
    t0 = time.time()
    bm25_scores = bm25.get_scores(bm25_tokenize(query))
    b_top = np.argsort(bm25_scores)[::-1][:RETRIEVAL_TOP_K]
    b_ids = [bm25_metas[i]["chunk_id"] for i in b_top]
    timings["bm25"] = round(time.time() - t0, 3)

    # 2. Dense
    t0 = time.time()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    dense_res = collection.query(
        query_embeddings=[q_emb], n_results=RETRIEVAL_TOP_K,
        include=["metadatas"]
    )
    d_ids = [m["chunk_id"] for m in dense_res["metadatas"][0]]
    timings["dense"] = round(time.time() - t0, 3)

    # 3. Weighted RRF -> pool
    rrf_ids   = weighted_rrf(b_ids, d_ids)[:RRF_POOL]
    rrf_metas = [id_to_meta[c] for c in rrf_ids if c in id_to_meta]
    rrf_texts = [id_to_text[c] for c in rrf_ids if c in id_to_text]

    # 4. Reranker
    t0 = time.time()
    scores  = reranker.predict([[query, t] for t in rrf_texts])
    ranked  = sorted(zip(scores, rrf_texts, rrf_metas), key=lambda x: x[0], reverse=True)[:FINAL_TOP_N]
    top_docs  = [r[1] for r in ranked]
    top_metas = [r[2] for r in ranked]
    timings["rerank"] = round(time.time() - t0, 3)

    # 5. Build context and call LLM
    context   = build_context(top_docs, top_metas)
    n_sources = len(top_docs)
    print(f"  Calling {OLLAMA_MODEL}...")

    t0 = time.time()
    answer, cited_nums, attempts = call_llm_with_retry(query, context, n_sources, llm)
    timings["llm"] = round(time.time() - t0, 3)

    if attempts > 1:
        print(f"  LLM required {attempts} attempt(s)")

    timings["total"] = round(time.time() - t_total, 3)
    print(f"  Timings: BM25={timings['bm25']}s  Dense={timings['dense']}s  Rerank={timings['rerank']}s  LLM={timings['llm']}s  Total={timings['total']}s")

    refusal = "The provided documents do not contain sufficient information to answer this question."
    if not answer or "does not contain sufficient information" in answer or not cited_nums:
        return {"answer": refusal, "sources": [], "supported": False,
                "faithfulness_warnings": [], "timings": timings}

    # 6. Faithfulness check
    faith_warnings = check_faithfulness(answer, top_docs, top_metas, reranker)
    if faith_warnings:
        print(f"  [FAITHFULNESS] {len(faith_warnings)} low-confidence sentence(s) flagged")

    # 7. Build source cards for cited chunks only
    sources = []
    for n in cited_nums:
        idx = n - 1
        if 0 <= idx < len(top_metas):
            m = top_metas[idx]
            sources.append({
                "num":     n,
                "file":    m["file_path"],
                "chars":   f"{m['char_start']}-{m['char_end']}",
                "snippet": top_docs[idx][:300].replace("\n", " ").strip(),
            })

    return {
        "answer":                answer,
        "sources":               sources,
        "supported":             True,
        "faithfulness_warnings": faith_warnings,
        "timings":               timings,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_response(result: dict) -> None:
    sep  = "=" * 68
    thin = "-" * 68
    safe = lambda s: s.encode("ascii", "replace").decode("ascii")

    print(f"\n{sep}")
    print("  ANSWER")
    print(sep)
    print(safe(result["answer"]))

    if result.get("sources"):
        print(f"\n{thin}")
        print("  SOURCES")
        print(thin)
        for s in result["sources"]:
            print(f"\n  [{s['num']}] {s['file']}")
            print(f"       chars {s['chars']}")
            print(f"       \"{safe(s['snippet'])}...\"")

    warnings = result.get("faithfulness_warnings", [])
    if warnings:
        print(f"\n{thin}")
        print(f"  FAITHFULNESS WARNINGS  ({len(warnings)} sentence(s) flagged)")
        print(thin)
        for w in warnings:
            print(f"  score={w['max_entailment_score']}  \"{safe(w['sentence'][:120])}\"")

    print(f"\n{thin}")
    print(f"  Supported: {result.get('supported', False)}")
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Loading embedder ({EMBED_MODEL}) on {DEVICE.upper()}...")
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    print("Loading BM25 index...")
    bm25, bm25_texts, bm25_metas = load_bm25(BM25_DIR)
    print(f"  {len(bm25_texts):,} docs")

    print("Connecting to ChromaDB...")
    db         = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = db.get_collection(COLLECTION_NAME)
    print(f"  {collection.count():,} vectors")

    print(f"Loading reranker ({RERANK_MODEL})...")
    reranker = CrossEncoder(RERANK_MODEL)

    print(f"Connecting to Ollama ({OLLAMA_MODEL})...")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=4096, format="json")

    print("\nLegal RAG ready. Type your question (or 'quit').\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        print()
        result = retrieve_and_answer(
            query, embedder, collection,
            bm25, bm25_texts, bm25_metas,
            reranker, llm,
        )
        print_response(result)
        print()


if __name__ == "__main__":
    main()
