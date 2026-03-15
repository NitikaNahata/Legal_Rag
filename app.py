"""
Legal RAG — Streamlit Chat UI
Lady Justice themed chatbot interface
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_ollama import ChatOllama

# ── must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="LegalBot — AI Contract Analyst",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from retrieve import (
    load_bm25, retrieve_and_answer,
    EMBED_MODEL, RERANK_MODEL, OLLAMA_MODEL,
    CHROMA_DIR, BM25_DIR, COLLECTION_NAME, DEVICE,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14;
    color: #e8e0d0;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #0a0c10;
    border-right: 1px solid #2a2410;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #2a2410;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: 0.04em;
    margin: 0.3rem 0;
}
.hero-subtitle {
    color: #8a7a5a;
    font-size: 0.9rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Scales SVG ── */
.scales-wrap { margin-bottom: 0.5rem; }

/* ── Chat messages ── */
.msg-user {
    background: #161a22;
    border-left: 3px solid #c9a84c;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    color: #e8e0d0;
}
.msg-bot {
    background: #111318;
    border-left: 3px solid #4a7c6f;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    color: #d8d0c0;
    line-height: 1.65;
}
.msg-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.msg-label-user { color: #c9a84c; }
.msg-label-bot  { color: #4a7c6f; }

/* ── Citation chips ── */
.cite-section {
    margin-top: 0.9rem;
    border-top: 1px solid #1e2230;
    padding-top: 0.7rem;
}
.cite-header {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6a5a3a;
    margin-bottom: 0.5rem;
}
.cite-card {
    background: #0e1018;
    border: 1px solid #2a2820;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}
.cite-num {
    display: inline-block;
    background: #c9a84c;
    color: #0d0f14;
    border-radius: 3px;
    padding: 0 5px;
    font-weight: 700;
    font-size: 0.75rem;
    margin-right: 6px;
}
.cite-file  { color: #8a9ab0; font-size: 0.78rem; }
.cite-chars { color: #5a6070; font-size: 0.74rem; margin-top: 2px; }
.cite-snip  { color: #a0987a; font-style: italic; margin-top: 4px; font-size: 0.8rem; }

/* ── Faith warnings ── */
.faith-warn {
    background: #1a1008;
    border: 1px solid #5a3010;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #c0804a;
}
.faith-header { color: #c05030; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }

/* ── Refusal ── */
.refusal {
    color: #806040;
    font-style: italic;
}

/* ── Supported badge ── */
.badge-yes { color: #4a9a6a; font-size: 0.75rem; }
.badge-no  { color: #9a5030; font-size: 0.75rem; }

/* ── Input box ── */
[data-testid="stChatInput"] textarea {
    background: #111318 !important;
    color: #e8e0d0 !important;
    border: 1px solid #2a2820 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}

/* ── Sidebar ── */
.sidebar-section {
    background: #0d0f14;
    border: 1px solid #1e2010;
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 1rem;
    font-size: 0.83rem;
    color: #7a7060;
}
.sidebar-title {
    color: #c9a84c;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    font-family: 'Playfair Display', serif;
}

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Lady Justice SVG ──────────────────────────────────────────────────────────
JUSTICE_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 220" width="110" height="120">
  <!-- Blindfold -->
  <ellipse cx="100" cy="48" rx="22" ry="8" fill="#c9a84c" opacity="0.9"/>
  <!-- Head -->
  <circle cx="100" cy="52" r="20" fill="#c9a84c" opacity="0.15" stroke="#c9a84c" stroke-width="1.2"/>
  <!-- Crown/laurel ring -->
  <ellipse cx="100" cy="33" rx="18" ry="4" fill="none" stroke="#c9a84c" stroke-width="1.5" opacity="0.7"/>
  <!-- Body/robe -->
  <path d="M78 72 Q100 68 122 72 L130 160 Q100 168 70 160 Z" fill="#c9a84c" opacity="0.08" stroke="#c9a84c" stroke-width="1"/>
  <!-- Arms -->
  <line x1="100" y1="90" x2="38" y2="105" stroke="#c9a84c" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="100" y1="90" x2="162" y2="105" stroke="#c9a84c" stroke-width="1.8" stroke-linecap="round"/>
  <!-- Central pole -->
  <line x1="100" y1="100" x2="100" y2="108" stroke="#c9a84c" stroke-width="2"/>
  <!-- Balance beam -->
  <line x1="38" y1="105" x2="162" y2="105" stroke="#c9a84c" stroke-width="1.5"/>
  <!-- Left pan string -->
  <line x1="50"  y1="105" x2="44"  y2="128" stroke="#c9a84c" stroke-width="1" opacity="0.8"/>
  <line x1="38"  y1="105" x2="44"  y2="128" stroke="#c9a84c" stroke-width="1" opacity="0.8"/>
  <!-- Left pan -->
  <path d="M34 128 Q44 135 54 128" fill="none" stroke="#c9a84c" stroke-width="1.5"/>
  <!-- Right pan string -->
  <line x1="150" y1="105" x2="156" y2="128" stroke="#c9a84c" stroke-width="1" opacity="0.8"/>
  <line x1="162" y1="105" x2="156" y2="128" stroke="#c9a84c" stroke-width="1" opacity="0.8"/>
  <!-- Right pan -->
  <path d="M146 128 Q156 135 166 128" fill="none" stroke="#c9a84c" stroke-width="1.5"/>
  <!-- Sword -->
  <line x1="100" y1="160" x2="100" y2="215" stroke="#c9a84c" stroke-width="2" stroke-linecap="round"/>
  <line x1="90"  y1="175" x2="110" y2="175" stroke="#c9a84c" stroke-width="1.5"/>
  <!-- Base -->
  <rect x="88" y="213" width="24" height="4" rx="2" fill="#c9a84c" opacity="0.6"/>
</svg>
"""

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    embedder  = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    bm25, bm25_texts, bm25_metas = load_bm25(BM25_DIR)
    db         = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = db.get_collection(COLLECTION_NAME)
    reranker   = CrossEncoder(RERANK_MODEL, device=DEVICE)
    llm        = ChatOllama(model=OLLAMA_MODEL, temperature=0, num_ctx=4096, format="json")
    return embedder, bm25, bm25_texts, bm25_metas, collection, reranker, llm


# ── Render a single chat turn ─────────────────────────────────────────────────
def render_turn(turn: dict):
    # User bubble
    st.markdown(f"""
    <div class="msg-user">
      <div class="msg-label msg-label-user">You</div>
      {turn['query']}
    </div>""", unsafe_allow_html=True)

    result  = turn["result"]
    answer  = result.get("answer", "")
    sources = result.get("sources", [])
    faith   = result.get("faithfulness_warnings", [])
    supported = result.get("supported", False)

    refusal_text = "The provided documents do not contain sufficient information"
    is_refusal   = refusal_text in answer

    answer_html = f'<span class="refusal">{answer}</span>' if is_refusal else answer

    badge = '<span class="badge-yes">✦ Grounded in sources</span>' if supported else \
            '<span class="badge-no">✦ Insufficient sources</span>'

    timings = result.get("timings", {})
    timing_html = ""
    if timings:
        timing_html = f"""
        <div style="margin-top:0.6rem; font-size:0.72rem; color:#4a5060; border-top:1px solid #1e2230; padding-top:0.5rem;">
          ⏱ BM25 {timings.get('bm25','-')}s &nbsp;·&nbsp;
          Dense {timings.get('dense','-')}s &nbsp;·&nbsp;
          Rerank {timings.get('rerank','-')}s &nbsp;·&nbsp;
          LLM {timings.get('llm','-')}s &nbsp;·&nbsp;
          <strong style="color:#6a7080;">Total {timings.get('total','-')}s</strong>
        </div>"""

    # Source cards HTML
    sources_html = ""
    if sources:
        cards = ""
        for s in sources:
            fname   = Path(s["file"]).name
            snippet = s["snippet"][:180].replace("<", "&lt;").replace(">", "&gt;")
            cards += f"""
            <div class="cite-card">
              <span class="cite-num">{s['num']}</span>
              <span class="cite-file">{fname}</span>
              <div class="cite-chars">chars {s['chars']}</div>
              <div class="cite-snip">"{snippet}…"</div>
            </div>"""
        sources_html = f"""
        <div class="cite-section">
          <div class="cite-header">⚖ Sources cited</div>
          {cards}
        </div>"""

    # Faithfulness warnings HTML
    faith_html = ""
    if faith:
        items = "".join(
            f"<div>⚠ score {w['max_entailment_score']} — \"{w['sentence'][:120]}\"</div>"
            for w in faith
        )
        faith_html = f"""
        <div class="faith-warn">
          <div class="faith-header">Faithfulness warnings</div>
          {items}
        </div>"""

    st.markdown(f"""
    <div class="msg-bot">
      <div class="msg-label msg-label-bot">LegalBot</div>
      {answer_html}
      {sources_html}
      {faith_html}
      <div style="margin-top:0.6rem">{badge}</div>
      {timing_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(collection_count: int, bm25_count: int):
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
          {JUSTICE_SVG}
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-section">
          <div class="sidebar-title">Corpus</div>
          <div>📚 {collection_count:,} vectors (ChromaDB)</div>
          <div>🔍 {bm25_count:,} docs (BM25)</div>
        </div>
        <div class="sidebar-section">
          <div class="sidebar-title">Pipeline</div>
          <div>Embed: BGE-base-en-v1.5</div>
          <div>Rerank: MiniLM-L6</div>
          <div>LLM: {OLLAMA_MODEL}</div>
          <div>Device: {DEVICE.upper()}</div>
        </div>
        <div class="sidebar-section">
          <div class="sidebar-title">Retrieval</div>
          <div>BM25 top-20 + Dense top-20</div>
          <div>RRF 1:3 (BM25:Dense) → pool-15</div>
          <div>Reranker → top-5 → LLM</div>
        </div>
        <div class="sidebar-section" style="color:#5a4a2a; font-size:0.75rem; font-style:italic;">
          This tool provides AI-assisted analysis of legal documents for research purposes only.
          It does not constitute legal advice.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if "history" not in st.session_state:
        st.session_state.history = []

    # Load models
    with st.spinner("Loading models…"):
        embedder, bm25, bm25_texts, bm25_metas, collection, reranker, llm = load_models()

    render_sidebar(collection.count(), len(bm25_texts))

    # Hero header
    st.markdown(f"""
    <div class="hero">
      <div class="scales-wrap">{JUSTICE_SVG}</div>
      <div class="hero-title">LegalBot</div>
      <div class="hero-subtitle">AI-powered contract & privacy document analyst</div>
    </div>""", unsafe_allow_html=True)

    # Render history
    for turn in st.session_state.history:
        render_turn(turn)

    # Input
    query = st.chat_input("Ask about contracts, NDAs, privacy policies…")
    if query:
        with st.spinner("Retrieving and reasoning…"):
            result = retrieve_and_answer(
                query, embedder, collection,
                bm25, bm25_texts, bm25_metas,
                reranker, llm,
            )
        st.session_state.history.append({"query": query, "result": result})
        st.rerun()


if __name__ == "__main__":
    main()
