"""Dev Docs RAG Assistant — Streamlit UI
Run:  streamlit run app/streamlit_app.py
"""
import html as _html
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, generate
from src.rag import RAG

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dev Docs Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Mode definitions ──────────────────────────────────────────────────────────
MODES = {
    "Dense": {
        "use_hybrid": False, "use_reranker": False,
        "tagline": "Semantic search only",
        "desc": "Embeds the query and finds the closest doc vectors via FAISS cosine similarity.",
        "active":   ["FAISS vector search"],
        "inactive": ["BM25 keyword search", "Cross-encoder reranker"],
        "recall5": 0.750, "mrr": 0.507, "correctness": 0.667,
        "accent": "#58a6ff",
    },
    "Hybrid": {
        "use_hybrid": True, "use_reranker": False,
        "tagline": "Dense + keyword search",
        "desc": "Fuses FAISS with BM25 keyword matching via Reciprocal Rank Fusion — better for exact API terms.",
        "active":   ["FAISS vector search", "BM25 keyword search", "RRF fusion"],
        "inactive": ["Cross-encoder reranker"],
        "recall5": 0.750, "mrr": 0.642, "correctness": 0.833,
        "accent": "#79c0ff",
    },
    "+ Reranker": {
        "use_hybrid": True, "use_reranker": True,
        "tagline": "Full pipeline",
        "desc": "Hybrid retrieval + a cross-encoder that re-scores every candidate on the full question–passage pair.",
        "active":   ["FAISS vector search", "BM25 keyword search", "RRF fusion", "Cross-encoder reranker"],
        "inactive": [],
        "recall5": 0.833, "mrr": 0.674, "correctness": 0.917,
        "accent": "#a78bfa",
    },
}

ABLATION_DF = pd.DataFrame([
    {"Configuration":      "Baseline  (MiniLM · chunk 800 · dense · k=5)",
     "Recall @ 5": 0.750, "MRR": 0.507, "Faithfulness": 0.667, "Correctness": 0.667},
    {"Configuration":      "Chunk size 400  (measurement artefact — see note)",
     "Recall @ 5": "—",   "MRR": "—",   "Faithfulness": 0.667, "Correctness": 0.750},
    {"Configuration":      "★  BGE-small-en-v1.5 embeddings",
     "Recall @ 5": 0.917, "MRR": 0.739, "Faithfulness": 0.917, "Correctness": 1.000},
    {"Configuration":      "Hybrid BM25 + dense  (Reciprocal Rank Fusion)",
     "Recall @ 5": 0.750, "MRR": 0.642, "Faithfulness": 0.750, "Correctness": 0.833},
    {"Configuration":      "Cross-encoder reranker  (ms-marco-MiniLM-L-6-v2)",
     "Recall @ 5": 0.833, "MRR": 0.674, "Faithfulness": 0.833, "Correctness": 0.917},
])

SUGGESTIONS = [
    "How do I define a path parameter?",
    "How does dependency injection work?",
    "How do I add request body validation?",
    "How do I return a custom HTTP error?",
    "Difference between async and sync routes?",
]

# ── Plotly ablation chart data ────────────────────────────────────────────────
CHART_CONFIGS  = ["Baseline", "Chunk-400*", "BGE-small ★", "Hybrid RRF", "+ Reranker"]
CHART_DATA = {
    "Recall@5":    [0.750, None,  0.917, 0.750, 0.833],
    "MRR":         [0.507, None,  0.739, 0.642, 0.674],
    "Faithfulness":[0.667, 0.667, 0.917, 0.750, 0.833],
    "Correctness": [0.667, 0.750, 1.000, 0.833, 0.917],
}
CHART_COLORS = {
    "Recall@5":    "#58a6ff",
    "MRR":         "#79c0ff",
    "Faithfulness":"#a78bfa",
    "Correctness": "#7ee787",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global base bump — every rem unit scales up */
html { font-size: 19px; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

/* ── Page padding ── */
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 100%; }

/* ── Tabs ── */
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.3rem;
    font-weight: 600;
    color: #8b949e;
    padding: 0.6rem 1.5rem;
    border: none;
    background: transparent;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #a78bfa;
    border-bottom: 2px solid #a78bfa;
}
div[data-testid="stTabs"] div[role="tabpanel"] { padding-top: 1.5rem; }

/* ── Panel column (left) ── */
.panel-wrap {
    border-right: 1px solid #21262d;
    padding-right: 1.8rem;
    min-height: 80vh;
}
.panel-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    color: #a78bfa;
    border: 1px solid rgba(167,139,250,0.3);
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    margin-bottom: 1.2rem;
}
.sb-heading {
    font-size: 0.95rem;
    font-weight: 700;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.3rem 0 0.65rem;
}

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-val { font-size: 2.1rem; font-weight: 800; color: #a78bfa; line-height: 1; }
.stat-lbl { font-size: 0.95rem; font-weight: 600; color: #484f58;
            text-transform: uppercase; letter-spacing: 0.08em; text-align: right; line-height: 1.4; }

/* ── Radio — segmented control ── */
div[data-testid="stRadio"] > div { gap: 0.3rem; }
div[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.85rem !important;
    font-size: 1.15rem !important;
    color: #8b949e !important;
    cursor: pointer !important;
    transition: all 0.18s !important;
    text-align: center !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(167,139,250,0.18) !important;
    border-color: rgba(167,139,250,0.55) !important;
    color: #c4b5fd !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] svg { display: none !important; }

/* ── Mode info card ── */
.mode-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin: 0.55rem 0 0.5rem;
}
.mode-tagline { font-size: 1.15rem; font-weight: 700; color: #e6edf3; margin-bottom: 0.35rem; }
.mode-desc    { font-size: 1.08rem; color: #8b949e; line-height: 1.65; margin-bottom: 0.65rem; }
.comp { font-size: 1.05rem; padding: 0.15rem 0; }
.comp.on  { color: #7ee787; }
.comp.off { color: #484f58; }

/* ── Metric bars ── */
.metric-row { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.55rem; }
.m-name { font-size: 1.0rem; color: #8b949e; width: 105px; flex-shrink: 0; }
.m-bar-bg { flex: 1; height: 5px; background: #21262d; border-radius: 99px; overflow: hidden; }
.m-bar {
    height: 100%;
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    border-radius: 99px;
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.m-val { font-size: 1.02rem; font-weight: 700; color: #a78bfa; width: 42px; text-align: right; flex-shrink: 0; }

/* ── Model pills ── */
.model-pill {
    display: block;
    background: rgba(255,255,255,0.05);
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.42rem 0.85rem;
    font-size: 1.08rem;
    font-family: 'Courier New', monospace;
    color: #7ee787;
    margin-bottom: 0.48rem;
}

/* ── Clear history button ── */
div[data-testid="stButton"] button[kind="secondary"] {
    font-size: 1.0rem !important;
    padding: 0.4rem 0.8rem !important;
    color: #484f58 !important;
    border-color: #21262d !important;
}

/* ── Hero ── */
.hero-title {
    font-size: 4.0rem; font-weight: 800; color: #f0f6fc;
    letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 0.85rem;
}
.hero-sub {
    font-size: 1.45rem; color: #8b949e; line-height: 1.78; margin-bottom: 1.8rem;
}

/* ── Compact header when history exists ── */
.compact-header {
    font-size: 1.55rem; font-weight: 700; color: #8b949e;
    letter-spacing: -0.01em; margin-bottom: 0.5rem;
}

/* ── Section labels ── */
.sec-label {
    font-size: 1.0rem; font-weight: 700; color: #484f58;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.55rem;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #21262d; margin: 1.6rem 0; }

/* ── Text input ── */
div[data-testid="stTextInput"] input {
    background: #0d1117 !important; border: 1.5px solid #30363d !important;
    border-radius: 12px !important; color: #e6edf3 !important;
    font-size: 1.45rem !important; padding: 1.1rem 1.4rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.18) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #484f58 !important; }

/* ── Form submit button ── */
div[data-testid="stFormSubmitButton"] button {
    background: rgba(167,139,250,0.15) !important;
    border: 1.5px solid rgba(167,139,250,0.4) !important;
    border-radius: 10px !important;
    color: #c4b5fd !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.18s !important;
    margin-top: 0.4rem !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: rgba(167,139,250,0.28) !important;
    border-color: rgba(167,139,250,0.7) !important;
}

/* ── Suggestion pills ── */
div[data-testid="stHorizontalBlock"] button {
    background: rgba(255,255,255,0.04) !important; border: 1px solid #21262d !important;
    border-radius: 99px !important; color: #8b9cf4 !important;
    font-size: 1.1rem !important; font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important; white-space: normal !important;
    line-height: 1.45 !important; min-height: 0 !important;
    transition: all 0.15s;
}
div[data-testid="stHorizontalBlock"] button:hover {
    background: rgba(167,139,250,0.12) !important;
    border-color: rgba(167,139,250,0.4) !important; color: #c4b5fd !important;
}

/* ── History question bubble ── */
.history-q {
    font-size: 1.38rem;
    font-weight: 700;
    color: #f0f6fc;
    padding: 0.5rem 0 0.3rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.3rem;
    line-height: 1.5;
}
.history-q::before {
    content: "Q";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 2.0rem;
    height: 2.0rem;
    border-radius: 50%;
    background: rgba(167,139,250,0.18);
    color: #a78bfa;
    font-size: 0.95rem;
    font-weight: 800;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

/* ── Latency strip ── */
.latency-strip {
    font-size: 1.05rem;
    color: #484f58;
    padding: 0.45rem 0 1.1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    flex-wrap: wrap;
}
.latency-strip strong { color: #8b949e; }
.latency-dot { color: #30363d; margin: 0 0.2rem; }

/* ── Mode badge ── */
.mode-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 1.05rem; font-weight: 700;
    padding: 0.32rem 0.95rem; border-radius: 99px; border: 1px solid;
    margin-bottom: 0.55rem; letter-spacing: 0.04em;
}

/* ── Answer card ── */
.answer-card {
    background: rgba(167,139,250,0.06);
    border: 1px solid rgba(167,139,250,0.18);
    border-left: 4px solid #a78bfa;
    border-radius: 12px; padding: 1.7rem 2.1rem;
    font-size: 1.45rem; line-height: 1.92; color: #e6edf3;
    margin: 0.4rem 0 0.6rem; white-space: pre-wrap;
}
.cursor { animation: blink 0.9s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }

/* ── Expanders ── */
div[data-testid="stExpander"] {
    background: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 10px !important; margin-bottom: 0.5rem !important;
}
div[data-testid="stExpander"] details summary p {
    font-size: 1.25rem !important; font-weight: 600 !important; color: #8b949e !important;
}
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem !important; color: #8b949e !important; line-height: 1.75 !important;
}

/* ── Eval tab ── */
.eval-hero { font-size: 2.75rem; font-weight: 800; color: #f0f6fc;
             letter-spacing: -0.02em; margin-bottom: 0.4rem; }
.eval-sub  { font-size: 1.35rem; color: #8b949e; line-height: 1.78; margin-bottom: 2rem; }
.finding-card {
    background: rgba(255,255,255,0.03); border: 1px solid #21262d;
    border-radius: 12px; padding: 1.25rem 1.55rem; margin-bottom: 0.75rem;
}
.finding-tag {
    font-size: 1.0rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.4rem;
}
.finding-text { font-size: 1.28rem; color: #e6edf3; line-height: 1.75; }
.metric-def { padding: 0.7rem 0; border-bottom: 1px solid #21262d; }
.metric-def:last-child { border-bottom: none; }
.metric-def-name { font-size: 1.2rem; font-weight: 700; color: #c4b5fd; }
.metric-def-desc { font-size: 1.15rem; color: #8b949e; line-height: 1.65; }

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }
div[data-testid="stDataFrame"] table { font-size: 1.15rem !important; }

/* ── Spinner text ── */
div[data-testid="stSpinner"] p { font-size: 1.1rem !important; color: #8b949e !important; }

/* ── Error card ── */
.error-card {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.3);
    border-left: 4px solid #f85149;
    border-radius: 10px;
    padding: 1.3rem 1.7rem;
    margin: 0.5rem 0 1.2rem;
}
.error-title { font-size: 1.25rem; font-weight: 700; color: #f85149; margin-bottom: 0.45rem; }
.error-body  { font-size: 1.05rem; color: #e6edf3; font-family: 'Courier New', monospace;
               background: rgba(0,0,0,0.3); padding: 0.5rem 0.85rem;
               border-radius: 6px; margin-bottom: 0.65rem; word-break: break-all; }
.error-hint  { font-size: 1.1rem; color: #8b949e; line-height: 1.7; }
.error-hint code { color: #7ee787; background: rgba(255,255,255,0.06);
                   padding: 0.1rem 0.45rem; border-radius: 4px; }

/* ── Copy button (rendered via st.components) ── */
.cp-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid #30363d;
    border-radius: 7px;
    color: #8b9cf4;
    cursor: pointer;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 0.42rem 1.1rem;
    font-family: 'Inter', -apple-system, sans-serif;
    transition: all 0.15s;
    display: inline-block;
    margin-bottom: 0.3rem;
}
.cp-btn:hover {
    background: rgba(167,139,250,0.15);
    color: #c4b5fd;
    border-color: rgba(167,139,250,0.45);
}
</style>
""", unsafe_allow_html=True)


# ── Per-mode cached RAG ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_rag(mode: str) -> RAG:
    m = MODES[mode]
    config.USE_HYBRID   = m["use_hybrid"]
    config.USE_RERANKER = m["use_reranker"]
    return RAG()


# ── Session state ─────────────────────────────────────────────────────────────
if "history"   not in st.session_state:
    st.session_state.history   = []      # list of completed turn dicts
if "pending_q" not in st.session_state:
    st.session_state.pending_q = ""      # set by suggestion pills


# ── Copy-to-clipboard button ──────────────────────────────────────────────────
def _copy_button(text: str, label: str = "📋 Copy answer") -> None:
    """Render a clipboard button that copies `text` when clicked."""
    payload = json.dumps(text)      # safe encoding handles quotes, newlines, etc.
    st.components.v1.html(f"""
    <style>
    .cp-btn {{
        background: rgba(255,255,255,0.05); border: 1px solid #30363d;
        border-radius: 7px; color: #8b9cf4; cursor: pointer;
        font-size: 1.05rem; font-weight: 600; padding: 0.42rem 1.1rem;
        font-family: 'Inter', -apple-system, sans-serif;
        transition: all 0.15s; display: inline-block;
    }}
    .cp-btn:hover {{ background: rgba(167,139,250,0.15); color: #c4b5fd;
                    border-color: rgba(167,139,250,0.45); }}
    </style>
    <button class="cp-btn" id="cb">{label}</button>
    <script>
    document.getElementById('cb').addEventListener('click', function() {{
        const t = {payload};
        if (navigator.clipboard) {{
            navigator.clipboard.writeText(t).then(() => {{
                this.textContent = '✓  Copied!';
                setTimeout(() => this.textContent = '{label}', 2000);
            }});
        }} else {{
            var ta = document.createElement('textarea');
            ta.value = t; document.body.appendChild(ta);
            ta.select(); document.execCommand('copy');
            document.body.removeChild(ta);
            this.textContent = '✓  Copied!';
            setTimeout(() => this.textContent = '{label}', 2000);
        }}
    }});
    </script>
    """, height=50)


# ── History turn renderer ─────────────────────────────────────────────────────
def _render_turn(turn: dict) -> None:
    """Render one completed Q&A turn from history."""
    accent = MODES.get(turn["mode"], MODES["Dense"])["accent"]
    # Question
    st.markdown(
        f'<div class="history-q">{_html.escape(turn["q"])}</div>',
        unsafe_allow_html=True,
    )
    # Answer + Sources side-by-side (same layout as live answer)
    a_col, s_col = st.columns([3, 2], gap="large")
    with a_col:
        st.markdown(
            f'<div class="mode-badge" style="color:{accent};border-color:{accent};">'
            f'⬤&nbsp; {turn["mode"]} mode</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="answer-card" style="border-left-color:{accent};">'
            f'{_html.escape(turn["answer"])}</div>',
            unsafe_allow_html=True,
        )
        # ── Latency strip ──────────────────────────────────────────────────
        st.markdown(
            f'<div class="latency-strip">'
            f'⚡ Retrieved in <strong>{turn["retrieval_ms"]:.0f} ms</strong>'
            f'<span class="latency-dot"> · </span>'
            f'Generated in <strong>{turn["generation_s"]:.1f} s</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _copy_button(turn["answer"])
    with s_col:
        st.markdown('<div class="sec-label">Sources used</div>', unsafe_allow_html=True)
        for h in turn["hits"]:
            with st.expander(f"📄  {h['source']}   ·   score {h['score']:.3f}"):
                st.write(h["text"])


# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_ask, tab_eval = st.tabs(["💬  Ask", "📊  Evaluation"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK
# ════════════════════════════════════════════════════════════════════════════════
with tab_ask:
    panel_col, main_col = st.columns([1, 3.2], gap="large")

    # ── LEFT PANEL ────────────────────────────────────────────────────────────
    with panel_col:
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="panel-badge">RAG · FastAPI Docs</div>', unsafe_allow_html=True)

        # Mode selector
        st.markdown('<div class="sb-heading">Retrieval Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            label="mode", options=list(MODES.keys()),
            index=0, horizontal=True,
            label_visibility="collapsed", key="retrieval_mode",
        )
        m = MODES[mode]

        # Mode info card
        active_html   = "".join(f'<div class="comp on">✓&nbsp;&nbsp;{c}</div>'  for c in m["active"])
        inactive_html = "".join(f'<div class="comp off">—&nbsp;&nbsp;{c}</div>' for c in m["inactive"])
        st.markdown(f"""
        <div class="mode-card">
          <div class="mode-tagline">{m["tagline"]}</div>
          <div class="mode-desc">{m["desc"]}</div>
          {active_html}{inactive_html}
        </div>""", unsafe_allow_html=True)

        # Metric bars
        def _pct(v: float) -> int: return int(v * 100)
        st.markdown(f"""
        <div class="sb-heading" style="margin-top:1rem;">Eval Metrics · ablation study</div>
        <div class="metric-row">
          <span class="m-name">Recall@5</span>
          <div class="m-bar-bg"><div class="m-bar" style="width:{_pct(m['recall5'])}%"></div></div>
          <span class="m-val">{m['recall5']:.3f}</span>
        </div>
        <div class="metric-row">
          <span class="m-name">MRR</span>
          <div class="m-bar-bg"><div class="m-bar" style="width:{_pct(m['mrr'])}%"></div></div>
          <span class="m-val">{m['mrr']:.3f}</span>
        </div>
        <div class="metric-row">
          <span class="m-name">Correctness</span>
          <div class="m-bar-bg"><div class="m-bar" style="width:{_pct(m['correctness'])}%"></div></div>
          <span class="m-val">{m['correctness']:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        st.markdown('<div class="sb-heading">System</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="stat-card">
          <div class="stat-val">153</div><div class="stat-lbl">Docs<br>indexed</div>
        </div>
        <div class="stat-card">
          <div class="stat-val">2,208</div><div class="stat-lbl">Text<br>chunks</div>
        </div>
        """, unsafe_allow_html=True)

        # Clear history
        if st.session_state.history:
            st.markdown('<div class="sb-heading">Session</div>', unsafe_allow_html=True)
            hist_count = len(st.session_state.history)
            if st.button(f"🗑  Clear {hist_count} turn{'s' if hist_count > 1 else ''}",
                         use_container_width=True):
                st.session_state.history = []
                st.rerun()

        # Models
        st.markdown('<div class="sb-heading">Models</div>', unsafe_allow_html=True)
        st.markdown("""
        <span class="model-pill">BGE-small-en-v1.5</span>
        <span class="model-pill">ms-marco-MiniLM reranker</span>
        <span class="model-pill">llama3.2 · Ollama</span>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── MAIN CONTENT ──────────────────────────────────────────────────────────
    with main_col:

        # Hero (empty state) vs compact header (conversation in progress)
        if not st.session_state.history:
            st.markdown('<div class="hero-title">Ask the docs.<br>Get cited answers.</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="hero-sub">Ask any FastAPI question. Answers are grounded '
                'strictly in the official documentation — every claim links back to '
                'the source passage.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="compact-header">Dev Docs Assistant</div>',
                        unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Conversation history ───────────────────────────────────────────
        for turn in st.session_state.history:
            _render_turn(turn)

        # Reserve a container for the current (live) answer — renders HERE,
        # above the search form, so the answer appears right after history.
        live_container = st.container()

        # ── Search form ───────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Your question</div>', unsafe_allow_html=True)
        with st.form("question_form", clear_on_submit=True):
            q_input = st.text_input(
                label="question",
                label_visibility="collapsed",
                placeholder="e.g.  How do I define a path parameter?",
            )
            submitted = st.form_submit_button("Ask  →", use_container_width=True)

        # ── Suggestion pills (shown only on empty state) ───────────────────
        if not st.session_state.history:
            st.markdown('<div class="sec-label" style="margin-top:1rem;">Try an example</div>',
                        unsafe_allow_html=True)
            pill_cols = st.columns(len(SUGGESTIONS))
            for col, sug in zip(pill_cols, SUGGESTIONS):
                with col:
                    if st.button(sug, use_container_width=True):
                        st.session_state.pending_q = sug
                        st.rerun()

        # ── Resolve which question to answer ──────────────────────────────
        q: str = ""
        if submitted and q_input.strip():
            q = q_input.strip()
        elif st.session_state.pending_q:
            q = st.session_state.pending_q
            st.session_state.pending_q = ""

        # ── Answer the question ───────────────────────────────────────────
        if q:
            accent = MODES[mode]["accent"]

            with live_container:
                # Question bubble
                st.markdown(
                    f'<div class="history-q">{_html.escape(q)}</div>',
                    unsafe_allow_html=True,
                )

                try:
                    # ── Retrieval ─────────────────────────────────────────
                    is_first = not st.session_state.get(f"loaded_{mode}")
                    with st.spinner(f"Initialising {mode} pipeline…" if is_first
                                    else "Searching documentation…"):
                        rag = get_rag(mode)
                        st.session_state[f"loaded_{mode}"] = True
                        t0 = time.perf_counter()
                        hits = rag.retriever.search(q)
                        retrieval_ms = (time.perf_counter() - t0) * 1000

                    # ── Answer + Sources columns ───────────────────────────
                    a_col, s_col = st.columns([3, 2], gap="large")

                    with a_col:
                        st.markdown(
                            f'<div class="mode-badge" style="color:{accent};border-color:{accent};">'
                            f'⬤&nbsp; {mode} mode</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<div class="sec-label">Answer</div>', unsafe_allow_html=True)

                        # Stream tokens into the placeholder
                        ans_placeholder = st.empty()
                        answer_text = ""
                        t1 = time.perf_counter()
                        for token in generate.stream_answer(q, [h["text"] for h in hits]):
                            answer_text += token
                            safe = _html.escape(answer_text)
                            ans_placeholder.markdown(
                                f'<div class="answer-card" style="border-left-color:{accent};">'
                                f'{safe}<span class="cursor">▌</span></div>',
                                unsafe_allow_html=True,
                            )
                        generation_s = time.perf_counter() - t1

                        # Finalise — remove blinking cursor
                        ans_placeholder.markdown(
                            f'<div class="answer-card" style="border-left-color:{accent};">'
                            f'{_html.escape(answer_text)}</div>',
                            unsafe_allow_html=True,
                        )

                        # ── Latency strip ──────────────────────────────────
                        st.markdown(
                            f'<div class="latency-strip">'
                            f'⚡ Retrieved in <strong>{retrieval_ms:.0f} ms</strong>'
                            f'<span class="latency-dot"> · </span>'
                            f'Generated in <strong>{generation_s:.1f} s</strong>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        _copy_button(answer_text)

                    with s_col:
                        st.markdown('<div class="sec-label">Sources used</div>', unsafe_allow_html=True)
                        for h in hits:
                            with st.expander(f"📄  {h['source']}   ·   score {h['score']:.3f}"):
                                st.write(h["text"])

                    # Persist completed turn into history
                    st.session_state.history.append({
                        "q":            q,
                        "answer":       answer_text,
                        "hits":         hits,
                        "mode":         mode,
                        "retrieval_ms": retrieval_ms,
                        "generation_s": generation_s,
                    })

                except RuntimeError as exc:
                    err = str(exc)
                    is_ollama = "ollama" in err.lower() or "connection" in err.lower()
                    hint = (
                        f"Make sure Ollama is running: <code>ollama serve</code><br>"
                        f"And the model is pulled: <code>ollama pull {config.OLLAMA_MODEL}</code>"
                        if is_ollama else
                        "Check the terminal for the full traceback."
                    )
                    st.markdown(f"""
                    <div class="error-card">
                      <div class="error-title">⚠ Generation failed</div>
                      <div class="error-body">{_html.escape(err)}</div>
                      <div class="error-hint">{hint}</div>
                    </div>""", unsafe_allow_html=True)

                except Exception as exc:
                    st.markdown(f"""
                    <div class="error-card">
                      <div class="error-title">⚠ Unexpected error</div>
                      <div class="error-body">{_html.escape(str(exc))}</div>
                      <div class="error-hint">Check the terminal for details.</div>
                    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ════════════════════════════════════════════════════════════════════════════════
with tab_eval:
    ev_left, ev_right = st.columns([2.2, 1], gap="large")

    with ev_left:
        st.markdown('<div class="eval-hero">Ablation Study</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="eval-sub">Each row changes exactly one variable from the '
            'baseline and re-evaluates on the same 12-question hand-curated gold set. '
            'Metrics are averaged across all questions. '
            'LLM-judged faithfulness and correctness use llama3.2 as judge.</div>',
            unsafe_allow_html=True)

        # ── Plotly grouped bar chart ───────────────────────────────────────
        st.markdown('<div class="sec-label">Results</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for metric, values in CHART_DATA.items():
            y_vals = [v if v is not None else 0 for v in values]
            labels = [f"{v:.3f}" if v is not None else "N/A" for v in values]
            fig.add_trace(go.Bar(
                name=metric,
                x=CHART_CONFIGS,
                y=y_vals,
                text=labels,
                textposition="outside",
                textfont=dict(size=13, color="#8b949e"),
                marker_color=CHART_COLORS[metric],
                marker_line_width=0,
                opacity=0.88,
            ))
        fig.update_layout(
            barmode="group",
            bargap=0.22,
            bargroupgap=0.04,
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="Inter, -apple-system, sans-serif", size=13),
            xaxis=dict(
                gridcolor="#21262d",
                linecolor="#30363d",
                tickfont=dict(size=13, color="#8b949e"),
                fixedrange=True,
            ),
            yaxis=dict(
                gridcolor="#21262d",
                linecolor="#30363d",
                range=[0, 1.22],
                tickformat=".2f",
                tickfont=dict(size=13, color="#8b949e"),
                title=dict(text="Score", font=dict(size=13, color="#484f58")),
                fixedrange=True,
            ),
            legend=dict(
                bgcolor="rgba(22,27,34,0.95)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(size=13),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=10, r=10, t=45, b=10),
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3", size=13),
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Detail table below chart ───────────────────────────────────────
        with st.expander("📋  Full results table", expanded=False):
            st.dataframe(
                ABLATION_DF,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Configuration": st.column_config.TextColumn(width="large"),
                    "Recall @ 5":    st.column_config.NumberColumn(format="%.3f"),
                    "MRR":           st.column_config.NumberColumn(format="%.3f"),
                    "Faithfulness":  st.column_config.NumberColumn(format="%.3f"),
                    "Correctness":   st.column_config.NumberColumn(format="%.3f"),
                },
            )
        st.markdown(
            '<div style="font-size:1.08rem;color:#484f58;margin-top:0.3rem;">'
            '★ Best configuration overall.&nbsp; · &nbsp;'
            '* Recall@5 invalid cross-chunk-size: gold IDs generated from 800-char chunks '
            'no longer exist after re-chunking.</div>',
            unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Key findings</div>', unsafe_allow_html=True)

        findings = [
            ("#a78bfa", "Embedding quality dominates",
             "Swapping MiniLM for BGE-small — the single highest-leverage change — raised "
             "Recall@5 from 0.75 → 0.92, faithfulness from 0.67 → 0.92, and correctness "
             "to a perfect 1.0. Better representations matter more than retrieval strategy."),
            ("#79c0ff", "Hybrid RRF rescues keyword queries",
             "BM25 + dense fusion improved MRR by +0.14 and correctness by +0.17 without "
             "changing which chunks were retrieved — BM25 catches exact function-name matches "
             "that pure dense similarity misses."),
            ("#7ee787", "Cross-encoder adds consistent gains",
             "Reranking delivered steady improvements across all four metrics (Recall@5 +0.08, "
             "MRR +0.17, Correctness +0.25) at the cost of extra per-query latency — a "
             "quality-vs-speed trade-off users can toggle live."),
            ("#f0883e", "Chunk-size comparison is a measurement artefact",
             "Recall@k collapsed with chunk-400 because gold chunk IDs were generated from "
             "800-char chunks and no longer exist after re-chunking. Faithfulness and "
             "correctness were unaffected — confirming the pipeline works."),
        ]
        for ac, title, body in findings:
            st.markdown(f"""
            <div class="finding-card">
              <div class="finding-tag" style="color:{ac};">{title}</div>
              <div class="finding-text">{body}</div>
            </div>""", unsafe_allow_html=True)

    with ev_right:
        st.markdown(
            '<div class="eval-hero" style="font-size:1.9rem;">Metric Definitions</div>',
            unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-def">
          <div class="metric-def-name">Recall @ 5</div>
          <div class="metric-def-desc">Fraction of questions where the gold passage
          appears in the top-5 retrieved chunks. Measures retrieval coverage.</div>
        </div>
        <div class="metric-def">
          <div class="metric-def-name">MRR  (Mean Reciprocal Rank)</div>
          <div class="metric-def-desc">Average of 1/rank of the first correct chunk.
          Rewards finding the right passage higher up the list.</div>
        </div>
        <div class="metric-def">
          <div class="metric-def-name">Faithfulness</div>
          <div class="metric-def-desc">LLM-judged score (0–1): is every claim in the
          answer supported by the retrieved context? Measures hallucination resistance.</div>
        </div>
        <div class="metric-def">
          <div class="metric-def-name">Correctness</div>
          <div class="metric-def-desc">LLM-judged score (0–1): does the answer match the
          reference answer? Measures end-to-end answer quality.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sb-heading">Gold Set Methodology</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:1.15rem;color:#8b949e;line-height:1.72;">
        24 Q&A pairs were generated from the corpus using llama3.2,
        then <strong style="color:#e6edf3;">manually reviewed</strong> — deleting
        release-note trivia, sponsor additions and version questions — leaving
        <strong style="color:#e6edf3;">12 high-quality developer questions</strong>
        that reflect real FastAPI usage.<br><br>
        Each ablation re-runs the full evaluation loop; results are appended to
        <code style="color:#7ee787;font-size:1.08rem;">experiments/results.csv</code>
        for reproducibility.
        </div>
        """, unsafe_allow_html=True)
