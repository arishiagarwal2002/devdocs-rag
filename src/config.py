"""Central configuration.

Everything tunable lives here so your ablation study is just a matter of
changing values and re-running. That single design choice is what makes the
evaluation reproducible.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # put the docs (.md / .txt / .html) here
INDEX_DIR = ROOT / "data" / "index"      # built FAISS index + chunk metadata
EVAL_DIR = ROOT / "data" / "eval"        # gold_qa.jsonl lives here
RESULTS_DIR = ROOT / "experiments"       # ablation result CSVs

# --- Corpus ---
# Swap the whole project to a different library by changing this label and
# dropping that library's docs into data/raw/. Nothing else needs to change.
CORPUS_NAME = "fastapi"

# --- Chunking (ablation variable #1) ---
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 120

# --- Embeddings (ablation variable #2) ---
# Small, free, local. Alternatives to try: "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval (ablation variables #3 and #4) ---
TOP_K = 5
USE_HYBRID = False
USE_RERANKER = False      # add a cross-encoder reranker (see retrieve.py)

# --- Generation ---
# LLM_BACKEND: "openai"  → needs OPENAI_API_KEY env var
#              "ollama"  → free local inference, needs `ollama serve` running
#                          and the model pulled: `ollama pull llama3.2`
#              "hf"      → HuggingFace Inference API (used on HF Spaces)
#                          needs HF_TOKEN env var set as a Space secret
import os as _os
_IS_HF_SPACES = _os.environ.get("SPACE_ID") is not None   # auto-detected on HF Spaces
LLM_BACKEND = "hf" if _IS_HF_SPACES else "ollama"

LLM_MODEL    = "gpt-4o-mini"                                    # openai backend
OLLAMA_MODEL = "llama3.2"                                        # ollama backend
HF_MODEL     = "mistralai/Mistral-7B-Instruct-v0.3"             # hf backend
MAX_NEW_TOKENS = 512
GEN_TEMPERATURE = 0.0
