---
title: FastAPI Docs RAG Assistant
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 📚 FastAPI Documentation RAG Assistant

> A production-quality **Retrieval-Augmented Generation** system that answers FastAPI questions grounded in the official documentation — with streaming answers, switchable retrieval pipelines, conversation history, and a full ablation study.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat&logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-0064A0?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black?style=flat)
[![HF Spaces](https://img.shields.io/badge/HF_Spaces-Live_Demo-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/arishiagarwal2002/devdocs-rag)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

<!-- ![Demo](docs/demo.gif) -->
> 📽 *Demo GIF coming soon — see the [live demo on HF Spaces](https://huggingface.co/spaces/arishiagarwal2002/devdocs-rag)*

## ✨ Features

| Feature | Details |
|---|---|
| **3 retrieval modes** — switchable live | Dense FAISS · Hybrid BM25+Dense (RRF) · + Cross-encoder reranker |
| **Streaming answers** | Token-by-token via Ollama / HF Inference API with a blinking cursor |
| **Conversation history** | All prior turns stay visible; ask natural follow-up questions |
| **Latency display** | Per-answer retrieval time (ms) and generation time (s) |
| **Copy button** | One-click clipboard copy for every answer |
| **REST API backend** | `POST /ask` · `/health` · `/modes` with auto-generated Swagger UI at `/docs` |
| **Evaluation tab** | Interactive Plotly ablation chart + key findings |
| **HF Spaces ready** | Auto-detects HF environment and switches to Mistral-7B via Inference API |
| **153 docs · 2,208 chunks** | Full official FastAPI documentation indexed |

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  FAISS Index (dense)        │  ← BGE-small-en-v1.5 embeddings
│  cosine similarity top-k    │
└────────┬────────────────────┘
         │ (Hybrid mode)  ┌──────────────────────┐
         ├────────────────┤  BM25 Index          │
         │  RRF Fusion    │  keyword search       │
         │                └──────────────────────┘
         │
    ┌────▼────────────────────┐
    │  Cross-encoder Reranker │  (+ Reranker mode)
    │  ms-marco-MiniLM-L-6-v2 │  full question–passage scoring
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  Ollama / OpenAI LLM    │  llama3.2 — streaming supported
    │  Grounded generation    │
    └────┬────────────────────┘
         │
    Cited answer (streamed token-by-token)
```

---

## 📊 Ablation Results

Evaluated on a 12-question hand-curated gold set. Each config changes **one variable** from the baseline.

| Configuration | Recall@5 | MRR | Faithfulness | Correctness |
|---|:---:|:---:|:---:|:---:|
| Baseline (MiniLM, chunk 800, dense) | 0.750 | 0.507 | 0.667 | 0.667 |
| Chunk size 400 *(artefact — see note)* | — | — | 0.667 | 0.750 |
| **★ BGE-small-en-v1.5** | **0.917** | **0.739** | **0.917** | **1.000** |
| Hybrid BM25 + Dense (RRF) | 0.750 | 0.642 | 0.750 | 0.833 |
| Cross-encoder reranker | 0.833 | 0.674 | 0.833 | 0.917 |

**★ Key finding:** Embedding quality dominates — swapping MiniLM → BGE-small raised Correctness from **0.67 → 1.0** (perfect) and Faithfulness from **0.67 → 0.92**.

> *Recall@5 for chunk-400 is invalid: gold passage IDs were generated from 800-char chunks and don't exist after re-chunking. Faithfulness/Correctness are unaffected.*

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve          # keep running in a separate terminal
```

### Install

```bash
git clone https://github.com/YOUR_USERNAME/devdocs-rag.git
cd devdocs-rag
pip install -r requirements.txt
```

### Build the index *(first time only)*

```bash
python -m src.ingest
# Downloads FastAPI docs → chunks → embeds → saves FAISS index
# ~2 min on CPU, ~30 s with GPU
```

### Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Run the REST API (optional)

```bash
uvicorn app.api:app --reload --port 8000
```

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc:       [http://localhost:8000/redoc](http://localhost:8000/redoc)

```bash
# Example request
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a path parameter?", "mode": "reranker"}'
```

### Deploy to HF Spaces

See [DEPLOY_HF_SPACES.md](DEPLOY_HF_SPACES.md) for step-by-step instructions.

---

## 📁 Project Structure

```
devdocs-rag/
├── app/
│   └── streamlit_app.py        # UI: mode switcher, chat history, eval tab, streaming
├── src/
│   ├── config.py               # All tuneable settings in one place
│   ├── ingest.py               # Scrape → chunk → embed → FAISS + JSONL
│   ├── retrieve.py             # Dense / Hybrid (RRF) / + Reranker retrieval
│   ├── generate.py             # Ollama / OpenAI generation + streaming generator
│   ├── rag.py                  # Orchestrator: retriever + generator
│   ├── evaluate.py             # Evaluation harness (Recall@5, MRR, faithfulness, correctness)
│   └── make_eval_set.py        # Gold Q&A set generation via LLM
├── data/
│   ├── chunks.jsonl            # 2,208 text chunks with metadata
│   └── faiss.index             # Binary FAISS flat index
├── experiments/
│   └── results.csv             # Ablation log — every run appended here
├── .streamlit/
│   └── config.toml             # Dark GitHub-style theme
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All knobs live in `src/config.py` — change one variable to run an ablation:

| Setting | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers embedding model |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `USE_HYBRID` | `True` | Enable BM25 + dense fusion via RRF |
| `USE_RERANKER` | `True` | Enable cross-encoder reranker |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `LLM_BACKEND` | `"ollama"` | `"ollama"` or `"openai"` |
| `OLLAMA_MODEL` | `"llama3.2"` | Ollama model name |
| `GEN_TEMPERATURE` | `0.1` | LLM generation temperature |

### Using OpenAI instead of Ollama

```python
# src/config.py
LLM_BACKEND = "openai"
LLM_MODEL   = "gpt-4o-mini"
```

```bash
export OPENAI_API_KEY=sk-...
```

---

## 🧪 Running the Evaluation

```bash
python -m src.evaluate
# Results appended to experiments/results.csv
# Each row = one config × one gold question
```

To add a new ablation: edit `src/config.py`, rebuild the index if needed (`python -m src.ingest`), then re-run `python -m src.evaluate`.

---

## 🗺 Roadmap

- [ ] Deploy to **Hugging Face Spaces** (free, shareable link)
- [ ] Interactive **Plotly** grouped bar chart in Evaluation tab
- [ ] **Source sentence highlighting** — mark the exact sentence used in the answer
- [ ] **Top-k slider** in the UI for live quality vs. speed trade-off
- [ ] **FastAPI backend** (`POST /ask`) decoupled from Streamlit frontend
- [ ] **Query history chips** — click any previous question to re-run it

---

## 📄 License

MIT

---

*Built as part of an MSc Artificial Intelligence & Machine Learning portfolio — Queen Mary University of London, 2025–2026.*
