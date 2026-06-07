# Deploying to Hugging Face Spaces

This gives you a **free public URL** to share on your resume and LinkedIn.
The app runs entirely on HF infrastructure — no local Ollama needed.

---

## How it works on Spaces

| Local dev | HF Spaces |
|---|---|
| Ollama (`llama3.2`) | HF Inference API (`Mistral-7B`) |
| `LLM_BACKEND = "ollama"` | `LLM_BACKEND = "hf"` (auto-detected) |
| FAISS index on disk | FAISS index committed in the repo |

`src/config.py` detects `SPACE_ID` (set automatically by HF) and switches backends.

---

## Step-by-step

### 1. Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Owner**: your username
   - **Space name**: `devdocs-rag` (or anything you like)
   - **SDK**: **Streamlit**
   - **Visibility**: Public
3. Click **Create Space**

### 2. Get a HF token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → **Read** permissions → copy it
3. In your Space → **Settings** → **Repository secrets**
4. Add secret: **Name** = `HF_TOKEN`, **Value** = the token you copied

### 3. Make sure data files are not gitignored

```bash
# In the project root, check .gitignore
# Make sure data/index/ is NOT ignored — the FAISS index must be committed
```

If `data/index/` is gitignored, remove that line.

### 4. Push the project

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/arishiagarwal2002/devdocs-rag
cd devdocs-rag

# Copy project files into the Space repo
# (or add the Space as a second remote to your existing repo)
git remote add space https://huggingface.co/spaces/arishiagarwal2002/devdocs-rag

# The Space needs the app at a specific path — set it in README metadata
# Add this YAML block at the very TOP of README.md (above everything else):
```

```yaml
---
title: FastAPI Docs RAG Assistant
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
```

```bash
# Push to the Space
git push space main
```

HF Spaces will install `requirements.txt`, then run:
```
streamlit run app/streamlit_app.py
```

### 5. Check the logs

In your Space → **Logs** tab. Common issues:

| Error | Fix |
|---|---|
| `ModuleNotFoundError: huggingface_hub` | `requirements.txt` must include `huggingface_hub>=0.23` ✅ already done |
| `401 Unauthorized` | Check your `HF_TOKEN` secret is set correctly |
| `FAISS index not found` | Commit `data/index/faiss.index` and `data/index/chunks.jsonl` to the repo |
| Out of memory | Switch to a smaller model: set `HF_MODEL = "microsoft/Phi-3-mini-4k-instruct"` in `config.py` |

---

## Updating the live Space

Any push to the Space repo triggers a rebuild:

```bash
git add .
git commit -m "update"
git push space main
```

---

## Adding the link to your resume

Once live, your Space URL is:
```
https://arishiagarwal2002-devdocs-rag.hf.space
```

Add it to:
- GitHub README badge: `[![Demo](https://img.shields.io/badge/Live_Demo-HF_Spaces-yellow)](YOUR_URL)`
- Resume: "Live demo: YOUR_URL"
- LinkedIn project section
