"""Ingestion: load docs -> chunk -> embed -> build FAISS index.

Run once (or whenever you change chunking/embedding settings):
    python -m src.ingest
"""
import json
import pickle
import re
from dataclasses import dataclass, asdict

import numpy as np

from src import config


@dataclass
class Chunk:
    chunk_id: str
    source: str          # filename the chunk came from
    text: str


def load_documents() -> list[tuple[str, str]]:
    """Return list of (filename, raw_text). Handles .md/.txt/.html crudely."""
    docs = []
    files = list(config.RAW_DIR.glob("**/*"))
    files = [f for f in files if f.suffix.lower() in {".md", ".txt", ".html", ".htm"}]
    if not files:
        raise FileNotFoundError(
            f"No docs found in {config.RAW_DIR}. "
            "Download the library's docs there first (see README, step 1)."
        )
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        if f.suffix.lower() in {".html", ".htm"}:
            text = re.sub(r"<[^>]+>", " ", text)        # strip tags (good enough to start)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        docs.append((str(f.relative_to(config.RAW_DIR)), text))
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-window chunker. TODO (ablation): try a markdown-header
    aware splitter and compare retrieval recall."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_chunks() -> list[Chunk]:
    chunks = []
    for fname, text in load_documents():
        for i, piece in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(Chunk(chunk_id=f"{fname}::{i}", source=fname, text=piece))
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vecs, dtype="float32")


def main():
    import faiss

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    print(f"{len(chunks)} chunks from corpus '{config.CORPUS_NAME}'")

    vecs = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])   # inner product == cosine (vectors normalized)
    index.add(vecs)

    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    with open(config.INDEX_DIR / "chunks.pkl", "wb") as fh:
        pickle.dump([asdict(c) for c in chunks], fh)
    # human-readable copy for debugging
    with open(config.INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")
    print(f"Index + metadata written to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
