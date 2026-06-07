"""Build faiss.index + chunks.pkl from the committed chunks.jsonl.

Used in the Dockerfile so binary files never enter git, while the text
JSONL (the ground-truth chunk store) stays version-controlled.

Run:
    python scripts/build_index_from_jsonl.py
"""
import json
import pickle
import sys
from pathlib import Path

# Make sure project root is on the path when called from Dockerfile WORKDIR
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402 — needs sys.path set first


def main() -> None:
    jsonl_path = config.INDEX_DIR / "chunks.jsonl"
    if not jsonl_path.exists():
        sys.exit(f"ERROR: {jsonl_path} not found. "
                 "Make sure chunks.jsonl is committed to the repository.")

    print(f"Loading chunks from {jsonl_path} …")
    chunks = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    print(f"  {len(chunks)} chunks loaded")

    print(f"Embedding with {config.EMBEDDING_MODEL} …")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=True, batch_size=64)
    vecs = np.asarray(vecs, dtype="float32")
    print(f"  Embedding shape: {vecs.shape}")

    print("Building FAISS index …")
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])   # cosine similarity (vecs are L2-normalised)
    index.add(vecs)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    with open(config.INDEX_DIR / "chunks.pkl", "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"Done — index + pkl written to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
