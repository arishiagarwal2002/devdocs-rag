"""Retrieval. Dense baseline now; hybrid + reranker are your ablation upgrades.

A Retriever returns a list of hits: (chunk_id, source, text, score).
Keeping a single return shape means evaluate.py never changes as you add
fancier retrieval.
"""
import pickle

import numpy as np

from src import config

_RRF_K = 60  # reciprocal rank fusion constant (standard value)


class Retriever:
    def __init__(self):
        import faiss
        self.index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))
        with open(config.INDEX_DIR / "chunks.pkl", "rb") as fh:
            self.chunks = pickle.load(fh)

        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        self._bm25 = None
        if config.USE_HYBRID:
            self._init_bm25()
        self._reranker = None
        if config.USE_RERANKER:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _init_bm25(self):
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _dense_search(self, query: str, fetch: int) -> list[dict]:
        qv = self.embedder.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        scores, idxs = self.index.search(qv, fetch)
        hits = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            c = self.chunks[i]
            hits.append({"chunk_id": c["chunk_id"], "source": c["source"],
                         "text": c["text"], "score": float(score)})
        return hits

    def _hybrid_search(self, query: str, fetch: int) -> list[dict]:
        """Reciprocal rank fusion of dense + BM25 results."""
        dense_hits = self._dense_search(query, fetch)
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ranking = np.argsort(bm25_scores)[::-1][:fetch]

        # Build RRF score map keyed by chunk_id
        rrf: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}

        for rank, h in enumerate(dense_hits):
            cid = h["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunk_by_id[cid] = h

        for rank, idx in enumerate(bm25_ranking):
            c = self.chunks[idx]
            cid = c["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = {"chunk_id": cid, "source": c["source"],
                                    "text": c["text"], "score": 0.0}

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        hits = []
        for cid, score in ranked:
            h = dict(chunk_by_id[cid])
            h["score"] = score
            hits.append(h)
        return hits

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K
        fetch = top_k * 4 if self._reranker else top_k * 2

        if self._bm25:
            hits = self._hybrid_search(query, fetch)
        else:
            hits = self._dense_search(query, fetch)

        if self._reranker:
            pairs = [(query, h["text"]) for h in hits[:top_k * 4]]
            rr = self._reranker.predict(pairs)
            for h, s in zip(hits, rr):
                h["score"] = float(s)
            hits.sort(key=lambda h: h["score"], reverse=True)

        return hits[:top_k]


if __name__ == "__main__":
    r = Retriever()
    for h in r.search("How do I define a path parameter?"):
        print(f"{h['score']:.3f}  {h['source']}\n    {h['text'][:120]}...\n")
