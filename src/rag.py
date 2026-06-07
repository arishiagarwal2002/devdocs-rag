"""The RAG pipeline: question -> retrieve -> generate.

    from src.rag import RAG
    rag = RAG()
    out = rag.ask("How do I return a custom status code?")
    print(out["answer"])
    for h in out["hits"]:
        print(h["source"])
"""
from src.retrieve import Retriever
from src import generate


class RAG:
    def __init__(self):
        self.retriever = Retriever()

    def ask(self, question: str, top_k: int | None = None) -> dict:
        hits = self.retriever.search(question, top_k=top_k)
        ans = generate.answer(question, [h["text"] for h in hits])
        return {"question": question, "answer": ans, "hits": hits}


if __name__ == "__main__":
    rag = RAG()
    out = rag.ask("How do I define a path parameter in FastAPI?")
    print("ANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for h in out["hits"]:
        print(" -", h["source"])
