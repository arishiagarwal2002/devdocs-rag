"""The evaluation harness -- the heart of the project.

Runs the current config (set in config.py) against the gold QA set and reports:

  Retrieval quality (gold-chunk based):
    - Recall@k : was the gold chunk among the top-k retrieved?
    - MRR      : 1/rank of the gold chunk (0 if missed)

  Answer quality (LLM-as-judge):
    - Faithfulness : is the answer supported by the retrieved context? (0/1)
    - Correctness  : does it match the reference answer? (0/1)

    python -m src.evaluate                  # writes a row to experiments/results.csv

Workflow for the ablation study: change ONE setting in config.py, re-run
ingest if you changed chunking/embeddings, then re-run this. Each run appends a
labelled row to results.csv -- that table is your headline deliverable.
"""
import csv
import json
import re
import time

from src import config, generate


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)
from src.rag import RAG

JUDGE_PROMPT = """Question: {q}
Reference answer: {ref}
Retrieved context: {ctx}
Model answer: {ans}

Score two things, each 0 or 1:
- faithfulness: is the model answer supported by the retrieved context (no invented facts)?
- correctness: does the model answer agree with the reference answer?
Return strict JSON: {{"faithfulness": 0 or 1, "correctness": 0 or 1}}"""


def load_gold() -> list[dict]:
    path = config.EVAL_DIR / "gold_qa.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run src.make_eval_set first, then hand-check gold_qa.jsonl")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge(q, ref, ctx, ans) -> dict:
    try:
        raw = generate.llm(
            JUDGE_PROMPT.format(q=q, ref=ref, ctx=ctx[:4000], ans=ans),
            system="You are a strict grader. Output JSON only.",
        )
        return _parse_json(raw)
    except Exception:
        return {"faithfulness": 0, "correctness": 0}


def run() -> dict:
    gold = load_gold()
    rag = RAG()

    recall, rr, faith, correct = 0, 0.0, 0, 0
    for row in gold:
        out = rag.ask(row["question"])
        ids = [h["chunk_id"] for h in out["hits"]]
        if row["gold_chunk_id"] in ids:
            recall += 1
            rr += 1.0 / (ids.index(row["gold_chunk_id"]) + 1)
        ctx = "\n".join(h["text"] for h in out["hits"])
        scores = judge(row["question"], row["reference_answer"], ctx, out["answer"])
        faith += int(scores.get("faithfulness", 0))
        correct += int(scores.get("correctness", 0))

    n = len(gold)
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "corpus": config.CORPUS_NAME,
        "embedding": config.EMBEDDING_MODEL.split("/")[-1],
        "chunk_size": config.CHUNK_SIZE,
        "top_k": config.TOP_K,
        "hybrid": config.USE_HYBRID,
        "reranker": config.USE_RERANKER,
        "n": n,
        "recall@k": round(recall / n, 3),
        "mrr": round(rr / n, 3),
        "faithfulness": round(faith / n, 3),
        "correctness": round(correct / n, 3),
    }


def main():
    res = run()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "results.csv"
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res.keys()))
        if write_header:
            w.writeheader()
        w.writerow(res)
    print(json.dumps(res, indent=2))
    print(f"\nAppended to {path}")


if __name__ == "__main__":
    main()
