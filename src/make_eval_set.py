"""Build a gold evaluation set from the docs themselves.

For a random sample of chunks, ask an LLM to write a realistic question a
developer would ask whose answer is contained in that chunk, plus the answer.
The originating chunk_id becomes the *gold* chunk for retrieval scoring.

    python -m src.make_eval_set --n 60

IMPORTANT: this produces *candidates*. Open data/eval/gold_qa.jsonl and
eyeball every row -- delete the bad ones, fix sloppy questions. A hand-checked
60-question set is the thing you describe on your CV, and the manual pass is
exactly the judgement reviewers want to see.
"""
import argparse
import json
import pickle
import random
import re

from src import config, generate


def _parse_json(raw: str) -> dict:
    """json.loads that tolerates ```json ... ``` fences from chat models."""
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)

GEN_PROMPT = """Here is a snippet from the {corpus} documentation:

\"\"\"{chunk}\"\"\"

Write ONE realistic question a developer might search for, whose answer is fully
contained in this snippet, and a short correct answer. Return strict JSON:
{{"question": "...", "answer": "..."}}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(config.INDEX_DIR / "chunks.pkl", "rb") as fh:
        chunks = pickle.load(fh)

    random.seed(args.seed)
    sample = random.sample(chunks, min(args.n, len(chunks)))

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_DIR / "gold_qa.jsonl"
    kept = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for c in sample:
            try:
                raw = generate.llm(
                    GEN_PROMPT.format(corpus=config.CORPUS_NAME, chunk=c["text"]),
                    system="You write precise documentation QA pairs. Output JSON only.",
                )
                qa = _parse_json(raw)
                row = {"question": qa["question"], "reference_answer": qa["answer"],
                       "gold_chunk_id": c["chunk_id"], "gold_source": c["source"]}
                fh.write(json.dumps(row) + "\n")
                kept += 1
            except Exception as e:  # noqa: BLE001 -- skip malformed generations
                print("skipped one:", e)

    print(f"Wrote {kept} candidate QA pairs to {out_path}")
    print("Now HAND-CHECK them before running evaluate.py.")


if __name__ == "__main__":
    main()
