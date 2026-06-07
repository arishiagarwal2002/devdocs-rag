"""Generation. One function -- llm() -- so swapping providers is trivial.

Backends controlled by config.LLM_BACKEND:
  "openai"  — reads OPENAI_API_KEY from environment, uses config.LLM_MODEL
  "ollama"  — free local inference; requires `ollama serve` and
              `ollama pull llama3.2` (or whatever config.OLLAMA_MODEL names).
              Install: https://ollama.com
"""
import os

from src import config

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question using ONLY "
    "the provided context snippets. If the answer is not in the context, say you "
    "don't know rather than guessing. Be concise and include short code examples "
    "when the context contains them."
)

ANSWER_TEMPLATE = """Context snippets:
{context}

Question: {question}

Answer (grounded only in the context above):"""


def _openai_llm(prompt: str, system: str, temperature: float) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Either export it or set LLM_BACKEND='ollama' "
            "in config.py for free local inference."
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def _ollama_llm(prompt: str, system: str, temperature: float) -> str:
    """Calls a locally-running Ollama server (default http://localhost:11434)."""
    import urllib.request, json
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "options": {"temperature": temperature},
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return body["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}\n"
            "Make sure `ollama serve` is running and the model is pulled:\n"
            f"  ollama pull {config.OLLAMA_MODEL}"
        ) from exc


def _hf_llm(prompt: str, system: str, temperature: float) -> str:
    """HuggingFace Inference API — no local model needed; works on HF Spaces."""
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=token)
    resp = client.chat_completion(
        model=config.HF_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": prompt}],
        temperature=max(temperature, 0.01),   # HF API rejects 0.0
        max_tokens=config.MAX_NEW_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def _hf_stream(prompt: str, system: str, temperature: float):
    """Generator that yields text tokens from HF Inference API (streaming)."""
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=token)
    stream = client.chat_completion(
        model=config.HF_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": prompt}],
        temperature=max(temperature, 0.01),
        max_tokens=config.MAX_NEW_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def llm(prompt: str, system: str = SYSTEM_PROMPT, temperature: float | None = None) -> str:
    temperature = config.GEN_TEMPERATURE if temperature is None else temperature
    if config.LLM_BACKEND == "openai":
        return _openai_llm(prompt, system, temperature)
    if config.LLM_BACKEND == "ollama":
        return _ollama_llm(prompt, system, temperature)
    if config.LLM_BACKEND == "hf":
        return _hf_llm(prompt, system, temperature)
    raise ValueError(f"Unknown LLM_BACKEND={config.LLM_BACKEND!r}. Choose 'openai', 'ollama', or 'hf'.")


def answer(question: str, contexts: list[str]) -> str:
    joined = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return llm(ANSWER_TEMPLATE.format(context=joined, question=question))


# ── Streaming ─────────────────────────────────────────────────────────────────

def _ollama_stream(prompt: str, system: str, temperature: float):
    """Generator that yields text tokens from Ollama's streaming API."""
    import urllib.request, json
    payload = json.dumps({
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": prompt}],
        "options": {"temperature": temperature},
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if body.get("done"):
                    break
                token = body.get("message", {}).get("content", "")
                if token:
                    yield token
    except Exception as exc:
        raise RuntimeError(
            f"Ollama streaming failed: {exc}\n"
            "Make sure `ollama serve` is running."
        ) from exc


def stream_answer(question: str, contexts: list[str]):
    """Generator yielding answer tokens for all backends."""
    joined = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = ANSWER_TEMPLATE.format(context=joined, question=question)
    if config.LLM_BACKEND == "ollama":
        yield from _ollama_stream(prompt, SYSTEM_PROMPT, config.GEN_TEMPERATURE)
    elif config.LLM_BACKEND == "hf":
        yield from _hf_stream(prompt, SYSTEM_PROMPT, config.GEN_TEMPERATURE)
    else:
        # OpenAI: single yield (no streaming SDK used here)
        yield answer(question, contexts)
