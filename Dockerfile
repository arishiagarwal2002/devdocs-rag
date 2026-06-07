FROM python:3.11-slim

# HF Spaces requires port 7860
EXPOSE 7860

WORKDIR /app

# Install system dependencies (needed for some sentence-transformers builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Build FAISS index from committed chunks.jsonl
# (binary files are excluded from git — this rebuilds them at image build time)
RUN python scripts/build_index_from_jsonl.py

# Streamlit config for HF Spaces (headless, correct port, no CORS)
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
