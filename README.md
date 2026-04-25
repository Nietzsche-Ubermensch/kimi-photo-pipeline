# kimi-photo-pipeline

Photo-generation + Brave Local RAG pipeline for Kimi K2.6.

## Stack
- **LLM routing** — Kimi K2.6 via OpenRouter (default)
- **Image generation** — Venice (Flux 2 Pro/Max, Grok Imagine, Recraft V4, Wan 2.7)
- **Embeddings** — Venice BGE-M3 (primary), NVIDIA NV-EmbedQA (high quality), HuggingFace fallback
- **Local search** — Brave Place Search API (3-stage: search → pois → descriptions)
- **Agent runtime** — Hermes on DigitalOcean via OpenRouter Spawn
- **MCP servers** — Brave Search, Brave Local RAG, GitHub

## Structure
```
sdk/
  embeddings.py     # Strategy router: venice | hf | nv-e5 | nv-llama | nv-qa
  local_rag.py      # 3-stage Brave POI fetch → embed → retrieve
  mcp_server.py     # MCP server exposing RAG tools to Hermes
  photo_gen.py      # Venice image generation
  rag.py            # General RAG utilities
  session_demo.py   # Kimi multi-turn session demo
  approval_policy.py
  tools.py
  agent.yaml
.hermes/
  config.yaml       # Hermes MCP server wiring
.kimi/
  config.toml       # Kimi CLI provider + model config
.github/workflows/
  pylint.yml        # CI with project deps pre-installed
```

## Quick Start
```bash
# Clone and install
git clone https://github.com/Nietzsche-Ubermensch/kimi-photo-pipeline
cd kimi-photo-pipeline
pip install -e .

# Set env vars
export BRAVE_API_KEY=...
export VENICE_API_KEY=...
export NVIDIA_API_KEY=...
export HF_TOKEN=...
export OPENROUTER_API_KEY=...

# Deploy to DigitalOcean via Spawn
curl -fsSL https://openrouter.ai/labs/spawn/cli/install.sh | bash
spawn hermes digitalocean
```

## Brave Local RAG Pipeline
```python
from sdk.local_rag import build_local_rag_index, retrieve_local
from sdk.embeddings import strategy_for

# Build index (Bradenton, FL — 5km radius)
index = await build_local_rag_index(
    query="photo studio OR art gallery",
    location="Bradenton, FL",
    radius=5000,
    strategy=strategy_for("rag_poi"),  # -> "venice"
)

# Retrieve top-5 semantically similar POIs
hits = await retrieve_local("dramatic portrait studio with natural light", top_k=5)
```
