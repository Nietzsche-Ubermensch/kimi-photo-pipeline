"""sdk/embeddings.py
────────────────────────────────────────────────────────────────────────────
Embedding router for the Kimi photo-gen + RAG pipeline.

Strategy  →  config.toml model key       →  provider / dims
────────────────────────────────────────────────────────────────────────────
venice      bge-m3                       Venice BGE-M3       1024-dim  <- default
hf          hf-bge-m3                    HuggingFace BGE-M3  1024-dim
nv-e5       nv-embed-v2                  NVIDIA E5-v5         768-dim
nv-llama    nemotron-llama-embed         NVIDIA LLaMA EmbedQA 4096-dim
nv-qa       nemotron-embed-qa            NVIDIA NV-Embed-QA   4096-dim
"""
from __future__ import annotations

import asyncio
import os
from typing import Literal

import httpx

Strategy = Literal["venice", "hf", "nv-e5", "nv-llama", "nv-qa"]

_ROUTES: dict[str, tuple[str, str, str]] = {
    "venice": (
        "https://api.venice.ai/api/v1",
        "text-embedding-bge-m3",
        "VENICE_API_KEY",
    ),
    "hf": (
        "https://api-inference.huggingface.co/v1",
        "BAAI/bge-m3",
        "HF_TOKEN",
    ),
    "nv-e5": (
        "https://integrate.api.nvidia.com/v1",
        "nvidia/nv-embedqa-e5-v5",
        "NVIDIA_API_KEY",
    ),
    "nv-llama": (
        "https://integrate.api.nvidia.com/v1",
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "NVIDIA_API_KEY",
    ),
    "nv-qa": (
        "https://integrate.api.nvidia.com/v1",
        "NV-Embed-QA",
        "NVIDIA_API_KEY",
    ),
}

_TASK_STRATEGY: dict[str, Strategy] = {
    "rag_poi":      "venice",
    "rag_dedup":    "venice",
    "retrieval_hq": "nv-llama",
    "retrieval_qa": "nv-qa",
    "fallback":     "hf",
}

DIMS: dict[Strategy, int] = {
    "venice":   1024,
    "hf":       1024,
    "nv-e5":     768,
    "nv-llama": 4096,
    "nv-qa":    4096,
}


def strategy_for(task: str) -> Strategy:
    """Return the best Strategy for a named task."""
    return _TASK_STRATEGY.get(task, "venice")


async def embed(
    texts: list[str],
    strategy: Strategy = "venice",
    batch_size: int = 20,
) -> list[list[float]]:
    """Embed a list of strings. Returns float vectors in input order."""
    base_url, model_id, env_var = _ROUTES[strategy]
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise EnvironmentError(
            f"Strategy '{strategy}' requires env var {env_var} to be set."
        )

    all_vectors: list[list[float]] = []

    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = await client.post(
                f"{base_url}/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model_id, "input": batch, "encoding_format": "float"},
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda x: x["index"])
            all_vectors.extend(item["embedding"] for item in data)

    return all_vectors


def embed_sync(texts: list[str], strategy: Strategy = "venice") -> list[list[float]]:
    """Synchronous wrapper around embed() for non-async callers."""
    return asyncio.run(embed(texts, strategy=strategy))
