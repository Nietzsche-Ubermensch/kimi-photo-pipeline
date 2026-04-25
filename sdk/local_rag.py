"""Brave Local Place Search → RAG pipeline.

Three-stage pipeline:
  1. fetch()    — Brave /local/place_search → /local/pois → /local/descriptions
  2. index()    — build_poi_document() → embed via Venice BGE-M3 → save JSON
  3. retrieve() — cosine similarity query at runtime

POI IDs returned by place_search are valid for ~8 hours.
Max 20 IDs per pois / descriptions call.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx

from sdk.embeddings import Strategy as EmbedStrategy
from sdk.embeddings import embed, strategy_for

logger = logging.getLogger("kimi_demo.local_rag")

BASE          = "https://api.search.brave.com/res/v1"
DEFAULT_STORE = Path("/tmp/rag_store/local_rag.json")
FRESH_HEADERS = {
    "Cache-Control": "no-cache",
}


def _brave_headers() -> dict[str, str]:
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("BRAVE_API_KEY is not set")
    return {"X-Subscription-Token": api_key, **FRESH_HEADERS}


# ---------------------------------------------------------------------------
# Stage 1 — Fetch
# ---------------------------------------------------------------------------

async def place_search(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    query: str,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    location: str = "",
    radius: int = 5000,
    count: int = 20,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"q": query, "count": count}
    if lat is not None and lng is not None:
        params.update({"latitude": lat, "longitude": lng, "radius": radius})
    elif location:
        params["location"] = location

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{BASE}/local/place_search",
            params=params,
            headers=_brave_headers(),
        )
        r.raise_for_status()

    results = r.json().get("results", [])
    logger.info("place_search: %d results for %r", len(results), query)
    return results


async def poi_details(ids: list[str]) -> dict[str, dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{BASE}/local/pois",
            params=[("ids", i) for i in ids[:20]],
            headers=_brave_headers(),
        )
        r.raise_for_status()
    return {p["id"]: p for p in r.json().get("results", [])}


async def poi_descriptions(ids: list[str]) -> dict[str, str]:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{BASE}/local/descriptions",
            params=[("ids", i) for i in ids[:20]],
            headers=_brave_headers(),
        )
        r.raise_for_status()
    return {d["id"]: d.get("description", "") for d in r.json().get("results", [])}


# ---------------------------------------------------------------------------
# Stage 2 — Build index
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,#/-]", "", text)
    return text.strip()


def _poi_hash(place: dict, detail: dict) -> str:
    addr = detail.get("postal_address", {})
    key = json.dumps({
        "title": _normalize(place.get("title", "")),
        "address": _normalize(addr.get("displayAddress", "")),
        "coords": place.get("coordinates"),
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _validate_poi(place: dict, detail: dict, description: str) -> bool:
    if not place.get("title") or len(place["title"]) < 3:
        return False
    if detail.get("rating", {}).get("ratingValue", 0) < 1.0:
        return False
    if len(description) < 20:
        return False
    return True


def build_poi_document(place: dict, detail: dict, description: str) -> str:
    addr   = detail.get("postal_address", {})
    hours  = "; ".join(detail.get("opening_hours", {}).get("current_day", []))
    rating = detail.get("rating", {})
    return (
        f"{place.get('title', '')}. "
        f"{description} "
        f"Address: {addr.get('displayAddress', '')}. "
        f"Phone: {detail.get('phone', '')}. "
        f"Hours: {hours}. "
        f"Rating: {rating.get('ratingValue', '?')}/5 "
        f"({rating.get('reviewCount', '?')} reviews). "
        f"Categories: {', '.join(detail.get('categories', []))}."
    ).strip()


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
async def build_local_rag_index(
    query: str,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    location: str = "",
    radius: int = 5000,
    strategy: EmbedStrategy = "venice",
    store_path: Path = DEFAULT_STORE,
    force: bool = False,
) -> dict:
    """Full fetch → enrich → dedup → validate → embed → save pipeline."""
    store_path.parent.mkdir(parents=True, exist_ok=True)

    # Smart TTL: skip rebuild if index is < 6 hours old
    if not force and store_path.exists():
        existing = json.loads(store_path.read_text(encoding="utf-8"))
        created  = existing.get("metadata", {}).get("created", "")
        if created:
            age = datetime.now() - datetime.fromisoformat(created)
            if age < timedelta(hours=6):
                age_hours = age.total_seconds() / 3600
                logger.info("Index fresh (%.1f h old), skipping rebuild.", age_hours)
                return existing

    places = await place_search(query, lat=lat, lng=lng, location=location, radius=radius)
    ids    = [p["id"] for p in places if p.get("id")]

    if not ids:
        logger.warning("No POI IDs for query %r", query)
        return {"metadata": {}, "meta": [], "docs": [], "vectors": []}

    details = await poi_details(ids)
    descs   = await poi_descriptions(ids)

    docs:   list[str]  = []
    meta:   list[dict] = []
    hashes: set[str]   = set()

    for place in places:
        pid    = place.get("id")
        detail = details.get(pid, {})
        desc   = descs.get(pid, "")

        if not _validate_poi(place, detail, desc):
            continue

        h = _poi_hash(place, detail)
        if h in hashes:
            continue
        hashes.add(h)

        doc = build_poi_document(place, detail, desc)
        docs.append(doc)
        meta.append({
            "id":     pid,
            "title":  place.get("title", ""),
            "coords": place.get("coordinates"),
            "url":    place.get("url", ""),
            "hash":   h,
        })

    if not docs:
        logger.warning("No valid POIs after dedup+validation")
        return {"metadata": {}, "meta": [], "docs": [], "vectors": []}

    vectors = await embed(docs, strategy=strategy)

    index = {
        "metadata": {
            "created":  datetime.now().isoformat(),
            "query":    query,
            "lat":      lat,
            "lng":      lng,
            "location": location,
            "count":    len(docs),
            "strategy": strategy,
        },
        "meta":    meta,
        "docs":    docs,
        "vectors": vectors,
    }
    store_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Indexed %d valid POIs → %s", len(docs), store_path)
    return index


# ---------------------------------------------------------------------------
# Stage 3 — Retrieve
# ---------------------------------------------------------------------------

def _cosine_batch(query_vec: list[float], matrix: list[list[float]]) -> list[float]:
    def norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))
    qn = norm(query_vec)
    scores = []
    for row in matrix:
        dot = sum(a * b for a, b in zip(query_vec, row))
        rn  = norm(row)
        scores.append(dot / (qn * rn) if qn and rn else 0.0)
    return scores


async def retrieve_local(
    user_query: str,
    store_path: Path = DEFAULT_STORE,
    strategy: EmbedStrategy = "venice",
    top_k: int = 5,
) -> list[dict]:
    """Embed user_query and return top_k POIs ranked by cosine similarity."""
    if not store_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: {store_path}. Run build_local_rag_index() first."
        )

    index   = json.loads(store_path.read_text(encoding="utf-8"))
    vectors = index["vectors"]
    meta    = index["meta"]
    docs    = index["docs"]

    q_vecs = await embed([user_query], strategy=strategy)
    scores = _cosine_batch(q_vecs[0], vectors)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {**meta[i], "doc": docs[i], "score": round(scores[i], 4)}
        for i in ranked
    ]


# ---------------------------------------------------------------------------
# Prompt enrichment
# ---------------------------------------------------------------------------

def build_location_prompt(
    base_prompt: str,
    poi_hits: list[dict],
    top_k: int = 3,
) -> str:
    """Append top-k POI descriptions as scene/location context for photo_gen."""
    lines = [
        f"- {h['title']}: {h['doc'][:200]}"
        for h in poi_hits[:top_k]
        if h.get("title")
    ]
    if not lines:
        return base_prompt
    context = "\n".join(lines)
    return (
        f"{base_prompt}\n\n"
        f"Nearby location context (use for atmosphere, setting, style):\n{context}"
    )


# ---------------------------------------------------------------------------
# CLI demo: python sdk/local_rag.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _demo() -> None:
        index = await build_local_rag_index(
            query="photo studio OR art gallery OR creative space",
            location="Bradenton, FL",
            radius=10_000,
            strategy=strategy_for("rag_poi"),
        )
        print(f"Indexed {len(index['docs'])} POIs")

        hits = await retrieve_local(
            "dramatic indoor portrait studio with natural light",
            strategy=strategy_for("rag_poi"),
            top_k=3,
        )
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['title']}")

    asyncio.run(_demo())
