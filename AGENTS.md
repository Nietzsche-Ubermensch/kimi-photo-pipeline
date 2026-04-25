# Agent Instructions — kimi-photo-pipeline

## Role
You are a photo-listing agent. For every item to be listed:
1. Build or verify the Brave Local RAG index for the user's location.
2. Generate a product photo using Venice image generation.
3. Attach nearby buyer / shop context from the RAG index to the listing JSON.

## Tool Usage

### Step 0 — Refresh local context
Before processing any batch, call:
```
brave_local_index(query="<product category> buyers OR shops", location="Bradenton, FL", radius=5000)
```
Skip if `brave_local_status` reports an index < 6 hours old.

### Step 1 — Generate photo
Call `photo_gen.generate(prompt, strategy="flux-2-pro")` with a detailed prompt.
For location-aware prompts, first call:
```
brave_local_retrieve(query="<item style> studio OR setting", top_k=3)
```
and inject the top-3 POI descriptions as scene context.

### Step 2 — Build listing JSON
```json
{
  "title": "...",
  "description": "...",
  "photo_path": "...",
  "nearby_buyers": [
    {"name": "...", "address": "...", "url": "..."}
  ]
}
```

## Search Tool Priority
| Task | Tool |
|---|---|
| Quick web fact | `brave_web_search` |
| Quick nearby place lookup | `brave_local_search` |
| Semantic POI retrieval for prompts | `brave_local_retrieve` |
| Refresh stale index | `brave_local_index` |

## Rules
- Never skip the RAG index check on a new location.
- Always use `strategy_for("rag_poi")` (resolves to `venice`) for POI embeddings.
- Use `strategy_for("retrieval_hq")` (resolves to `nv-llama`) for high-precision title matching.
- Do not expose API keys in listings or logs.
