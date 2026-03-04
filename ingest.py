"""
Re-ingestion script — embeds vetvoice_rag_qa.csv with Jina AI
and upserts into Supabase.

Run once locally:
    python ingest.py
"""

import csv
import time
import httpx
from supabase import create_client
from config.settings import SUPABASE_URL, SUPABASE_KEY, JINA_API_KEY

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL         = "jina-embeddings-v3"
CSV_PATH           = "data/vetvoice_rag_qa.csv"
BATCH_SIZE         = 20  # Jina free tier: up to 2048 tokens per item, batch up to 2048 items


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Jina AI (task=retrieval.passage for documents)."""
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      JINA_MODEL,
        "input":      texts,
        "task":       "retrieval.passage",
        "normalized": True,
    }
    response = httpx.post(JINA_EMBEDDING_URL, headers=headers, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()["data"]
    # Sort by index to preserve order
    data.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data]


def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Load CSV
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # Clear existing rows before re-ingesting with new embeddings
    print("Clearing existing rows...")
    supabase.table("documents").delete().neq("title", "").execute()
    print("Cleared.")

    # Process in batches
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

        # Combine title + body for richer document embeddings
        texts = [f"{r['title']}\n{r['body']}" for r in batch]

        print(f"Embedding rows {i+1}–{i+len(batch)}...")
        embeddings = embed_batch(texts)

        # Insert into Supabase (table: documents, columns: title, body, embedding)
        records = [
            {
                "title":     row["title"],
                "body":      row["body"],
                "embedding": embedding,
            }
            for row, embedding in zip(batch, embeddings)
        ]
        supabase.table("documents").insert(records).execute()
        print(f"  Inserted {len(records)} records.")

        # Small delay to stay within rate limits
        if i + BATCH_SIZE < len(rows):
            time.sleep(0.5)

    print("Done. All rows embedded and stored in Supabase.")


if __name__ == "__main__":
    main()
