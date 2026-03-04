import httpx
from supabase import create_client

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL         = "jina-embeddings-v3"


class RAGRetriever:
    def __init__(self, supabase_url: str, supabase_key: str, jina_api_key: str):
        self.supabase     = create_client(supabase_url, supabase_key)
        self.jina_api_key = jina_api_key

    async def _embed(self, text: str) -> list:
        """
        Embed query text via Jina AI API (jina-embeddings-v3).
        Uses task='retrieval.query' for optimized query-side embeddings.
        """
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":      JINA_MODEL,
            "input":      [text],
            "task":       "retrieval.query",
            "normalized": True,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                JINA_EMBEDDING_URL,
                headers = headers,
                json    = payload,
                timeout = 30.0,
            )
            response.raise_for_status()

        return response.json()["data"][0]["embedding"]

    async def retrieve(self, query: str, threshold: float = 0.5, top_k: int = 5) -> list:
        """Embed query and retrieve matching documents from Supabase."""
        query_embedding = await self._embed(query)

        results = self.supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_threshold':  threshold,
            'match_count':      top_k,
        }).execute()

        return results.data

    def format_context(self, docs: list) -> str:
        """Format retrieved docs into a single context string for the LLM."""
        if not docs:
            return "No relevant information found."

        context = ""
        for i, doc in enumerate(docs, 1):
            title = doc.get('title') or doc.get('question') or ''
            body  = doc.get('body')  or doc.get('answer')  or doc.get('content') or ''
            context += f"[{i}] {title}\n{body}\n\n"

        return context.strip()
