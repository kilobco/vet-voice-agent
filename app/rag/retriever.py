import math
import httpx
from supabase import create_client

HF_EMBEDDING_URL = "https://api-inference.huggingface.co/models/BAAI/bge-m3"


class RAGRetriever:
    def __init__(self, supabase_url: str, supabase_key: str, hf_token: str):
        self.supabase  = create_client(supabase_url, supabase_key)
        self.hf_token  = hf_token

    async def _embed(self, text: str) -> list:
        """
        Embed query text via HuggingFace Inference API (BAAI/bge-m3).
        No local model — no RAM spike. Returns a normalized embedding vector.
        """
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_EMBEDDING_URL,
                headers = headers,
                json    = {"inputs": text},
                timeout = 30.0,
            )
            response.raise_for_status()

        embedding = response.json()

        # HF returns [[...]] for a single string input
        if isinstance(embedding[0], list):
            embedding = embedding[0]

        # L2 normalize to match how the stored embeddings were created
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

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
