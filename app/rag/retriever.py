from supabase import create_client


class RAGRetriever:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase   = create_client(supabase_url, supabase_key)
        self._model     = None  # lazy loaded on first call

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import — keeps torch off the startup path
            self._model = SentenceTransformer('BAAI/bge-m3')
        return self._model

    def retrieve(self, query: str, threshold: float = 0.5, top_k: int = 5) -> list:
        """Convert query to vector and retrieve matching documents."""
        query_embedding = self._get_model().encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.supabase.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_threshold':  threshold,
            'match_count':      top_k
        }).execute()

        return results.data

    def format_context(self, docs: list) -> str:
        """Format retrieved docs into a single context string for the LLM."""
        if not docs:
            return "No relevant information found."

        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"[{i}] {doc['title']}\n{doc['body']}\n\n"

        return context.strip()
