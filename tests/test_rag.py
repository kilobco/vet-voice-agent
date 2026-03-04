from unittest.mock import MagicMock, patch
from app.rag.retriever import RAGRetriever


def _make_retriever():
    with patch("supabase.create_client", return_value=MagicMock()):
        return RAGRetriever(supabase_url="http://localhost", supabase_key="test_key", jina_api_key="test_jina_key")


# ── format_context ────────────────────────────────────────────────────────────

def test_format_context_empty():
    r = _make_retriever()
    assert r.format_context([]) == "No relevant information found."


def test_format_context_with_docs():
    r    = _make_retriever()
    docs = [{"title": "Vaccines", "body": "Annual vaccines are recommended."}]
    ctx  = r.format_context(docs)
    assert "Vaccines" in ctx
    assert "Annual vaccines" in ctx


def test_format_context_fallback_field_names():
    """Should handle question/answer field names from alternate Supabase schemas."""
    r    = _make_retriever()
    docs = [{"question": "What are your hours?", "answer": "9am to 5pm."}]
    ctx  = r.format_context(docs)
    assert "What are your hours?" in ctx
    assert "9am to 5pm." in ctx


# ── retrieve ──────────────────────────────────────────────────────────────────

async def test_retrieve_returns_list():
    r = _make_retriever()

    mock_result      = MagicMock()
    mock_result.data = [{"title": "Vaccines", "body": "Annual vaccines are recommended."}]
    r.supabase.rpc.return_value.execute.return_value = mock_result

    # Mock the HuggingFace embedding API call
    with patch.object(r, "_embed", return_value=[0.1] * 1024):
        results = await r.retrieve("What vaccines does my dog need?")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Vaccines"


async def test_retrieve_empty_results():
    r = _make_retriever()

    mock_result      = MagicMock()
    mock_result.data = []
    r.supabase.rpc.return_value.execute.return_value = mock_result

    with patch.object(r, "_embed", return_value=[0.0] * 1024):
        results = await r.retrieve("What vaccines does my dog need?")

    assert results == []
