from unittest.mock import MagicMock, patch
from app.rag.retriever import RAGRetriever


def _make_retriever():
    with patch("supabase.create_client", return_value=MagicMock()):
        return RAGRetriever(supabase_url="http://localhost", supabase_key="test_key")


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

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    r._model = mock_model

    results = await r.retrieve("What vaccines does my dog need?")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["title"] == "Vaccines"


async def test_retrieve_empty_query():
    r = _make_retriever()

    mock_result      = MagicMock()
    mock_result.data = []

    r.supabase.rpc.return_value.execute.return_value = mock_result

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 1024)
    r._model = mock_model

    results = await r.retrieve("")
    assert results == []
