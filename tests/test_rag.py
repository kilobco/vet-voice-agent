import pytest
from unittest.mock import patch, MagicMock


@patch("app.rag.retriever.supabase")
@patch("app.rag.retriever.model")
def test_retrieve_returns_list(mock_model, mock_supabase):
    mock_model.encode.return_value = [0.1] * 1024
    mock_supabase.rpc.return_value.execute.return_value.data = [
        {"title": "Vaccines", "body": "Annual vaccines are recommended."}
    ]

    from app.rag.retriever import retrieve

    results = retrieve("What vaccines does my dog need?")
    assert isinstance(results, list)
    assert len(results) == 1
    assert "title" in results[0]


@patch("app.rag.retriever.supabase")
@patch("app.rag.retriever.model")
def test_retrieve_empty_query(mock_model, mock_supabase):
    mock_model.encode.return_value = [0.0] * 1024
    mock_supabase.rpc.return_value.execute.return_value.data = []

    from app.rag.retriever import retrieve

    results = retrieve("")
    assert results == []
