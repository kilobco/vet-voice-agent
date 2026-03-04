from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _get_client():
    # Patch supabase so module-level RAGRetriever init doesn't fail
    with patch("supabase.create_client", return_value=MagicMock()):
        from app.telephony.twilio_handler import app
        return TestClient(app)


client = _get_client()


def test_health_get():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_head():
    response = client.head("/health")
    assert response.status_code == 200


def test_incoming_call_returns_xml():
    response = client.post("/incoming-call")
    assert response.status_code == 200
    assert "xml" in response.headers["content-type"]
    assert b"<Response>" in response.content


def test_incoming_call_contains_stream():
    response = client.post("/incoming-call")
    # TwiML should instruct Twilio to open a media stream
    assert b"Stream" in response.content or b"stream" in response.content.lower()


def test_incoming_call_contains_greeting():
    response = client.post("/incoming-call")
    assert b"veterinary" in response.content.lower() or b"Alexander" in response.content
