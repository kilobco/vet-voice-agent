import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def flask_client():
    from app.telephony.twilio_handler import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_voice_webhook_returns_xml(flask_client):
    response = flask_client.post("/voice", data={"CallSid": "CA123"})
    assert response.status_code == 200
    assert b"<Response>" in response.data
    assert b"Gather" in response.data or b"gather" in response.data.lower()


@patch("app.telephony.twilio_handler.answer")
def test_handle_speech_returns_xml(mock_answer, flask_client):
    mock_answer.return_value = "We are open Monday to Friday."
    response = flask_client.post(
        "/handle-speech",
        data={
            "CallSid": "CA123",
            "SpeechResult": "What are your hours?",
        },
    )
    assert response.status_code == 200
    assert b"<Response>" in response.data


def test_call_status_cleans_up(flask_client):
    from app.telephony.twilio_handler import _conversations
    _conversations["CA999"] = [{"role": "user", "content": "hi"}]
    response = flask_client.post("/call-status", data={"CallSid": "CA999"})
    assert response.status_code == 204
    assert "CA999" not in _conversations
