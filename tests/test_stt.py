import pytest
from unittest.mock import patch, MagicMock


@patch("app.stt.deepgram_stt.client")
def test_transcribe_file_returns_string(mock_client, tmp_path):
    # Create a dummy audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

    mock_alternative = MagicMock()
    mock_alternative.transcript = "Hello, I need help with my pet."
    mock_client.listen.rest.v.return_value.transcribe_file.return_value \
        .results.channels[0].alternatives[0] = mock_alternative

    from app.stt.deepgram_stt import transcribe_file

    result = transcribe_file(str(audio_file))
    assert isinstance(result, str)
    assert len(result) > 0


@patch("app.stt.deepgram_stt.client")
def test_transcribe_url_returns_string(mock_client):
    mock_alternative = MagicMock()
    mock_alternative.transcript = "My cat is not eating."
    mock_client.listen.rest.v.return_value.transcribe_url.return_value \
        .results.channels[0].alternatives[0] = mock_alternative

    from app.stt.deepgram_stt import transcribe_url

    result = transcribe_url("https://example.com/audio.wav")
    assert isinstance(result, str)
