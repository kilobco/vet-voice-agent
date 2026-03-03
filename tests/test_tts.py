import pytest
from unittest.mock import patch, MagicMock


@patch("app.tts.deepgram_tts.client")
def test_synthesize_to_file(mock_client, tmp_path):
    output_path = str(tmp_path / "output.wav")
    mock_client.speak.rest.v.return_value.save.return_value = None

    from app.tts.deepgram_tts import synthesize_to_file

    synthesize_to_file("Hello, how can I help you?", output_path)
    mock_client.speak.rest.v.return_value.save.assert_called_once()


@patch("app.tts.deepgram_tts.client")
def test_synthesize_to_bytes_returns_bytes(mock_client):
    mock_stream = MagicMock()
    mock_stream.stream = [b"audio_chunk_1", b"audio_chunk_2"]
    mock_client.speak.rest.v.return_value.stream.return_value = mock_stream

    from app.tts.deepgram_tts import synthesize_to_bytes

    result = synthesize_to_bytes("We are open Monday to Friday.")
    assert isinstance(result, bytes)
    assert len(result) > 0
