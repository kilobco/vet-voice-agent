from unittest.mock import AsyncMock, MagicMock
from app.stt.deepgram_stt import DeepgramSTT


def test_init():
    stt = DeepgramSTT(api_key="test_key")
    assert stt.model == "nova-3"


async def test_transcribe_stream_starts_connection():
    stt = DeepgramSTT(api_key="test_key")

    mock_connection = MagicMock()
    mock_connection.on    = MagicMock()
    mock_connection.start = AsyncMock()

    stt.client = MagicMock()
    stt.client.listen.asyncwebsocket.v.return_value = mock_connection

    connection = await stt.transcribe_stream(AsyncMock())

    mock_connection.start.assert_called_once()
    assert connection is mock_connection


async def test_keep_alive_calls_connection():
    stt = DeepgramSTT(api_key="test_key")

    mock_connection = MagicMock()
    mock_connection.keep_alive = AsyncMock()

    await stt.keep_alive(mock_connection)

    mock_connection.keep_alive.assert_called_once()


async def test_close_stream_calls_finish():
    stt = DeepgramSTT(api_key="test_key")

    mock_connection = MagicMock()
    mock_connection.finish = AsyncMock()

    await stt.close_stream(mock_connection)

    mock_connection.finish.assert_called_once()
