from unittest.mock import AsyncMock, MagicMock, patch
from app.tts.deepgram_tts import DeepgramTTS, _split_text


# ── _split_text ───────────────────────────────────────────────────────────────

def test_split_text_short_passthrough():
    assert _split_text("Hello.") == ["Hello."]


def test_split_text_long_breaks_at_sentences():
    long_text = ("This is a sentence. " * 120).strip()  # well over 2000 chars
    chunks = _split_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 2000


def test_split_text_exactly_at_limit():
    text = "A" * 2000
    assert _split_text(text) == [text]


# ── DeepgramTTS ───────────────────────────────────────────────────────────────

def test_init_uses_aura2():
    tts = DeepgramTTS(api_key="test_key")
    assert tts.model == "aura-2-asteria-en"


async def test_synthesize_returns_bytes():
    tts = DeepgramTTS(api_key="test_key")

    mock_response = MagicMock()
    mock_response.content = b"audio_data"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tts.synthesize("Hello.")

    assert isinstance(result, bytes)
    assert result == b"audio_data"


async def test_synthesize_chunks_long_text():
    """Texts over 2000 chars should trigger multiple POST calls."""
    tts = DeepgramTTS(api_key="test_key")

    long_text = ("Short sentence. " * 130).strip()  # > 2000 chars

    mock_response = MagicMock()
    mock_response.content = b"chunk"
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tts.synthesize(long_text)

    assert mock_client.post.call_count > 1
    assert isinstance(result, bytes)
