import re
import httpx


DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"
MAX_CHARS = 2000  # Deepgram TTS hard limit per request


def _split_text(text: str) -> list:
    """Split text into chunks <= MAX_CHARS, breaking at sentence boundaries."""
    if len(text) <= MAX_CHARS:
        return [text]

    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= MAX_CHARS:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


class DeepgramTTS:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model   = "aura-asteria-en"   # Aura-1: stable, widely available
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type":  "application/json",
        }

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech and return raw audio bytes.
        Automatically chunks text exceeding the 2000-character limit.
        Audio is returned as mulaw 8kHz — ready for Twilio streaming.
        """
        params = {
            "model":       self.model,
            "encoding":    "mulaw",
            "sample_rate": "8000",
            "container":   "none",
        }
        audio = b""
        async with httpx.AsyncClient() as client:
            for chunk in _split_text(text):
                response = await client.post(
                    DEEPGRAM_TTS_URL,
                    headers = self.headers,
                    params  = params,
                    json    = {"text": chunk},
                    timeout = 10.0,
                )
                response.raise_for_status()
                audio += response.content

        print(f"[TTS] Synthesized: {text[:60]}...")
        return audio

    async def synthesize_and_stream(self, text: str):
        """
        Stream audio chunks as they arrive — lower latency for live calls.
        Automatically chunks text exceeding the 2000-character limit.
        Yields audio chunks as bytes.
        """
        params = {
            "model":       self.model,
            "encoding":    "mulaw",
            "sample_rate": "8000",
            "container":   "none",
        }
        async with httpx.AsyncClient() as client:
            for text_chunk in _split_text(text):
                async with client.stream(
                    "POST",
                    DEEPGRAM_TTS_URL,
                    headers = self.headers,
                    params  = params,
                    json    = {"text": text_chunk},
                    timeout = 30.0,
                ) as response:
                    response.raise_for_status()
                    async for audio_chunk in response.aiter_bytes(chunk_size=640):
                        yield audio_chunk
