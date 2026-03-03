import httpx


DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"


class DeepgramTTS:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model   = "aura-asteria-en"   # natural female voice
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type":  "application/json",
        }

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech and return raw audio bytes.
        Audio is returned as mulaw 8kHz — ready for Twilio streaming.
        """
        params  = {
            "model":       self.model,
            "encoding":    "mulaw",     # Twilio-compatible format
            "sample_rate": "8000",      # Twilio sample rate
            "container":   "none",      # raw audio, no container
        }
        payload = {"text": text}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                DEEPGRAM_TTS_URL,
                headers = self.headers,
                params  = params,
                json    = payload,
                timeout = 10.0,
            )
            response.raise_for_status()
            print(f"[TTS] Synthesized: {text[:60]}...")
            return response.content

    async def synthesize_and_stream(self, text: str):
        """
        Stream audio chunks as they arrive — lower latency for live calls.
        Yields audio chunks as bytes.
        """
        params  = {
            "model":       self.model,
            "encoding":    "mulaw",
            "sample_rate": "8000",
            "container":   "none",
        }
        payload = {"text": text}

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                DEEPGRAM_TTS_URL,
                headers = self.headers,
                params  = params,
                json    = payload,
                timeout = 30.0,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=640):
                    yield chunk
