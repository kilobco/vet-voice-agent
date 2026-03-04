import asyncio
import base64
import json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from app.stt.deepgram_stt import DeepgramSTT
from app.llm.agent         import LLMAgent
from app.tts.deepgram_tts  import DeepgramTTS
from app.rag.retriever     import RAGRetriever
from config.settings       import (
    DEEPGRAM_API_KEY,
    ANTHROPIC_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
)

app = FastAPI()


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


# ── Initialize all services ───────────────────────────────────────────────────
rag   = RAGRetriever(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)
stt   = DeepgramSTT(api_key=DEEPGRAM_API_KEY)
tts   = DeepgramTTS(api_key=DEEPGRAM_API_KEY)
agent = LLMAgent(anthropic_api_key=ANTHROPIC_API_KEY, rag=rag)


# ── Step 1: Twilio hits this endpoint when a call comes in ────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio calls this webhook when someone calls your number.
    We respond with TwiML that tells Twilio to open a media stream.
    """
    response = VoiceResponse()
    response.say("Thank you for calling. How can I help you today?")

    connect = Connect()
    connect.stream(url=f"wss://{request.headers['host']}/media-stream")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


# ── Step 2: Twilio streams audio to this WebSocket ───────────────────────────
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio sends audio chunks here as base64 encoded mulaw.
    We pipe them to Deepgram STT → LLM → Deepgram TTS → back to Twilio.
    """
    await websocket.accept()
    print("[Twilio] WebSocket connected")

    stream_sid     = None
    stt_connection = None

    # ── Callback: STT → LLM → TTS → Twilio ──────────────────────────────────
    async def handle_transcript(text: str):
        print(f"[STT→LLM] Received: {text}")

        # Stream sentence chunks from LLM → TTS → Twilio as they arrive
        async for sentence in agent.ask_stream(text):
            async for audio_chunk in tts.synthesize_and_stream(sentence):
                audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                await websocket.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_b64
                    }
                }))

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            # Twilio sends different event types
            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                print(f"[Twilio] Stream started: {stream_sid}")
                # ── Open Deepgram STT stream only after stream_sid is set ────
                stt_connection = await stt.transcribe_stream(handle_transcript)

            elif data["event"] == "media":
                # Decode base64 audio and send to Deepgram
                if stt_connection:
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    await stt_connection.send(audio_bytes)

            elif data["event"] == "stop":
                print("[Twilio] Stream stopped")
                break

    except Exception as e:
        print(f"[Twilio] Error: {e}")

    finally:
        if stt_connection:
            await stt.close_stream(stt_connection)
        print("[Twilio] WebSocket closed")
