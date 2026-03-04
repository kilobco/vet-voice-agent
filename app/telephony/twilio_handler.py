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
    response.say("Thank you for calling Dr. Alexander's veterinary clinic. Please hold while we connect you.")

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

    stream_sid           = None
    stt_connection       = None
    keepalive_task       = None
    response_task        = None
    conversation_history = []

    async def send_audio(audio_chunk: bytes):
        audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
        await websocket.send_text(json.dumps({
            "event":     "media",
            "streamSid": stream_sid,
            "media":     {"payload": audio_b64},
        }))

    async def send_mark(label: str):
        """Notify Twilio when audio playback reaches this point."""
        await websocket.send_text(json.dumps({
            "event":     "mark",
            "streamSid": stream_sid,
            "mark":      {"name": label},
        }))

    async def _respond(text: str):
        """Run the full LLM → TTS → Twilio pipeline for one transcript."""
        nonlocal conversation_history
        full_answer = ""
        try:
            async for sentence in agent.ask_stream(text, list(conversation_history)):
                full_answer += sentence
                async for audio_chunk in tts.synthesize_and_stream(sentence):
                    await send_audio(audio_chunk)
            await send_mark("end_of_response")
            # Update history after a full successful response
            conversation_history = conversation_history + [
                {"role": "user",      "content": text},
                {"role": "assistant", "content": full_answer},
            ]
        except asyncio.CancelledError:
            pass  # barge-in — new transcript arrived, this response was cancelled

    async def handle_transcript(text: str):
        nonlocal response_task
        print(f"[STT→LLM] Received: {text}")
        # Cancel any in-progress response (basic barge-in support)
        if response_task and not response_task.done():
            response_task.cancel()
        response_task = asyncio.create_task(_respond(text))

    async def run_keepalive():
        """Send keepalive every 5s to prevent Deepgram disconnecting during silence."""
        while True:
            await asyncio.sleep(5)
            try:
                await stt.keep_alive(stt_connection)
            except Exception:
                break

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data["event"] == "start":
                stream_sid     = data["start"]["streamSid"]
                print(f"[Twilio] Stream started: {stream_sid}")
                stt_connection = await stt.transcribe_stream(handle_transcript)
                keepalive_task = asyncio.create_task(run_keepalive())

            elif data["event"] == "media":
                if stt_connection:
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    await stt_connection.send(audio_bytes)

            elif data["event"] == "mark":
                print(f"[Twilio] Mark received: {data['mark']['name']}")

            elif data["event"] == "stop":
                print("[Twilio] Stream stopped")
                break

    except Exception as e:
        print(f"[Twilio] Error: {e}")

    finally:
        if keepalive_task:
            keepalive_task.cancel()
        if response_task and not response_task.done():
            response_task.cancel()
        if stt_connection:
            await stt.close_stream(stt_connection)
        print("[Twilio] WebSocket closed")
