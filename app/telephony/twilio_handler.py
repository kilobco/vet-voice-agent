import asyncio
import base64
import json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect

from app.stt.deepgram_stt  import DeepgramSTT
from app.llm.agent         import LLMAgent
from app.tts.deepgram_tts  import DeepgramTTS
from app.rag.retriever     import RAGRetriever
from app.booking.tools     import BookingTools
from config.settings       import (
    DEEPGRAM_API_KEY,
    ANTHROPIC_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
    JINA_API_KEY,
)

app = FastAPI()

GREETING = "Hello! Thank you for calling Dr. Alexander's veterinary clinic. How can I help you today?"
FALLBACK  = "I'm sorry, I had a little trouble with that. Could you please repeat your question?"


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


# ── Initialize all services ───────────────────────────────────────────────────
rag     = RAGRetriever(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY, jina_api_key=JINA_API_KEY)
booking = BookingTools(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)
stt     = DeepgramSTT(api_key=DEEPGRAM_API_KEY)
tts     = DeepgramTTS(api_key=DEEPGRAM_API_KEY)
agent   = LLMAgent(anthropic_api_key=ANTHROPIC_API_KEY, rag=rag, booking=booking)


# ── Step 1: Twilio hits this endpoint when a call comes in ────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Return TwiML that opens a media stream.
    The caller's phone number is passed as a custom parameter so the
    WebSocket handler can give it to the agent for caller lookup.
    """
    form        = await request.form()
    caller_phone = form.get("From", "")

    response = VoiceResponse()
    connect  = Connect()
    stream   = connect.stream(url=f"wss://{request.headers['host']}/media-stream")
    stream.parameter(name="caller_phone", value=caller_phone)
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


# ── Step 2: Twilio streams audio to this WebSocket ───────────────────────────
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio sends audio chunks here as base64-encoded mulaw.
    Flow: greeting TTS → wait for mark → STT → LLM (with tools) → TTS → Twilio.
    """
    await websocket.accept()
    print("[Twilio] WebSocket connected")

    stream_sid    = None
    caller_phone  = None
    stt_connection = None
    response_task  = None
    full_messages  = []   # full history including tool calls — persists entire call
    ready_to_listen = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def send_audio(audio_chunk: bytes):
        await websocket.send_text(json.dumps({
            "event":     "media",
            "streamSid": stream_sid,
            "media":     {"payload": base64.b64encode(audio_chunk).decode("utf-8")},
        }))

    async def send_mark(label: str):
        await websocket.send_text(json.dumps({
            "event":     "mark",
            "streamSid": stream_sid,
            "mark":      {"name": label},
        }))

    async def send_greeting():
        print("[Agent] Sending greeting...")
        async for chunk in tts.synthesize_and_stream(GREETING):
            await send_audio(chunk)
        await send_mark("greeting_done")

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def _respond(text: str):
        try:
            async for sentence in agent.ask_stream(
                text,
                full_messages,
                caller_phone=caller_phone,
            ):
                async for chunk in tts.synthesize_and_stream(sentence):
                    await send_audio(chunk)
            await send_mark("response_done")
        except asyncio.CancelledError:
            pass  # barge-in: a new transcript arrived
        except Exception as e:
            print(f"[Agent] Error in pipeline: {e}")
            try:
                async for chunk in tts.synthesize_and_stream(FALLBACK):
                    await send_audio(chunk)
            except Exception:
                pass

    async def handle_transcript(text: str):
        nonlocal response_task, ready_to_listen
        if not ready_to_listen:
            return
        print(f"[STT→LLM] Received: {text}")
        if response_task and not response_task.done():
            response_task.cancel()
        response_task = asyncio.create_task(_respond(text))

    # ── WebSocket message loop ────────────────────────────────────────────────

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data["event"] == "start":
                stream_sid   = data["start"]["streamSid"]
                caller_phone = data["start"].get("customParameters", {}).get("caller_phone")
                print(f"[Twilio] Stream started: {stream_sid} | caller: {caller_phone}")
                stt_connection = await stt.transcribe_stream(handle_transcript)
                asyncio.create_task(send_greeting())

            elif data["event"] == "media":
                if stt_connection:
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    await stt_connection.send(audio_bytes)

            elif data["event"] == "mark":
                label = data["mark"]["name"]
                print(f"[Twilio] Mark: {label}")
                if label == "greeting_done":
                    ready_to_listen = True
                    print("[Agent] Now listening for caller...")

            elif data["event"] == "stop":
                print("[Twilio] Stream stopped")
                break

    except Exception as e:
        print(f"[Twilio] Error: {e}")

    finally:
        if response_task and not response_task.done():
            response_task.cancel()
        if stt_connection:
            await stt.close_stream(stt_connection)
        print("[Twilio] WebSocket closed")
