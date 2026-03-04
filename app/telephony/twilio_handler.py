import asyncio
import base64
import json
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect

from app.stt.deepgram_stt import DeepgramSTT
from app.llm.agent         import LLMAgent
from app.tts.deepgram_tts  import DeepgramTTS
from app.rag.retriever     import RAGRetriever
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
rag   = RAGRetriever(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY, jina_api_key=JINA_API_KEY)
stt   = DeepgramSTT(api_key=DEEPGRAM_API_KEY)
tts   = DeepgramTTS(api_key=DEEPGRAM_API_KEY)
agent = LLMAgent(anthropic_api_key=ANTHROPIC_API_KEY, rag=rag)


# ── Step 1: Twilio hits this endpoint when a call comes in ────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Return TwiML that opens a media stream — no <Say> here to avoid
    the greeting audio being echoed into Deepgram STT.
    The greeting is sent as TTS over the WebSocket once the stream opens.
    """
    response = VoiceResponse()
    connect  = Connect()
    connect.stream(url=f"wss://{request.headers['host']}/media-stream")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


# ── Step 2: Twilio streams audio to this WebSocket ───────────────────────────
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio sends audio chunks here as base64-encoded mulaw.
    Flow: greeting TTS → wait for mark → STT → LLM → TTS → Twilio.
    """
    await websocket.accept()
    print("[Twilio] WebSocket connected")

    stream_sid           = None
    stt_connection       = None
    response_task        = None
    conversation_history = []
    ready_to_listen      = False  # True only after the greeting has finished playing

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
        """Play the opening greeting via TTS, then mark it so we know when it's done."""
        print("[Agent] Sending greeting...")
        async for chunk in tts.synthesize_and_stream(GREETING):
            await send_audio(chunk)
        await send_mark("greeting_done")

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def _respond(text: str):
        nonlocal conversation_history
        full_answer = ""
        try:
            async for sentence in agent.ask_stream(text, list(conversation_history)):
                full_answer += sentence
                async for chunk in tts.synthesize_and_stream(sentence):
                    await send_audio(chunk)
            await send_mark("response_done")
            conversation_history = conversation_history + [
                {"role": "user",      "content": text},
                {"role": "assistant", "content": full_answer},
            ]
        except asyncio.CancelledError:
            pass  # barge-in: a new transcript arrived
        except Exception as e:
            print(f"[Agent] Error in pipeline: {e}")
            # Play fallback so the caller never hears dead silence
            try:
                async for chunk in tts.synthesize_and_stream(FALLBACK):
                    await send_audio(chunk)
            except Exception:
                pass

    async def handle_transcript(text: str):
        nonlocal response_task, ready_to_listen
        if not ready_to_listen:
            return  # ignore anything picked up while greeting was playing
        print(f"[STT→LLM] Received: {text}")
        if response_task and not response_task.done():
            response_task.cancel()
        response_task = asyncio.create_task(_respond(text))

    # ── WebSocket message loop ────────────────────────────────────────────────

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data["event"] == "start":
                stream_sid     = data["start"]["streamSid"]
                print(f"[Twilio] Stream started: {stream_sid}")
                stt_connection = await stt.transcribe_stream(handle_transcript)
                asyncio.create_task(send_greeting())   # play greeting, then mark

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
