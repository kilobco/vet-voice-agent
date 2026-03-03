from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)


class DeepgramSTT:
    def __init__(self, api_key: str):
        config        = DeepgramClientOptions(verbose=False)
        self.client   = DeepgramClient(api_key, config)
        self.model    = "nova-3"

    async def transcribe_stream(self, on_transcript):
        """
        Real-time streaming transcription from Twilio audio.
        on_transcript: async callback that receives transcript text.
        Returns the live connection — pipe Twilio audio chunks into it.
        """
        connection = self.client.listen.asynclive.v("1")

        async def on_message(self_event, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if sentence.strip() and result.is_final:
                await on_transcript(sentence)

        async def on_error(self_event, error, **kwargs):
            print(f"[STT] Error: {error}")

        async def on_utterance_end(self_event, utterance_end, **kwargs):
            print("[STT] Utterance ended — user finished speaking")

        connection.on(LiveTranscriptionEvents.Transcript,   on_message)
        connection.on(LiveTranscriptionEvents.Error,        on_error)
        connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)

        options = LiveOptions(
            model           = self.model,
            smart_format    = True,
            language        = "en-US",
            encoding        = "mulaw",   # Twilio audio format
            channels        = 1,
            sample_rate     = 8000,      # Twilio sample rate
            interim_results = True,      # partial results while speaking
            utterance_end_ms= "500",     # end of utterance after 0.5s silence
            vad_events      = True,      # voice activity detection
        )

        await connection.start(options)
        return connection

    async def close_stream(self, connection) -> None:
        """Cleanly close the stream when Twilio call ends."""
        await connection.finish()
