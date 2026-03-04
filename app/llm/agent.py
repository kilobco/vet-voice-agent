import anthropic
from app.rag.retriever import RAGRetriever


SYSTEM_PROMPT = """
You are a helpful voice assistant for Dr. Alexander's veterinary clinic.
You answer customer questions about pet health and frequently asked questions about the clinic.
Keep responses short, clear, and conversational — they will be spoken out loud.
Do not use bullet points, markdown, or lists. Speak in natural sentences.
Do not re-introduce yourself on every response — only greet the caller once at the start of the call.
If you don't know the answer, politely say you'll connect them with a staff member.
"""


class LLMAgent:
    def __init__(self, anthropic_api_key: str, rag: RAGRetriever):
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.rag    = rag

    async def ask(self, question: str, conversation_history: list = None) -> str:
        """
        1. Retrieve relevant context from RAG
        2. Send question + context to LLM (with conversation history)
        3. Return spoken response
        """
        if conversation_history is None:
            conversation_history = []

        docs    = await self.rag.retrieve(question)
        context = self.rag.format_context(docs)

        user_message = f"""Context from veterinary knowledge base:
{context}

Customer question: {question}

Answer the customer's question based on the context above."""

        messages = conversation_history + [{"role": "user", "content": user_message}]

        response = await self.client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 300,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        )

        answer = response.content[0].text
        print(f"[LLM] Question: {question}")
        print(f"[LLM] Answer: {answer}")
        return answer

    async def ask_stream(self, question: str, conversation_history: list = None):
        """
        Like ask(), but yields complete sentence chunks as they arrive.
        Allows TTS to start on the first sentence while LLM generates the rest.
        """
        if conversation_history is None:
            conversation_history = []

        docs    = await self.rag.retrieve(question)
        context = self.rag.format_context(docs)

        user_message = f"""Context from veterinary knowledge base:
{context}

Customer question: {question}

Answer the customer's question based on the context above."""

        messages = conversation_history + [{"role": "user", "content": user_message}]

        buffer = ""
        async with self.client.messages.stream(
            model      = "claude-sonnet-4-6",
            max_tokens = 300,
            system     = SYSTEM_PROMPT,
            messages   = messages,
        ) as stream:
            async for text in stream.text_stream:
                buffer += text
                # flush on sentence-ending punctuation only (not commas)
                if any(buffer.rstrip().endswith(p) for p in (".", "?", "!", "...")):
                    sentence = buffer.strip()
                    if sentence:
                        print(f"[LLM] Chunk: {sentence}")
                        yield sentence
                    buffer = ""

        # yield any remaining text
        if buffer.strip():
            print(f"[LLM] Chunk: {buffer.strip()}")
            yield buffer.strip()
