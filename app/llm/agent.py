import anthropic
from app.rag.retriever import RAGRetriever


SYSTEM_PROMPT = """
You are a helpful voice assistant for Dr. Alexander's veterinary clinic.
The caller has already been greeted — do NOT say hello or introduce the clinic again.
Answer the caller's question directly and concisely.
Keep responses short, clear, and conversational — they will be spoken out loud.
Do not use bullet points, markdown, or lists. Speak in natural sentences.
If you don't know the answer, politely say you will connect them with a staff member.
"""


class LLMAgent:
    def __init__(self, anthropic_api_key: str, rag: RAGRetriever):
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.rag    = rag

    async def ask(self, question: str, conversation_history: list = None) -> str:
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
                if any(buffer.rstrip().endswith(p) for p in (".", "?", "!", "...")):
                    sentence = buffer.strip()
                    if sentence:
                        print(f"[LLM] Chunk: {sentence}")
                        yield sentence
                    buffer = ""

        if buffer.strip():
            print(f"[LLM] Chunk: {buffer.strip()}")
            yield buffer.strip()
