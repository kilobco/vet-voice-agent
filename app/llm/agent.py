import anthropic
from app.rag.retriever import RAGRetriever


SYSTEM_PROMPT = """
You are a helpful voice assistant for a veterinary clinic.
You answer customer questions about the pets and questions about the pet's health and freaqently asked questions.
Keep responses short, clear, and conversational — they will be spoken out loud.
Do not use bullet points, markdown, or lists. Speak in natural sentences.
If you don't know the answer, politely say you'll connect them with staff.
"""


class LLMAgent:
    def __init__(self, anthropic_api_key: str, rag: RAGRetriever):
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.rag    = rag

    async def ask(self, question: str) -> str:
        """
        1. Retrieve relevant context from RAG
        2. Send question + context to LLM
        3. Return spoken response
        """
        # Step 1 — retrieve relevant docs
        docs    = self.rag.retrieve(question)
        context = self.rag.format_context(docs)

        # Step 2 — build prompt with context
        user_message = f"""
Context from veterinary knowledge base:
{context}

Customer question: {question}

Answer the customer's question based on the context above.
"""

        # Step 3 — call LLM
        response = await self.client.messages.create(
            model      = "claude-sonnet-4-6",
            max_tokens = 150,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        print(f"[LLM] Question: {question}")
        print(f"[LLM] Answer: {answer}")
        return answer

    async def ask_stream(self, question: str):
        """
        Like ask(), but yields complete sentence chunks as they arrive.
        Allows TTS to start on the first sentence while LLM generates the rest.
        """
        docs    = self.rag.retrieve(question)
        context = self.rag.format_context(docs)

        user_message = f"""
Context from veterinary knowledge base:
{context}

Customer question: {question}

Answer the customer's question based on the context above.
"""

        buffer = ""
        async with self.client.messages.stream(
            model      = "claude-sonnet-4-6",
            max_tokens = 150,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                buffer += text
                # flush buffer to TTS whenever a sentence boundary is hit
                if any(buffer.rstrip().endswith(p) for p in (".", "?", "!", "...", ",")):
                    sentence = buffer.strip()
                    if sentence:
                        print(f"[LLM] Chunk: {sentence}")
                        yield sentence
                    buffer = ""

        # yield any remaining text
        if buffer.strip():
            print(f"[LLM] Chunk: {buffer.strip()}")
            yield buffer.strip()
