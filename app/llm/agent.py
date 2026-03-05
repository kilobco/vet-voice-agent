import re
import asyncio
import anthropic
from datetime import datetime

from app.rag.retriever    import RAGRetriever
from app.booking.tools    import BookingTools, TOOL_DEFINITIONS


def _build_system_prompt(caller_phone: str = None) -> str:
    today      = datetime.now().strftime("%A, %B %d, %Y")
    phone_line = f"The caller's phone number is {caller_phone}." if caller_phone else ""
    return f"""You are a helpful voice assistant for Dr. Alexander's veterinary clinic.
The caller has already been greeted — do NOT say hello or introduce the clinic again.
Answer the caller's question directly and concisely.
Keep responses short, clear, and conversational — they will be spoken out loud.
Do not use bullet points, markdown, or lists. Speak in natural sentences.
Today is {today}. {phone_line}

You can answer general veterinary questions AND book appointments.

When a caller wants to book an appointment, follow these steps in order:
1. Call lookup_caller with their phone number to check if they are in the system.
2. If not found, ask for their first and last name, then call create_caller.
3. Call get_pets to see their pets on file. If they have pets, ask which one. If none, ask for pet details and call create_pet.
4. Ask the reason for the visit to determine which specialist is needed.
5. Call get_doctors to find a suitable doctor.
6. Ask for the caller's preferred date, then call get_available_slots.
7. Suggest available times and let the caller pick one.
8. Confirm all details out loud (pet, doctor, date, time, reason), then call book_appointment.

If you don't know the answer to a clinical question, politely say you will connect them with a staff member."""


def _split_sentences(text: str) -> list[str]:
    # Protect common abbreviations from being treated as sentence ends
    protected = re.sub(
        r'\b(Dr|Mr|Mrs|Ms|Prof|St|vs|etc)\.\s+',
        lambda m: m.group(0).replace('. ', '<<DOT>>'),
        text,
    )
    parts = re.split(r'(?<=[.!?])\s+', protected.strip())
    return [p.replace('<<DOT>>', '. ').strip() for p in parts if p.strip()]


class LLMAgent:
    def __init__(self, anthropic_api_key: str, rag: RAGRetriever, booking: BookingTools):
        self.client  = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.rag     = rag
        self.booking = booking

    async def ask_stream(self, question: str, full_messages: list, caller_phone: str = None):
        """
        full_messages is the mutable conversation history for the entire call.
        Tool calls and results are stored in it so the agent never re-does
        lookups it already performed earlier in the call.
        """
        docs    = await self.rag.retrieve(question)
        context = self.rag.format_context(docs)

        user_content = f"""Context from veterinary knowledge base:
{context}

Customer question: {question}

Answer the customer's question. If they want to book an appointment, use the available tools."""

        full_messages.append({"role": "user", "content": user_content})
        system = _build_system_prompt(caller_phone)

        # ── Tool-calling loop ─────────────────────────────────────────────────
        while True:
            response = await self.client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 500,
                system     = system,
                messages   = full_messages,
                tools      = TOOL_DEFINITIONS,
            )

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]

                for b in tool_blocks:
                    print(f"[Tool] {b.name}({b.input})")

                # Run all tools in this turn in parallel
                raw_results = await asyncio.gather(
                    *[self.booking.execute(b.name, b.input) for b in tool_blocks]
                )

                tool_results = []
                for block, result in zip(tool_blocks, raw_results):
                    print(f"[Tool] Result: {result}")
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
                # Store full tool exchange in history so next turn remembers it
                full_messages.append({"role": "assistant", "content": response.content})
                full_messages.append({"role": "user",      "content": tool_results})

            else:
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                full_messages.append({"role": "assistant", "content": final_text})

                for sentence in _split_sentences(final_text):
                    print(f"[LLM] Chunk: {sentence}")
                    yield sentence
                break
