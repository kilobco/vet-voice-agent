import re
import asyncio
import anthropic
from datetime import datetime

from app.rag.retriever    import RAGRetriever
from app.booking.tools    import BookingTools, TOOL_DEFINITIONS


def _build_system_prompt(caller_phone: str = None) -> str:
    today      = datetime.now().strftime("%A, %B %d, %Y")
    phone_line = f"The caller's phone number is {caller_phone}." if caller_phone else ""
    return f"""You are a warm, friendly receptionist at Dr. Alexander's veterinary clinic. You are speaking on the phone — everything you say will be heard, not read.

Today is {today}. {phone_line}

HOW TO SPEAK:
- Talk the way a real person does on the phone. Use natural, flowing sentences.
- Use contractions — say "I'll", "we've", "that's", "you're" instead of the formal versions.
- Keep responses short. One or two sentences at a time, then pause and let the caller respond.
- Never use lists, bullet points, or numbered steps — just talk naturally.
- Avoid robotic phrases like "Certainly!", "Absolutely!", "Of course!" — just respond naturally.
- Do NOT greet the caller or introduce the clinic — they've already been welcomed.

BOOKING APPOINTMENTS:
When someone wants to book, work through it naturally in conversation — one question at a time.
- First call lookup_caller with their phone number to check if they're in the system.
- If they're new, ask for their name, then call create_caller.
- Call get_pets to check their pets on file, or ask about their pet and call create_pet.
- Ask what's going on with the pet to figure out which doctor fits best, then call get_doctors.
- Ask what day works for them, then call get_available_slots.
- Offer a couple of time options — don't read out the whole list.
- Once they pick a time, confirm the details naturally in one sentence, then call book_appointment.

If you don't know the answer to something clinical, just say you'll have someone from the team call them back."""


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
