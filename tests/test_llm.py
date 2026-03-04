from unittest.mock import AsyncMock, MagicMock, patch
from app.llm.agent import LLMAgent


def _make_agent():
    mock_rag = MagicMock()
    mock_rag.retrieve      = AsyncMock(return_value=[
        {"title": "Hours", "body": "We are open Monday to Friday 9am-5pm."}
    ])
    mock_rag.format_context = MagicMock(return_value="[1] Hours\nWe are open Monday to Friday 9am-5pm.")
    return LLMAgent(anthropic_api_key="test_key", rag=mock_rag)


async def test_ask_returns_string():
    agent = _make_agent()

    mock_content          = MagicMock()
    mock_content.text     = "We are open Monday to Friday 9am to 5pm."
    mock_response         = MagicMock()
    mock_response.content = [mock_content]

    agent.client = MagicMock()
    agent.client.messages.create = AsyncMock(return_value=mock_response)

    result = await agent.ask("What are your hours?")

    assert isinstance(result, str)
    assert len(result) > 0


async def test_ask_passes_conversation_history():
    agent   = _make_agent()
    history = [
        {"role": "user",      "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]

    mock_content          = MagicMock()
    mock_content.text     = "Yes, we handle emergencies."
    mock_response         = MagicMock()
    mock_response.content = [mock_content]

    agent.client = MagicMock()
    agent.client.messages.create = AsyncMock(return_value=mock_response)

    result = await agent.ask("Do you offer emergency services?", conversation_history=history)

    assert isinstance(result, str)
    # History should be prepended — check messages arg
    call_kwargs = agent.client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "user", "content": "Hi"}
    assert messages[1] == {"role": "assistant", "content": "Hello! How can I help?"}


async def test_ask_stream_yields_sentences():
    agent = _make_agent()

    async def fake_text_stream():
        for token in ["Hello", ".", " How", " can", " I", " help", "?"]:
            yield token

    mock_stream = MagicMock()
    mock_stream.text_stream            = fake_text_stream()
    mock_stream.__aenter__             = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__              = AsyncMock(return_value=False)
    agent.client = MagicMock()
    agent.client.messages.stream      = MagicMock(return_value=mock_stream)

    chunks = []
    async for chunk in agent.ask_stream("Hi"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


async def test_ask_stream_no_comma_flush():
    """Commas must not trigger a TTS flush."""
    agent = _make_agent()

    async def fake_text_stream():
        for token in ["Hello", ",", " how", " are", " you", "?"]:
            yield token

    mock_stream = MagicMock()
    mock_stream.text_stream       = fake_text_stream()
    mock_stream.__aenter__        = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__         = AsyncMock(return_value=False)
    agent.client = MagicMock()
    agent.client.messages.stream = MagicMock(return_value=mock_stream)

    chunks = []
    async for chunk in agent.ask_stream("Hi"):
        chunks.append(chunk)

    # Should be one chunk ending with "?", not split on ","
    assert len(chunks) == 1
    assert chunks[0].endswith("?")
