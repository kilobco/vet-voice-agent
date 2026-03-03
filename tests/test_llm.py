import pytest
from unittest.mock import patch, MagicMock


@patch("app.llm.agent.retrieve")
@patch("app.llm.agent.client")
def test_answer_returns_string(mock_client, mock_retrieve):
    mock_retrieve.return_value = [
        {"title": "Hours", "body": "We are open Monday to Friday 9am-5pm."}
    ]
    mock_content = MagicMock()
    mock_content.text = "We are open Monday to Friday 9am to 5pm."
    mock_client.messages.create.return_value.content = [mock_content]

    from app.llm.agent import answer

    response = answer("What are your hours?")
    assert isinstance(response, str)
    assert len(response) > 0


@patch("app.llm.agent.retrieve")
@patch("app.llm.agent.client")
def test_answer_with_history(mock_client, mock_retrieve):
    mock_retrieve.return_value = []
    mock_content = MagicMock()
    mock_content.text = "I'm sorry, I don't have that information. Please call us directly."
    mock_client.messages.create.return_value.content = [mock_content]

    from app.llm.agent import answer

    history = [
        {"role": "user",      "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    response = answer("Do you offer emergency services?", conversation_history=history)
    assert isinstance(response, str)
