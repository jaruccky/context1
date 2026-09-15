"""LLMAgentPolicy / LLMAnswerGenerator against a fake OpenAI-compatible client --
no live vLLM/OpenAI endpoint needed, just something duck-typing the response shape.
"""
from dataclasses import dataclass

from context_agent.agent.policy import LLMAgentPolicy
from context_agent.answering.generator import LLMAnswerGenerator
from context_agent.evidence.models import EvidencePack


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return _Response(choices=[_Choice(message=_Message(content=self.content))])


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def test_llm_agent_policy_returns_raw_completion_and_passes_model():
    fake_client = _FakeClient('{"thought": "search now", "tool": "search_corpus", "arguments": {"query": "x"}}')
    policy = LLMAgentPolicy(fake_client, model="Qwen/Qwen3-8B", temperature=0.1, max_tokens=64)

    raw = policy.next_action("query", "context", ["search_corpus"], step=1)

    assert "search_corpus" in raw
    call = fake_client.chat.completions.last_call
    assert call["model"] == "Qwen/Qwen3-8B"
    assert call["temperature"] == 0.1
    assert call["messages"][0]["role"] == "system"


def test_llm_agent_policy_handles_empty_content():
    fake_client = _FakeClient(None)
    policy = LLMAgentPolicy(fake_client, model="m")
    raw = policy.next_action("q", "ctx", [], step=1)
    assert raw == ""


def test_llm_answer_generator_uses_evidence_pack_text():
    fake_client = _FakeClient("Paris is the capital of France.")
    generator = LLMAnswerGenerator(fake_client, model="m")
    pack = EvidencePack(query="What is the capital of France?", evidence=[])

    answer = generator.generate(pack)

    assert answer == "Paris is the capital of France."
    call = fake_client.chat.completions.last_call
    assert "capital of France" in call["messages"][1]["content"]
