"""Agent policies: turn (query, visible context) into a raw action string.

`next_action` returns the *raw* completion text rather than an already-parsed
AgentAction — that's what a real LLM actually produces, and it lets
`agent.actions.parse_action` (and its malformed-JSON handling) be exercised the same
way for both Mock and LLM-backed policies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from context_agent.agent.prompts import SYSTEM_PROMPT, render_user_turn


class AgentPolicy(ABC):
    @abstractmethod
    def next_action(self, query: str, context_text: str, available_tools: list[str], step: int) -> str:
        """Return raw model output for this turn (expected to contain a JSON action)."""


class MockAgentPolicy(AgentPolicy):
    """Deterministic, scripted policy. No GPU, no network — used in tests and demos.

    `script` is a list of raw JSON strings, one per call to `next_action`. If the
    script is exhausted, returns a `finish` action so the harness always terminates.
    """

    def __init__(self, script: list[str]):
        self.script = script
        self.calls = 0

    def next_action(self, query: str, context_text: str, available_tools: list[str], step: int) -> str:
        if self.calls < len(self.script):
            raw = self.script[self.calls]
        else:
            raw = '{"thought": "script exhausted", "finish": true}'
        self.calls += 1
        return raw


class LLMAgentPolicy(AgentPolicy):
    """OpenAI-compatible chat completion policy (e.g. vLLM serving Qwen3-8B).

    `client` must expose `.chat.completions.create(model=..., messages=...)` returning
    an object with `.choices[0].message.content` (the `openai` SDK client satisfies
    this, and so does any fake used in tests).
    """

    def __init__(self, client, model: str, temperature: float = 0.0, max_tokens: int = 512):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def next_action(self, query: str, context_text: str, available_tools: list[str], step: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_turn(context_text, available_tools)},
            ],
        )
        return response.choices[0].message.content or ""
