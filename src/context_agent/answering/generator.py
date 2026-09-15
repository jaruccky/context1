"""Downstream answer model: EvidencePack -> final answer.

Deliberately decoupled from context_agent.agent — it only ever imports EvidencePack,
never the harness/trajectory internals, so retrieval quality and answer quality can
be evaluated independently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from context_agent.evidence.models import EvidencePack

ANSWER_SYSTEM_PROMPT = (
    "Answer the question using only the evidence provided. If the evidence is "
    "insufficient, say you don't have enough information. Be concise."
)


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, evidence_pack: EvidencePack) -> str: ...


class MockAnswerGenerator(AnswerGenerator):
    """Extractive stub: no model, no network. Used in tests and CI demos."""

    def generate(self, evidence_pack: EvidencePack) -> str:
        if not evidence_pack.evidence:
            return "I don't have enough evidence to answer."
        top = evidence_pack.evidence[0]
        return top.text[:300]


class LLMAnswerGenerator(AnswerGenerator):
    """OpenAI-compatible chat completion answer generator."""

    def __init__(self, client, model: str, temperature: float = 0.0, max_tokens: int = 256):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, evidence_pack: EvidencePack) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": evidence_pack.render_text()},
            ],
        )
        return response.choices[0].message.content or ""
