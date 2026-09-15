"""Typer CLI entrypoint: `context-agent ask "..."`."""
from __future__ import annotations

import json

import typer

from context_agent.agent.harness import AgentHarness
from context_agent.config import (
    build_answer_generator,
    build_harness_config,
    build_policy,
    build_retriever_from_config,
    load_corpus,
    load_yaml,
)

app = typer.Typer(help="Context-1: agentic retrieval and context-management agent.")


@app.callback()
def _main() -> None:
    """Context-1: agentic retrieval and context-management agent."""


@app.command()
def ask(
    query: str,
    data_config: str = "configs/default.yaml",
    agent_config: str = "configs/agent.yaml",
    retrieval_config: str = "configs/retrieval.yaml",
    answer: bool = typer.Option(False, help="Also run the downstream AnswerGenerator on the result."),
) -> None:
    """Run Context-1 on a question and print the resulting EvidencePack (not an answer)."""
    data_cfg = load_yaml(data_config)
    retrieval_cfg = load_yaml(retrieval_config)
    agent_cfg = load_yaml(agent_config)

    corpus = load_corpus(data_cfg)
    retriever = build_retriever_from_config(corpus, retrieval_cfg)
    policy = build_policy(agent_cfg, query=query)
    harness = AgentHarness(policy, corpus, retriever, build_harness_config(agent_cfg))

    result = harness.run(query)
    typer.echo(f"stop_reason: {result.stop_reason}")
    typer.echo(json.dumps(result.evidence_pack.model_dump(), indent=2, ensure_ascii=False))

    if answer:
        answer_generator = build_answer_generator({"answer_generator": "mock"})
        typer.echo(f"\nAnswer: {answer_generator.generate(result.evidence_pack)}")


if __name__ == "__main__":
    app()
