"""Wiring: turn plain YAML config dicts into the objects the rest of the code needs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from context_agent.agent.harness import HarnessConfig
from context_agent.agent.policy import AgentPolicy, LLMAgentPolicy, MockAgentPolicy
from context_agent.answering.generator import AnswerGenerator, LLMAnswerGenerator, MockAnswerGenerator
from context_agent.data.corpus import Corpus
from context_agent.retrieval.hybrid import HybridRetriever, RetrievalConfig
from context_agent.retrieval.indexing import build_retriever


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_corpus(data_cfg: dict[str, Any]) -> Corpus:
    return Corpus.from_jsonl(
        data_cfg["corpus_path"],
        chunk_size=data_cfg.get("chunk_size", 400),
        overlap=data_cfg.get("chunk_overlap", 50),
    )


def build_retrieval_config(raw: dict[str, Any]) -> RetrievalConfig:
    return RetrievalConfig(
        use_bm25=raw.get("use_bm25", True),
        use_dense=raw.get("use_dense", True),
        use_reranker=raw.get("use_reranker", False),
        bm25_top_k=raw.get("bm25_top_k", 20),
        dense_top_k=raw.get("dense_top_k", 20),
        rrf_k=raw.get("rrf_k", 60),
        rerank_candidates=raw.get("rerank_candidates", 20),
        final_top_k=raw.get("final_top_k", 5),
    )


def build_retriever_from_config(corpus: Corpus, raw: dict[str, Any]) -> HybridRetriever:
    return build_retriever(
        corpus,
        build_retrieval_config(raw),
        dense_model=raw.get("dense_model", "hashing"),
        reranker_model=raw.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        qdrant_path=raw.get("qdrant_path"),
    )


def build_harness_config(raw: dict[str, Any]) -> HarnessConfig:
    return HarnessConfig(
        max_steps=raw.get("max_steps", 8),
        max_tool_calls=raw.get("max_tool_calls", 6),
        token_budget=raw.get("token_budget", 3000),
        default_top_k=raw.get("default_top_k", 5),
    )


def _build_llm_client(raw_llm: dict[str, Any]):
    from openai import OpenAI

    return OpenAI(base_url=raw_llm.get("base_url"), api_key=raw_llm.get("api_key", "EMPTY"))


def build_policy(raw: dict[str, Any], query: str | None = None, script: list[str] | None = None) -> AgentPolicy:
    """`raw` is the parsed agent.yaml. For policy=mock with no explicit script, a
    minimal single-search-then-finish script is generated from `query` so scripts
    have something concrete to demo without needing a live LLM endpoint.
    """
    kind = raw.get("policy", "mock")
    if kind == "mock":
        if script is not None:
            return MockAgentPolicy(script)
        if query:
            top_k = raw.get("default_top_k", 5)
            default_script = [
                json.dumps(
                    {
                        "thought": "Search broadly for evidence relevant to the question.",
                        "tool": "search_corpus",
                        "arguments": {"query": query, "top_k": top_k},
                    }
                ),
                json.dumps({"thought": "Evidence gathered, finishing.", "finish": True}),
            ]
            return MockAgentPolicy(default_script)
        return MockAgentPolicy(['{"thought": "no scripted actions", "finish": true}'])
    if kind == "llm":
        llm = raw.get("llm", {})
        client = _build_llm_client(llm)
        return LLMAgentPolicy(
            client,
            model=llm.get("model", "Qwen/Qwen3-8B"),
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 512),
        )
    raise ValueError(f"unknown policy kind: {kind}")


def build_answer_generator(raw: dict[str, Any], llm_raw: dict[str, Any] | None = None) -> AnswerGenerator:
    kind = raw.get("answer_generator", "mock")
    if kind == "mock":
        return MockAnswerGenerator()
    if kind == "llm":
        llm = llm_raw or raw.get("llm", {})
        client = _build_llm_client(llm)
        return LLMAnswerGenerator(
            client,
            model=llm.get("model", "Qwen/Qwen3-8B"),
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 256),
        )
    raise ValueError(f"unknown answer_generator kind: {kind}")
