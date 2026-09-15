import pytest

from context_agent.context.manager import ContextManager
from context_agent.tools import grep, read
from context_agent.tools.errors import ToolError


def test_grep_literal_match(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    cm = ContextManager(query="q", token_budget=100_000)
    args = grep.GrepArguments(pattern="Analytical Engine", regex=False)

    observation = grep.execute(args, corpus, cm, step=1)

    assert observation["results"]
    assert any(r["document_id"] == "analytical-engine" for r in observation["results"])
    assert "analytical-engine::0" in cm.visible_chunk_ids


def test_grep_regex_match(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    cm = ContextManager(query="q", token_budget=100_000)
    args = grep.GrepArguments(pattern=r"\b18\d{2}\b", regex=True, max_results=20)

    observation = grep.execute(args, corpus, cm, step=1)
    assert observation["results"]


def test_grep_invalid_regex_raises_tool_error(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    cm = ContextManager(query="q", token_budget=100_000)
    args = grep.GrepArguments(pattern="(unclosed", regex=True)
    with pytest.raises(ToolError):
        grep.execute(args, corpus, cm, step=1)


def test_grep_unknown_document_id_raises_tool_error(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    cm = ContextManager(query="q", token_budget=100_000)
    args = grep.GrepArguments(pattern="engine", document_ids=["does-not-exist"])
    with pytest.raises(ToolError):
        grep.execute(args, corpus, cm, step=1)


def test_read_document_full(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    cm = ContextManager(query="q", token_budget=100_000)
    args = read.ReadArguments(document_id="charles-babbage")

    observation = read.execute(args, corpus, cm, step=1)
    assert observation["document_id"] == "charles-babbage"
    assert observation["results"]
    assert "charles-babbage::0" in cm.visible_chunk_ids


def test_read_document_unknown_raises_tool_error(synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    cm = ContextManager(query="q", token_budget=100_000)
    args = read.ReadArguments(document_id="nope")
    with pytest.raises(ToolError):
        read.execute(args, corpus, cm, step=1)


def test_read_document_invalid_range_raises_validation_error():
    with pytest.raises(Exception):
        read.ReadArguments(document_id="doc", start_chunk=5, end_chunk=1)
