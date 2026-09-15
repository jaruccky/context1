import json

from context_agent.agent.harness import AgentHarness, HarnessConfig
from context_agent.agent.policy import MockAgentPolicy


def test_harness_runs_full_search_read_grep_prune_finish(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    script = [
        json.dumps({"thought": "search for Ada", "tool": "search_corpus", "arguments": {"query": "Ada Lovelace", "top_k": 3}}),
        json.dumps({"thought": "read Babbage doc", "tool": "read_document", "arguments": {"document_id": "charles-babbage"}}),
        json.dumps({"thought": "grep for exact term", "tool": "grep_corpus", "arguments": {"pattern": "Analytical Engine"}}),
        json.dumps({"thought": "drop the babbage chunk, not needed", "tool": "prune_chunks", "arguments": {"chunk_ids": ["charles-babbage::0"]}}),
        json.dumps({"thought": "enough evidence now", "finish": True}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=10, max_tool_calls=10))
    result = harness.run("Who designed the Analytical Engine that Ada Lovelace wrote about?", task_id="t1")

    assert result.stop_reason == "finished"
    assert len(result.context_manager.state.steps) == 5
    assert result.context_manager.state.tool_call_count == 4
    assert "charles-babbage::0" not in result.evidence_pack.chunk_ids()
    assert any(cid.startswith("ada-lovelace") for cid in result.evidence_pack.chunk_ids())
    # full trajectory keeps the pruned chunk even though evidence doesn't
    assert "charles-babbage::0" in result.context_manager.state.retrieved_chunks
    # every step captured its context snapshot for later SFT extraction
    assert all(step.context_snapshot for step in result.context_manager.state.steps)


def test_harness_handles_malformed_action_and_continues(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    script = [
        "this is not JSON at all",
        json.dumps({"thought": "recovered", "finish": True}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=5))
    result = harness.run("some question")

    assert result.stop_reason == "finished"
    assert result.context_manager.state.invalid_action_count == 1
    assert result.context_manager.state.steps[0].error is not None


def test_harness_falls_back_when_max_steps_exhausted(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    # policy never finishes; harness must still terminate and return a usable pack
    script = [json.dumps({"thought": "search", "tool": "search_corpus", "arguments": {"query": "Guido"}})] * 3
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=3, max_tool_calls=10))
    result = harness.run("Who created Python?")

    assert result.stop_reason == "max_steps"
    assert result.evidence_pack is not None


def test_harness_respects_max_tool_calls(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    script = [
        json.dumps({"thought": "s", "tool": "search_corpus", "arguments": {"query": "Guido"}}),
        json.dumps({"thought": "s", "tool": "search_corpus", "arguments": {"query": "Python"}}),
        json.dumps({"thought": "s", "tool": "search_corpus", "arguments": {"query": "Turing"}}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=10, max_tool_calls=1))
    result = harness.run("q")

    assert result.stop_reason == "max_tool_calls"
    assert result.context_manager.state.tool_call_count == 1


def test_harness_invalid_tool_arguments_are_recorded(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    script = [
        json.dumps({"thought": "bad args", "tool": "search_corpus", "arguments": {"top_k": "not-an-int"}}),
        json.dumps({"thought": "done", "finish": True}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=5))
    result = harness.run("q")

    assert result.context_manager.state.invalid_action_count == 1
    assert result.context_manager.state.tool_call_count == 0
