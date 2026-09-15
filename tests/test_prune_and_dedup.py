import json

from context_agent.agent.harness import AgentHarness, HarnessConfig
from context_agent.agent.policy import MockAgentPolicy


def test_prune_tool_keeps_full_trajectory(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    script = [
        json.dumps({"thought": "search", "tool": "search_corpus", "arguments": {"query": "Ada Lovelace", "top_k": 3}}),
        json.dumps({"thought": "prune first hit", "tool": "prune_chunks", "arguments": {"chunk_ids": ["ada-lovelace::0"]}}),
        json.dumps({"thought": "done", "finish": True}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=5))
    result = harness.run("Who was Ada Lovelace?")

    cm = result.context_manager
    assert "ada-lovelace::0" not in cm.visible_chunk_ids
    assert "ada-lovelace::0" in cm.state.retrieved_chunks  # never lost from full trajectory
    assert "ada-lovelace::0" not in result.evidence_pack.chunk_ids()


def test_duplicate_tool_call_is_detected_and_not_reexecuted(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    call = {"thought": "search", "tool": "search_corpus", "arguments": {"query": "Ada Lovelace", "top_k": 3}}
    script = [json.dumps(call), json.dumps(call), json.dumps({"thought": "done", "finish": True})]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=5))
    result = harness.run("Who was Ada Lovelace?")

    state = result.context_manager.state
    assert state.tool_call_count == 1
    assert state.duplicate_call_count == 1
    assert state.steps[1].observation.get("duplicate") is True
