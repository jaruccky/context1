import json

from context_agent.agent.harness import AgentHarness, HarnessConfig
from context_agent.agent.policy import MockAgentPolicy
from context_agent.training.dataset import trajectories_to_sft_dataset, trajectory_to_sft_examples


def test_trajectory_to_sft_examples_uses_context_snapshots(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    script = [
        json.dumps({"thought": "search", "tool": "search_corpus", "arguments": {"query": "Guido van Rossum"}}),
        json.dumps({"thought": "done", "finish": True}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=5))
    result = harness.run("Who created Python?")

    trajectory = result.context_manager.state.model_dump(mode="json")
    examples = trajectory_to_sft_examples(trajectory)

    assert len(examples) == 2
    assert examples[0]["prompt"]  # non-empty, derived from the real context snapshot
    completion = json.loads(examples[0]["completion"])
    assert completion["tool"] == "search_corpus"
    assert json.loads(examples[1]["completion"])["finish"] is True


def test_trajectories_to_sft_dataset_concatenates_multiple_runs(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, _ = synthetic_corpus_and_tasks
    harness = AgentHarness(
        MockAgentPolicy([json.dumps({"thought": "done", "finish": True})]),
        corpus,
        bm25_retriever,
        HarnessConfig(max_steps=3),
    )
    trajectories = [harness.run("q1").context_manager.state.model_dump(mode="json") for _ in range(2)]

    examples = trajectories_to_sft_dataset(trajectories)
    assert len(examples) == 2
