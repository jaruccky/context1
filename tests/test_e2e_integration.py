"""One end-to-end pass: synthetic corpus -> retriever -> agent -> EvidencePack ->
answer generator. No GPU, no network -- MockAgentPolicy + MockAnswerGenerator.
"""
import json

from context_agent.agent.harness import AgentHarness, HarnessConfig
from context_agent.agent.policy import MockAgentPolicy
from context_agent.answering.generator import MockAnswerGenerator
from context_agent.evidence.models import EvidencePack


def test_end_to_end_pipeline(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, tasks = synthetic_corpus_and_tasks
    task = next(t for t in tasks if t.task_id == "synthetic-1")

    script = [
        json.dumps(
            {"thought": "find the first programmer", "tool": "search_corpus", "arguments": {"query": "first computer programmer notes", "top_k": 3}}
        ),
        json.dumps(
            {"thought": "find who designed the engine", "tool": "search_corpus", "arguments": {"query": "designed the Analytical Engine", "top_k": 3}}
        ),
        json.dumps({"thought": "I have enough evidence", "finish": True}),
    ]
    harness = AgentHarness(MockAgentPolicy(script), corpus, bm25_retriever, HarnessConfig(max_steps=6, max_tool_calls=6))

    result = harness.run(task.question, task_id=task.task_id)
    pack = result.evidence_pack

    assert isinstance(pack, EvidencePack)
    assert pack.evidence, "agent should have retrieved some evidence"
    assert result.stop_reason == "finished"

    covered_docs = set(pack.document_ids())
    assert covered_docs & set(task.supporting_document_ids), "should retrieve at least one gold document"

    answer_generator = MockAnswerGenerator()
    answer = answer_generator.generate(pack)
    assert isinstance(answer, str)
    assert answer
