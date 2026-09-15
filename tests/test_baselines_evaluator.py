from context_agent.agent.harness import AgentHarness, HarnessConfig
from context_agent.answering.generator import MockAnswerGenerator
from context_agent.config import build_policy
from context_agent.evaluation.evaluator import evaluate_baseline, summarize


def _harness_factory(corpus, retriever):
    def factory(query: str) -> AgentHarness:
        policy = build_policy({"policy": "mock", "default_top_k": 5}, query=query)
        return AgentHarness(policy, corpus, retriever, HarnessConfig(max_steps=4, max_tool_calls=4))

    return factory


def test_all_four_baselines_run_and_produce_bounded_metrics(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, tasks = synthetic_corpus_and_tasks
    answer_generator = MockAnswerGenerator()

    for name in ("direct", "one_shot_rag", "long_context_rag", "agentic"):
        rows = evaluate_baseline(
            name,
            tasks,
            answer_generator,
            retriever=bm25_retriever,
            harness_factory=_harness_factory(corpus, bm25_retriever) if name == "agentic" else None,
            top_k=5,
            long_context_top_k=20,
        )
        assert len(rows) == len(tasks)
        summary = summarize(rows)
        assert summary["n_tasks"] == len(tasks)
        for metric_value in summary["retrieval"].values():
            assert 0.0 <= metric_value <= 1.0


def test_agentic_baseline_retrieves_more_than_direct(bm25_retriever, synthetic_corpus_and_tasks):
    corpus, tasks = synthetic_corpus_and_tasks
    answer_generator = MockAnswerGenerator()

    direct_rows = evaluate_baseline("direct", tasks, answer_generator)
    agentic_rows = evaluate_baseline(
        "agentic", tasks, answer_generator, harness_factory=_harness_factory(corpus, bm25_retriever)
    )

    direct_recall = summarize(direct_rows)["retrieval"]["supporting_document_recall"]
    agentic_recall = summarize(agentic_rows)["retrieval"]["supporting_document_recall"]
    assert agentic_recall >= direct_recall
