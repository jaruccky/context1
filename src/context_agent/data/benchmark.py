"""Multi-hop benchmark tasks: HotpotQA subset builder + a network-free synthetic fixture."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from context_agent.data.corpus import Document


class SupportingFact(BaseModel):
    document_id: str
    sentence_index: int


class BenchmarkTask(BaseModel):
    task_id: str
    question: str
    answer: str
    supporting_document_ids: list[str]
    supporting_facts: list[SupportingFact] = Field(default_factory=list)
    split: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "doc"


def build_hotpotqa_subset(
    n_examples: int = 40,
    split: str = "validation",
    config: str = "distractor",
    dataset_id: str = "hotpotqa/hotpot_qa",
    seed: int = 13,
) -> tuple[list[Document], list[BenchmarkTask]]:
    """Download a small HotpotQA slice and turn it into (documents, tasks).

    Requires network access (huggingface `datasets`). Each hotpot example's
    context paragraphs become Documents (deduplicated by title); the question
    becomes a BenchmarkTask with gold supporting document ids and sentence-level
    supporting facts.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n_examples, len(ds))))

    documents: dict[str, Document] = {}
    tasks: list[BenchmarkTask] = []

    for row in ds:
        titles = row["context"]["title"]
        sentences_per_doc = row["context"]["sentences"]
        title_to_doc_id: dict[str, str] = {}

        for title, sentences in zip(titles, sentences_per_doc):
            document_id = _slugify(title)
            title_to_doc_id[title] = document_id
            if document_id not in documents:
                documents[document_id] = Document(
                    document_id=document_id,
                    title=title,
                    text=" ".join(sentences),
                    source="hotpotqa",
                    metadata={"sentences": sentences},
                )

        sf_titles = row["supporting_facts"]["title"]
        sf_sent_ids = row["supporting_facts"]["sent_id"]
        supporting_facts = [
            SupportingFact(document_id=title_to_doc_id[t], sentence_index=s)
            for t, s in zip(sf_titles, sf_sent_ids)
            if t in title_to_doc_id
        ]
        supporting_document_ids = sorted({sf.document_id for sf in supporting_facts})

        tasks.append(
            BenchmarkTask(
                task_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                supporting_document_ids=supporting_document_ids,
                supporting_facts=supporting_facts,
                split=split,
            )
        )

    return list(documents.values()), tasks


def load_synthetic_benchmark() -> tuple[list[Document], list[BenchmarkTask]]:
    """A tiny, hand-written multi-hop corpus/benchmark. No network needed.

    Used by tests and as a CI-safe fallback for `prepare_data.py`.
    """
    documents = [
        Document(
            document_id="ada-lovelace",
            title="Ada Lovelace",
            text=(
                "Ada Lovelace was a 19th century mathematician who worked with Charles "
                "Babbage on the Analytical Engine. She is often regarded as the first "
                "computer programmer for her notes on the engine, published in 1843. "
                "She was born in London in 1815."
            ),
            source="synthetic",
        ),
        Document(
            document_id="analytical-engine",
            title="Analytical Engine",
            text=(
                "The Analytical Engine was a proposed mechanical general-purpose computer "
                "designed by Charles Babbage. It was never completed during his lifetime, "
                "but its design influenced later computer architecture. Babbage began "
                "designing it in 1837."
            ),
            source="synthetic",
        ),
        Document(
            document_id="charles-babbage",
            title="Charles Babbage",
            text=(
                "Charles Babbage was an English mathematician and inventor, credited with "
                "originating the concept of a programmable computer. He was born in London "
                "in 1791 and is considered the father of the computer."
            ),
            source="synthetic",
        ),
        Document(
            document_id="turing-award",
            title="Turing Award",
            text=(
                "The Turing Award is an annual prize given by the Association for Computing "
                "Machinery for contributions of lasting importance to computing. It is often "
                "referred to as the Nobel Prize of computing. It was first awarded in 1966."
            ),
            source="synthetic",
        ),
        Document(
            document_id="acm",
            title="Association for Computing Machinery",
            text=(
                "The Association for Computing Machinery (ACM) is a professional society for "
                "computing founded in 1947. It administers several awards, including the "
                "Turing Award, and publishes numerous journals and conference proceedings."
            ),
            source="synthetic",
        ),
        Document(
            document_id="python-language",
            title="Python (programming language)",
            text=(
                "Python is a high-level, general-purpose programming language created by "
                "Guido van Rossum and first released in 1991. It emphasizes code readability "
                "and is widely used for scripting, data science, and web development."
            ),
            source="synthetic",
        ),
        Document(
            document_id="guido-van-rossum",
            title="Guido van Rossum",
            text=(
                "Guido van Rossum is a Dutch programmer best known as the creator of the "
                "Python programming language. He was born in 1956 in Haarlem, Netherlands, "
                "and worked at Google and Dropbox before joining Microsoft."
            ),
            source="synthetic",
        ),
    ]

    tasks = [
        BenchmarkTask(
            task_id="synthetic-1",
            question=(
                "Who designed the machine that the first computer programmer wrote notes "
                "about, and in what city was that person born?"
            ),
            answer="Charles Babbage, London",
            supporting_document_ids=["ada-lovelace", "analytical-engine", "charles-babbage"],
            supporting_facts=[
                SupportingFact(document_id="ada-lovelace", sentence_index=0),
                SupportingFact(document_id="analytical-engine", sentence_index=0),
                SupportingFact(document_id="charles-babbage", sentence_index=1),
            ],
            split="synthetic",
        ),
        BenchmarkTask(
            task_id="synthetic-2",
            question=(
                "What organization gives the award nicknamed the 'Nobel Prize of computing', "
                "and in what year was that organization founded?"
            ),
            answer="Association for Computing Machinery, 1947",
            supporting_document_ids=["turing-award", "acm"],
            supporting_facts=[
                SupportingFact(document_id="turing-award", sentence_index=1),
                SupportingFact(document_id="acm", sentence_index=0),
            ],
            split="synthetic",
        ),
        BenchmarkTask(
            task_id="synthetic-3",
            question=(
                "In what year was the creator of Python born, and what language did he create?"
            ),
            answer="1956, Python",
            supporting_document_ids=["python-language", "guido-van-rossum"],
            supporting_facts=[
                SupportingFact(document_id="python-language", sentence_index=0),
                SupportingFact(document_id="guido-van-rossum", sentence_index=1),
            ],
            split="synthetic",
        ),
    ]

    return documents, tasks


def save_tasks_jsonl(tasks: list[BenchmarkTask], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")


def load_tasks_jsonl(path: str | Path) -> list[BenchmarkTask]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(BenchmarkTask.model_validate_json(line))
    return tasks
