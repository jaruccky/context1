"""The agent loop: observe -> reason -> tool call -> observe -> ...

Enforces max_steps, max_tool_calls, a token budget (via ContextManager), duplicate
tool-call detection, malformed-action handling, and tool-exception handling, with a
fallback finish when the loop runs out of steps without the policy explicitly
finishing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import ValidationError

from context_agent.agent.actions import TOOL_NAMES, ActionParseError, AgentAction, parse_action
from context_agent.agent.policy import AgentPolicy
from context_agent.context.manager import ContextManager
from context_agent.data.corpus import Corpus
from context_agent.evidence.models import EvidencePack
from context_agent.retrieval.hybrid import HybridRetriever
from context_agent.tools import grep, prune, read, search
from context_agent.tools.errors import ToolError
from context_agent.utils.ids import hash_call


@dataclass
class HarnessConfig:
    max_steps: int = 10
    max_tool_calls: int = 8
    token_budget: int = 4000
    default_top_k: int = 5


@dataclass
class HarnessResult:
    evidence_pack: EvidencePack
    context_manager: ContextManager
    stop_reason: str  # "finished" | "max_steps" | "max_tool_calls"


class AgentHarness:
    def __init__(
        self,
        policy: AgentPolicy,
        corpus: Corpus,
        retriever: HybridRetriever,
        config: HarnessConfig | None = None,
    ):
        self.policy = policy
        self.corpus = corpus
        self.retriever = retriever
        self.config = config or HarnessConfig()

    def run(self, query: str, task_id: str | None = None) -> HarnessResult:
        cm = ContextManager(query=query, task_id=task_id, token_budget=self.config.token_budget)
        seen_call_hashes: set[str] = set()
        stop_reason = "max_steps"

        for step in range(1, self.config.max_steps + 1):
            context_snapshot = cm.render()
            start = time.perf_counter()
            raw = self.policy.next_action(query, context_snapshot, list(TOOL_NAMES), step)
            action = parse_action(raw)
            latency_ms = (time.perf_counter() - start) * 1000

            if isinstance(action, ActionParseError):
                cm.state.invalid_action_count += 1
                cm.state.add_step(
                    step,
                    {"raw": raw},
                    {"error": action.error},
                    latency_ms,
                    error=action.error,
                    context_snapshot=context_snapshot,
                )
                continue

            if action.finish:
                cm.state.add_step(
                    step,
                    action.model_dump(mode="json"),
                    {"finished": True},
                    latency_ms,
                    context_snapshot=context_snapshot,
                )
                stop_reason = "finished"
                break

            observation, error, _dispatched = self._handle_tool_call(action, cm, step, seen_call_hashes)
            cm.state.add_step(
                step,
                action.model_dump(mode="json"),
                observation,
                latency_ms,
                error=error,
                context_snapshot=context_snapshot,
            )

            if observation.get("blocked") == "max_tool_calls":
                stop_reason = "max_tool_calls"
                break

        return HarnessResult(evidence_pack=cm.to_evidence_pack(), context_manager=cm, stop_reason=stop_reason)

    def _handle_tool_call(
        self, action: AgentAction, cm: ContextManager, step: int, seen_call_hashes: set[str]
    ) -> tuple[dict, str | None, bool]:
        tool_name = action.tool.value  # type: ignore[union-attr]
        call_hash = hash_call(tool_name, action.arguments)

        if call_hash in seen_call_hashes:
            cm.state.duplicate_call_count += 1
            return {"duplicate": True, "message": "identical tool call already executed"}, None, False

        if cm.state.tool_call_count >= self.config.max_tool_calls:
            return {"blocked": "max_tool_calls"}, "max_tool_calls exceeded", False

        seen_call_hashes.add(call_hash)

        try:
            observation = self._execute_tool(tool_name, action.arguments, cm, step)
            cm.state.tool_call_count += 1
            return observation, None, True
        except ValidationError as exc:
            cm.state.invalid_action_count += 1
            error = f"invalid arguments for {tool_name}: {exc}"
            return {"error": error}, error, False
        except ToolError as exc:
            cm.state.tool_call_count += 1
            return {"error": str(exc)}, str(exc), True

    def _execute_tool(self, tool_name: str, arguments: dict, cm: ContextManager, step: int) -> dict:
        if tool_name == "search_corpus":
            args = search.SearchArguments.model_validate(arguments)
            return search.execute(args, self.retriever, cm, step)
        if tool_name == "grep_corpus":
            args = grep.GrepArguments.model_validate(arguments)
            return grep.execute(args, self.corpus, cm, step)
        if tool_name == "read_document":
            args = read.ReadArguments.model_validate(arguments)
            return read.execute(args, self.corpus, cm, step)
        if tool_name == "prune_chunks":
            args = prune.PruneArguments.model_validate(arguments)
            return prune.execute(args, cm, step)
        raise ToolError(f"unhandled tool: {tool_name}")
