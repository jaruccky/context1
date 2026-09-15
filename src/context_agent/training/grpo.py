"""GRPO (Group Relative Policy Optimization) training for the agent policy (optional
extension, GPU-only).

Unlike `training/sft.py` (which imitates logged trajectories), this samples full
multi-step episodes from the *current* policy against the real corpus and tools (via
`agent.harness.AgentHarness`), scores each finished episode with the reward function
in `training/rewards.py`, and pushes the policy towards the higher-reward episodes
within each group of samples drawn for the same question -- Group Relative Policy
Optimization (Shao et al., 2024): REINFORCE with a per-question, per-group baseline
instead of a learned value function, plus a KL penalty back to the frozen reference
policy.

Requires CUDA + transformers/peft, so it is NOT exercised by this repo's test suite or
required to run the rest of Context-1. Only the pure `group_advantages` helper is
unit-tested (see tests/test_grpo_advantages.py); everything else is import-checked
only. Heavy dependencies are imported lazily inside `run()` / `HFAgentPolicy` /
`_sequence_logprobs` so importing this module never requires them.
"""
from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass, field

from context_agent.agent.harness import AgentHarness, HarnessConfig
from context_agent.agent.policy import AgentPolicy
from context_agent.agent.prompts import SYSTEM_PROMPT, render_user_turn
from context_agent.data.benchmark import BenchmarkTask, load_tasks_jsonl
from context_agent.data.corpus import Corpus
from context_agent.retrieval.hybrid import HybridRetriever
from context_agent.training.rewards import compute_reward


@dataclass
class GRPOConfig:
    base_model: str = "Qwen/Qwen3-8B"
    sft_adapter: str | None = None  # LoRA checkpoint from training/sft.py to continue from
    corpus_path: str = "data/corpus/corpus.jsonl"
    benchmark_path: str = "data/benchmark/tasks.jsonl"
    retrieval_config: str = "configs/retrieval.yaml"
    output_dir: str = "artifacts/grpo/qwen3-8b-context1"

    group_size: int = 4  # G: episodes sampled per question per iteration
    tasks_per_iteration: int = 4
    num_iterations: int = 200
    save_every: int = 20

    max_steps: int = 8  # AgentHarness.max_steps
    max_tool_calls: int = 6
    token_budget: int = 3000

    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256

    learning_rate: float = 1e-5
    kl_coef: float = 0.04
    max_grad_norm: float = 1.0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    reward_alpha: float = 0.02
    reward_beta: float = 0.0005
    reward_gamma: float = 0.1

    seed: int = 0


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """Group-relative advantage: (r_i - mean(group)) / (std(group) + eps).

    Returns all-zero advantages for a degenerate group (<=1 sample, or every sample
    scored identically) -- there is no relative signal to learn from, so those
    episodes contribute nothing to the policy-gradient loss.
    """
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance**0.5
    if std < eps:
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + eps) for r in rewards]


@dataclass
class _RolloutStep:
    prompt_ids: list[int]
    completion_ids: list[int]


@dataclass
class _Episode:
    task_id: str
    reward: float
    steps: list[_RolloutStep] = field(default_factory=list)
    advantage: float = 0.0


class HFAgentPolicy(AgentPolicy):
    """Samples raw completions from a local HF causal LM instead of an API call.

    Records every (prompt_ids, completion_ids) pair it generates in `.rollout`, so a
    caller can replay the exact sampled tokens for a teacher-forced log-prob pass
    later -- re-tokenizing the decoded completion text would not reliably round-trip
    to the same token ids (special tokens, byte-fallback, etc).
    """

    def __init__(self, model, tokenizer, device: str, temperature: float, top_p: float, max_new_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.rollout: list[_RolloutStep] = []

    def next_action(self, query: str, context_text: str, available_tools: list[str], step: int) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_turn(context_text, available_tools)},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output_ids[0, prompt_len:].tolist()
        self.rollout.append(_RolloutStep(prompt_ids=inputs["input_ids"][0].tolist(), completion_ids=completion_ids))
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)


def _run_episode(
    task: BenchmarkTask,
    corpus: Corpus,
    retriever: HybridRetriever,
    model,
    tokenizer,
    device: str,
    config: GRPOConfig,
) -> _Episode:
    policy = HFAgentPolicy(model, tokenizer, device, config.temperature, config.top_p, config.max_new_tokens)
    harness = AgentHarness(
        policy,
        corpus,
        retriever,
        HarnessConfig(
            max_steps=config.max_steps,
            max_tool_calls=config.max_tool_calls,
            token_budget=config.token_budget,
        ),
    )
    result = harness.run(task.question, task_id=task.task_id)
    retrieved_ids = result.evidence_pack.document_ids()
    success = result.stop_reason == "finished"
    reward = compute_reward(
        result.evidence_pack.trajectory_summary,
        retrieved_ids,
        task.supporting_document_ids,
        success,
        alpha=config.reward_alpha,
        beta=config.reward_beta,
        gamma=config.reward_gamma,
    ).total
    return _Episode(task_id=task.task_id, reward=reward, steps=policy.rollout)


def _sequence_logprobs(model, prompt_ids: list[int], completion_ids: list[int], device: str, use_adapter: bool):
    """Per-token log p(token | prefix) over `completion_ids`, teacher-forced.

    `use_adapter=False` runs the frozen base weights (via `model.disable_adapter()`)
    to get reference-policy log-probs for the KL penalty, without keeping a second
    full copy of the model in memory. Only valid when `model` is a PEFT model.
    """
    import torch
    import torch.nn.functional as F

    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    ctx = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with ctx:
        logits = model(input_ids).logits

    completion_len = len(completion_ids)
    # logits[t] predicts token t+1, so the completion's first token is predicted by
    # the logit at the last prompt position.
    relevant_logits = logits[0, len(prompt_ids) - 1 : -1, :]
    log_probs = F.log_softmax(relevant_logits.float(), dim=-1)
    target = torch.tensor(completion_ids, device=device)
    return log_probs.gather(1, target.unsqueeze(1)).squeeze(1)[-completion_len:]


def run(config: GRPOConfig) -> None:
    """Runs GRPO training. Requires a CUDA GPU; not exercised by tests."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from context_agent.config import build_retriever_from_config, load_yaml

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=torch.bfloat16, device_map=device)
    if config.sft_adapter:
        model = PeftModel.from_pretrained(model, config.sft_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    corpus = Corpus.from_jsonl(config.corpus_path)
    retriever = build_retriever_from_config(corpus, load_yaml(config.retrieval_config))
    tasks = load_tasks_jsonl(config.benchmark_path)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)

    for iteration in range(1, config.num_iterations + 1):
        batch_tasks = random.choices(tasks, k=config.tasks_per_iteration)

        model.eval()
        episodes: list[_Episode] = []
        for task in batch_tasks:
            group = [
                _run_episode(task, corpus, retriever, model, tokenizer, device, config)
                for _ in range(config.group_size)
            ]
            for episode, advantage in zip(group, group_advantages([ep.reward for ep in group])):
                episode.advantage = advantage
            episodes.extend(group)

        model.train()
        optimizer.zero_grad()
        total_steps = sum(len(ep.steps) for ep in episodes) or 1
        mean_reward = sum(ep.reward for ep in episodes) / len(episodes)

        for episode in episodes:
            if episode.advantage == 0.0 or not episode.steps:
                continue
            for rollout_step in episode.steps:
                policy_logps = _sequence_logprobs(
                    model, rollout_step.prompt_ids, rollout_step.completion_ids, device, use_adapter=True
                )
                with torch.no_grad():
                    ref_logps = _sequence_logprobs(
                        model, rollout_step.prompt_ids, rollout_step.completion_ids, device, use_adapter=False
                    )
                pg_loss = -episode.advantage * policy_logps.mean()
                log_ratio = ref_logps - policy_logps
                kl = (log_ratio.exp() - log_ratio - 1).mean()  # unbiased (k3) KL estimator
                loss = (pg_loss + config.kl_coef * kl) / total_steps
                loss.backward()

        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), config.max_grad_norm)
        optimizer.step()

        print(
            f"iter {iteration}/{config.num_iterations}  mean_reward={mean_reward:.3f}  "
            f"episodes={len(episodes)}  steps={total_steps}"
        )

        if iteration % config.save_every == 0 or iteration == config.num_iterations:
            model.save_pretrained(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)


def main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    run(GRPOConfig(**raw))


if __name__ == "__main__":
    main()
