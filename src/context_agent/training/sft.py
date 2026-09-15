  """QLoRA SFT training entrypoint (optional extension, GPU-only).

Fine-tunes a Qwen policy model on the (prompt, completion) pairs produced by
`training/dataset.py`. Requires CUDA + bitsandbytes for 4-bit quantization, so it is
NOT exercised by this repo's test suite or required to run the rest of Context-1 —
only import-checked. Heavy dependencies (torch/transformers/peft/trl) are imported
lazily inside `run()` so importing this module never requires them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class SFTConfig:
    base_model: str = "Qwen/Qwen3-8B"
    train_file: str = "data/trajectories/sft_train.jsonl"
    output_dir: str = "artifacts/sft/qwen3-8b-context1"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096


def build_example_text(example: dict) -> str:
    return f"<|system|>\n{example['system']}\n<|user|>\n{example['prompt']}\n<|assistant|>\n{example['completion']}"


def run(config: SFTConfig) -> None:
    """Runs QLoRA SFT. Requires a CUDA GPU; not exercised by tests."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    with open(config.train_file, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    dataset = Dataset.from_list([{"text": build_example_text(e)} for e in examples])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TRLSFTConfig(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            max_seq_length=config.max_seq_length,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)


def main() -> None:
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    run(SFTConfig(**raw))


if __name__ == "__main__":
    main()
