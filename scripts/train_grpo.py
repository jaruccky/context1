#!/usr/bin/env python
"""GRPO training entrypoint (optional, requires a CUDA GPU; see training/grpo.py).

Usage:
    python scripts/train_grpo.py --config configs/grpo.yaml
"""
from __future__ import annotations

from context_agent.training.grpo import main

if __name__ == "__main__":
    main()
