#!/usr/bin/env python
"""QLoRA SFT training entrypoint (optional, requires a CUDA GPU; see training/sft.py).

Usage:
    python scripts/train_sft.py --config configs/sft.yaml
"""
from __future__ import annotations

from context_agent.training.sft import main

if __name__ == "__main__":
    main()
