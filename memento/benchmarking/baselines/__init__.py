"""
Baseline implementations of competing prompt evolution methods.

This module contains implementations of:
- PromptBreeder: Evolutionary prompt optimization
- SelfEvolvingGPT: Experience-based learning
- Auto-Evolve: Self-reasoning framework
"""

from .auto_evolve import AutoEvolve
from .promptbreeder import PromptBreeder
from .self_evolving_gpt import SelfEvolvingGPT

__all__ = ["PromptBreeder", "SelfEvolvingGPT", "AutoEvolve"]
