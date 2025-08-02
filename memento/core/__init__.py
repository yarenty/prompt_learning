"""
Core components of the Memento framework.

This module contains the main classes and functionality for the meta-cognitive
learning system.
"""

from .collector import FeedbackCollector
from .learner import PromptLearner
from .processor import PromptProcessor

__all__ = ["PromptLearner", "FeedbackCollector", "PromptProcessor"]
