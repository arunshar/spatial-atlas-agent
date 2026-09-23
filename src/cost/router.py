"""
Spatial Atlas: 3-Tier Model Router

Routes tasks to appropriate model tiers based on complexity.
Fast for classification/parsing, Standard for reasoning/code, Strong for spatial/reflection.
"""

from config import Config

# Task types mapped to model tiers
FAST_TASKS = {"classify", "parse", "format", "extract_text", "detect_format"}
STANDARD_TASKS = {"code_gen", "analyze", "plan", "reason", "summarize"}
STRONG_TASKS = {"spatial_reasoning", "complex_vision", "reflection", "numerical", "json_analysis"}


class CostRouter:
    def __init__(self, config: Config):
        self.config = config

    def select_tier(self, task_type: str) -> str:
        """Select model tier based on task type."""
        if task_type in FAST_TASKS:
            return "fast"
        elif task_type in STRONG_TASKS:
            return "strong"
        else:
            return "standard"

    def select_model(self, task_type: str) -> str:
        """Select actual model name based on task type."""
        tier = self.select_tier(task_type)
        return self.config.model_tiers[tier]
