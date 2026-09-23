"""
Spatial Atlas: Centralized Configuration

All configurable parameters in one place.
Environment variables override defaults.

Model tier layout:
  - fast     (gpt-4.1-mini)        cheap classification, parsing, formatting
  - standard (gpt-4.1)              code generation and mid-complexity analysis
  - strong   (gpt-4.1)              spatial reasoning, reflection, hard MLE tasks
  - vision   (gpt-4.1)              multimodal image/PDF/video description

All tiers default to OpenAI, requiring only a single API key. Override any
tier via ATLAS_*_MODEL env vars if you want to use a different provider.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("spatial-atlas.config")


def _env_or(var: str, default: str) -> str:
    """
    Return os.environ[var] if set AND non-empty, otherwise `default`.

    os.getenv returns an empty string for explicitly-blank env vars,
    which in turn propagates an empty model string into litellm and
    triggers 'LLM Provider NOT provided' errors. Treat empty strings
    as 'not set' to keep the deploy tolerant of blank secrets on
    Hugging Face Spaces.
    """
    value = os.environ.get(var)
    return value if value else default


def _optional_bool_env(var: str) -> bool | None:
    """Parse an optional boolean environment variable strictly."""
    value = os.environ.get(var)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{var} must be one of true/false, yes/no, on/off, or 1/0; got {value!r}")


# Default model identifiers. Kept at module scope (not just inside the
# dataclass defaults) so both Config and startup logging can reference
# them and stay in sync.
DEFAULT_FAST_MODEL = "openai/gpt-4.1-mini"
DEFAULT_STANDARD_MODEL = "openai/gpt-4.1"
DEFAULT_STRONG_MODEL = "openai/gpt-4.1"
DEFAULT_VISION_MODEL = "openai/gpt-4.1"


@dataclass
class Config:
    # === Model Tiers ===
    fast_model: str = field(default_factory=lambda: _env_or("ATLAS_FAST_MODEL", DEFAULT_FAST_MODEL))
    standard_model: str = field(
        default_factory=lambda: _env_or("ATLAS_STANDARD_MODEL", DEFAULT_STANDARD_MODEL)
    )
    strong_model: str = field(
        default_factory=lambda: _env_or("ATLAS_STRONG_MODEL", DEFAULT_STRONG_MODEL)
    )
    vision_model: str = field(
        default_factory=lambda: _env_or("ATLAS_VISION_MODEL", DEFAULT_VISION_MODEL)
    )
    # Optional vLLM chat-template control. Leave unset for hosted providers.
    # Remote vLLM benchmark runners set this to false so Qwen returns task content
    # directly instead of spending the output budget on hidden reasoning.
    llm_enable_thinking: bool | None = field(
        default_factory=lambda: _optional_bool_env("ATLAS_ENABLE_THINKING")
    )

    # === Cost Budgets ===
    max_tokens_per_task: int = 150_000
    max_reflection_rounds: int = 2

    # === FieldWork-specific ===
    max_video_frames: int = 30
    spatial_precision: int = 2  # decimal places for coordinates
    # Perception engine for the FieldWork scene step:
    #   "scenegraph" (default) = LLM extracts entities + estimates coordinates from text;
    #   "metric"               = SpatialClaw exact benchmark regions or SAM3 plus
    #                            Depth-Anything-3 reconstruct estimated 3D positions
    #                            (requires the spatial_agent package + a running GPU tool
    #                            server; falls back to scenegraph if unavailable).
    fieldwork_engine: str = field(
        default_factory=lambda: _env_or("ATLAS_FIELDWORK_ENGINE", "scenegraph")
    )
    # Evaluation drivers set this when a metric row must fail rather than silently
    # falling back to the scene-graph baseline. The A2A path keeps graceful fallback.
    fieldwork_metric_strict: bool = False
    reconstruct_max_frames: int = 32  # max frames per SpatialClaw Reconstruct call

    # === MLE-Bench-specific ===
    code_execution_timeout: int = 600  # seconds per code execution
    max_code_iterations: int = 3  # error-recovery retries on the first successful build
    # Score-driven iterations AFTER the first successful run. Each iteration
    # asks the strong model to propose an improved pipeline given the prior
    # code and its validation score, re-runs it, and keeps the best one.
    # Set to 0 to disable refinement entirely.
    max_refinement_iterations: int = 2
    # Hard wall-clock ceiling across all refinement iterations. Protects the
    # MLE-Bench per-task budget when a pipeline is slow to train.
    refinement_wall_time_seconds: int = 900

    @property
    def model_tiers(self) -> dict[str, str]:
        return {
            "fast": self.fast_model,
            "standard": self.standard_model,
            "strong": self.strong_model,
            "vision": self.vision_model,
        }

    def log_resolved_tiers(self) -> None:
        """
        Dump the resolved model tier map to stdout + logger at startup.

        This is the single best diagnostic for 'LLM Provider NOT provided'
        errors: if a tier logged here is empty or missing its provider
        prefix, the env var on the Space is wrong.
        """
        tiers = self.model_tiers
        lines = [f"{name:10s} = {value!r}" for name, value in tiers.items()]
        banner = "Resolved model tiers:\n  " + "\n  ".join(lines)
        logger.info(banner)
        print(banner)

        # Hard validation: empty or provider-less models will blow up
        # inside litellm at the first call. Crash early with a clear
        # message instead.
        for name, value in tiers.items():
            if not value:
                raise RuntimeError(
                    f"Model tier {name!r} is empty. Check the ATLAS_{name.upper()}_MODEL "
                    f"env var (or equivalent Space secret); blank strings are not allowed."
                )
            if "/" not in value:
                raise RuntimeError(
                    f"Model tier {name!r} = {value!r} has no provider prefix "
                    f"(expected something like 'openai/gpt-4.1' or "
                    f"'openai/gpt-4.1'). Fix the env var."
                )
