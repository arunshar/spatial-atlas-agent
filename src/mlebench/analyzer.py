"""
Spatial Atlas: Competition Analyzer

Reads MLE-Bench competition description and data files to determine:
1. Task type (classification, regression, NLP, vision, time series)
2. Target column / metric
3. Data characteristics
4. Recommended strategy
"""

import json
import logging
from dataclasses import dataclass, field

from llm import LLMClient

logger = logging.getLogger("spatial-atlas.mlebench.analyzer")


@dataclass
class CompetitionAnalysis:
    """Structured analysis of an MLE-Bench competition."""

    competition_id: str = ""
    task_type: str = "tabular_classification"  # tabular_classification, tabular_regression, nlp, vision, timeseries, general
    metric: str = "accuracy"  # evaluation metric name
    metric_direction: str = "maximize"  # maximize or minimize
    target_column: str = ""  # column to predict
    submission_format: str = ""  # expected CSV format description
    data_description: str = ""  # brief description of the data
    available_files: list[str] = field(default_factory=list)
    strategy: str = "general"  # which strategy template to use
    key_insights: list[str] = field(default_factory=list)


ANALYSIS_PROMPT = """Analyze this Kaggle competition and extract key information.

## Competition Description
{description}

## Available Data Files
{file_listing}

## Sample Data Preview
{data_preview}

Return a JSON object with these fields:
{{
  "task_type": "tabular_classification" | "tabular_regression" | "nlp" | "vision" | "timeseries" | "general",
  "metric": "<evaluation metric name, e.g. accuracy, rmse, auc, f1, log_loss>",
  "metric_direction": "maximize" | "minimize",
  "target_column": "<column name to predict>",
  "submission_format": "<description of expected submission.csv format>",
  "data_description": "<1-2 sentence summary of the data>",
  "strategy": "tabular" | "nlp" | "vision" | "timeseries" | "general",
  "key_insights": ["<insight 1>", "<insight 2>", "<insight 3>"]
}}

Rules:
- If the task involves images/photos, set task_type to "vision" and strategy to "vision"
- If the task involves text classification/NLP, set task_type to "nlp" and strategy to "nlp"
- If there are date/time columns and prediction involves future values, set task_type to "timeseries"
- For binary/multiclass classification on structured data, set task_type to "tabular_classification" and strategy to "tabular"
- For regression on structured data, set task_type to "tabular_regression" and strategy to "tabular"
- If unsure, set strategy to "general"
- Be precise about the metric name; this determines how we optimize"""


class CompetitionAnalyzer:
    """Analyze MLE-Bench competition to determine optimal approach."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def analyze(
        self,
        description: str,
        file_listing: str,
        data_preview: str = "",
        competition_id: str = "",
    ) -> CompetitionAnalysis:
        """Analyze competition description and data to determine approach."""
        prompt = ANALYSIS_PROMPT.format(
            description=description[:6000],
            file_listing=file_listing,
            data_preview=data_preview[:2000],
        )

        try:
            result = await self.llm.generate(
                prompt,
                model_tier="standard",
                json_mode=True,
                max_tokens=2048,
            )
            data = json.loads(result)
        # Deliberately broad: an LLM call plus a json.loads, and any failure of either
        # falls back to the default analysis rather than failing the competition. Written
        # as a bare Exception because json.JSONDecodeError subclasses ValueError
        # subclasses Exception, so naming it alongside Exception caught nothing extra and
        # only implied a JSON-specific path that does not exist.
        except Exception as e:
            logger.warning(f"Competition analysis failed, using defaults: {e}")
            return CompetitionAnalysis(
                competition_id=competition_id,
                strategy="general",
            )

        analysis = CompetitionAnalysis(
            competition_id=competition_id,
            task_type=data.get("task_type", "general"),
            metric=data.get("metric", "accuracy"),
            metric_direction=data.get("metric_direction", "maximize"),
            target_column=data.get("target_column", ""),
            submission_format=data.get("submission_format", ""),
            data_description=data.get("data_description", ""),
            available_files=[],
            strategy=data.get("strategy", "general"),
            key_insights=data.get("key_insights", []),
        )

        logger.info(
            f"Competition analyzed: type={analysis.task_type}, "
            f"metric={analysis.metric}, strategy={analysis.strategy}"
        )
        return analysis
