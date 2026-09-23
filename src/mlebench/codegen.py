"""
Spatial Atlas: ML code generator

Generates complete, self-contained Python scripts for MLE-Bench competitions.
Uses strategy templates + LLM to produce runnable ML pipelines.
"""

import logging

from llm import LLMClient
from mlebench.analyzer import CompetitionAnalysis
from mlebench.strategies import get_strategy_template
from mlebench.strategies.leaks import leak_prompt_block, match_leak

logger = logging.getLogger("spatial-atlas.mlebench.codegen")


CODEGEN_SYSTEM_PROMPT = """You are an expert Kaggle grandmaster who writes complete, runnable ML pipeline scripts.
Your code must:
- Be COMPLETELY self-contained (no human intervention)
- Handle edge cases (missing values, unexpected dtypes, empty columns)
- Use robust, proven approaches (XGBoost, LightGBM, sklearn, pandas, numpy)
- Print progress messages to stdout
- Save predictions to the exact submission path specified
- Match the submission format EXACTLY (correct columns, correct dtypes)
- Include a simple train/validation split for sanity checking
- Never crash (wrap risky operations in try/except with fallbacks)
- NEVER use display() or show() or plt.show() (headless environment)
- On the validation split, print exactly one line of the form
      VALIDATION_SCORE: <float>
  where <float> is the competition metric on the held-out fold. This line
  is machine-parsed to drive score-based refinement. If the competition
  metric is not directly computable, print the closest proxy (e.g. AUC,
  accuracy, RMSE) and prefix the line identically."""


CODEGEN_PROMPT = """Generate a complete Python script that solves this Kaggle competition.

## Competition Description
{description}

## Available Data Files
{file_listing}

## Data Preview
{data_preview}

## Analysis
- Task type: {task_type}
- Metric: {metric} ({metric_direction})
- Target column: {target_column}
- Submission format: {submission_format}

{leak_block}
## Strategy Template (use as starting point, adapt as needed)
{strategy_template}

## Requirements
- Read data from: {data_dir}/
- Save submission to: {submission_path}
- Use ONLY these libraries: pandas, numpy, sklearn, xgboost, lightgbm, scipy (all pre-installed)
- The script must be COMPLETE and RUNNABLE with no edits
- Handle missing values explicitly (fillna, dropna, or impute)
- Handle categorical columns (label encode or one-hot encode)
- Print progress: "Loading data...", "Training model...", "Generating predictions...", "Submission saved."
- If target column is unclear, infer it from the submission format
- Produce a submission.csv with the exact required format
- If the Leak Audit section above fires a real hit, write the leak-derived
  submission first, then train the baseline anyway as a fallback

Generate ONLY the Python code. No markdown fences, no explanation."""


REFINE_PROMPT = """The ML pipeline below ran successfully and produced a valid submission,
but we want a stronger score. Propose ONE targeted improvement and return the
full updated script.

## Current Script
```python
{code}
```

## Current Validation Score
{current_score}  (metric: {metric}, higher_is_{metric_direction})

## Competition Description (first 3000 chars)
{description}

## Data Files
{file_listing}

## Improvement Menu (pick the most promising ONE for this competition)
- Stronger model family (LightGBM -> XGBoost with different params, or add CatBoost)
- K-fold cross validation instead of single holdout (report mean score)
- Target encoding or more aggressive feature engineering on categoricals
- Stacking or blending two model types
- Better handling of missing values / outliers
- Feature interactions or polynomial features
- Class-balanced sampling for imbalanced targets
- Hyperparameter tuning via a small grid or Optuna
- Leak exploit if the Leak Audit section suggested one

## Requirements
- Keep the VALIDATION_SCORE: <float> print line so refinement can continue.
- Keep the exact submission path and output format.
- Keep the script complete and runnable with no human edits.
- Prefer ONE change over many: we will iterate again if this improves things.
- Output ONLY the complete Python code. No markdown fences, no explanation.
"""


FIX_PROMPT = """The ML pipeline script below failed with an error. Fix the script.

## Original Script
```python
{code}
```

## Error
{error}

## Stdout Before Error
{stdout}

## Competition Description (first 2000 chars)
{description}

## Data Files
{file_listing}

## Instructions
- Fix ONLY the error; do not rewrite the whole script unless necessary
- If a file doesn't exist, check the data directory listing for the correct filename
- If a column doesn't exist, print available columns first, then adapt
- If a library is missing, replace with an available one (pandas, numpy, sklearn, xgboost, lightgbm)
- Keep all existing good logic intact
- Output ONLY the complete fixed Python script, no explanation"""


class MLCodeGenerator:
    """Generate and fix ML pipeline code for competitions."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate(
        self,
        description: str,
        data_dir: str,
        file_listing: str,
        data_preview: str,
        analysis: CompetitionAnalysis,
        submission_path: str = "submission.csv",
    ) -> str:
        """Generate a complete ML pipeline script."""
        strategy_template = get_strategy_template(analysis.strategy)
        leak_block = leak_prompt_block(description, file_listing)
        matched = match_leak(description, file_listing)
        if matched is not None:
            logger.info(
                f"Leak hint matched: {matched.slug} ({matched.title}); "
                "injecting targeted exploit guidance into codegen prompt"
            )

        prompt = CODEGEN_PROMPT.format(
            description=description[:5000],
            file_listing=file_listing,
            data_preview=data_preview[:2000],
            task_type=analysis.task_type,
            metric=analysis.metric,
            metric_direction=analysis.metric_direction,
            target_column=analysis.target_column,
            submission_format=analysis.submission_format,
            leak_block=leak_block,
            strategy_template=strategy_template,
            data_dir=data_dir,
            submission_path=submission_path,
        )

        code = await self.llm.generate(
            prompt,
            model_tier="strong",
            system_prompt=CODEGEN_SYSTEM_PROMPT,
            max_tokens=8192,
            temperature=0.1,
        )

        code = self._clean_code(code)
        logger.info(f"Generated pipeline: {len(code)} chars, {code.count(chr(10))} lines")
        return code

    async def refine(
        self,
        code: str,
        current_score: float,
        metric: str,
        metric_direction: str,
        description: str,
        file_listing: str,
    ) -> str:
        """
        Propose an improved version of a pipeline that already works.

        Called after a successful run to drive score-based iteration. The
        caller is responsible for running the returned code, comparing the
        new VALIDATION_SCORE against `current_score`, and keeping whichever
        submission is better.
        """
        prompt = REFINE_PROMPT.format(
            code=code,
            current_score=current_score,
            metric=metric,
            metric_direction=metric_direction,
            description=description[:3000],
            file_listing=file_listing,
        )

        refined = await self.llm.generate(
            prompt,
            model_tier="strong",
            system_prompt=CODEGEN_SYSTEM_PROMPT,
            max_tokens=8192,
            temperature=0.3,  # slight temperature: we want creative variation
        )

        refined = self._clean_code(refined)
        logger.info(f"Refined pipeline: {len(refined)} chars")
        return refined

    async def fix(
        self,
        code: str,
        error: str,
        stdout: str,
        description: str,
        file_listing: str,
    ) -> str:
        """Fix a failed pipeline script based on the error."""
        prompt = FIX_PROMPT.format(
            code=code,
            error=error[-2000:],
            stdout=stdout[-1000:],
            description=description[:2000],
            file_listing=file_listing,
        )

        fixed_code = await self.llm.generate(
            prompt,
            model_tier="strong",
            system_prompt=CODEGEN_SYSTEM_PROMPT,
            max_tokens=8192,
            temperature=0.0,
        )

        fixed_code = self._clean_code(fixed_code)
        logger.info(f"Fixed pipeline: {len(fixed_code)} chars")
        return fixed_code

    def _clean_code(self, code: str) -> str:
        """Remove markdown fences and clean up generated code."""
        code = code.strip()
        # Remove markdown code fences
        if code.startswith("```python"):
            code = code[len("```python") :]
        elif code.startswith("```"):
            code = code[3:]
        code = code.removesuffix("```")
        code = code.strip()

        # Ensure there's a newline at the end
        if not code.endswith("\n"):
            code += "\n"

        return code
