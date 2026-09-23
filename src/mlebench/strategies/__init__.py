"""
Strategy template selector for MLE-Bench competitions.
"""

from mlebench.strategies.autogluon import AUTOGLUON_TEMPLATE
from mlebench.strategies.general import GENERAL_TEMPLATE
from mlebench.strategies.nlp import NLP_TEMPLATE
from mlebench.strategies.tabular import TABULAR_TEMPLATE
from mlebench.strategies.timeseries import TIMESERIES_TEMPLATE
from mlebench.strategies.vision_ml import VISION_TEMPLATE

STRATEGY_MAP = {
    "tabular": AUTOGLUON_TEMPLATE,  # AutoGluon is the default for tabular (falls back to LightGBM)
    "tabular_basic": TABULAR_TEMPLATE,  # Manual LightGBM/XGBoost
    "nlp": NLP_TEMPLATE,
    "vision": VISION_TEMPLATE,
    "timeseries": TIMESERIES_TEMPLATE,
    "general": GENERAL_TEMPLATE,
    "autogluon": AUTOGLUON_TEMPLATE,
}


def get_strategy_template(strategy: str) -> str:
    """Get the strategy template code for a given strategy name."""
    return STRATEGY_MAP.get(strategy, GENERAL_TEMPLATE)
