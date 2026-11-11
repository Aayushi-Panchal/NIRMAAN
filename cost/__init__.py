"""
NIRMAAN Cost Estimation Engine Package
Provides estimate_repair_cost and cost configuration matrix.
"""
from .cost_estimator import estimate_repair_cost
from .cost_config import (
    COST_CALIBRATION_CONFIG,
    SEVERITY_TIER_BOUNDS,
    LOCATION_RATE_TABLE,
    COST_ROUNDING_STEP,
    COST_ESTIMATION_METHOD
)

__all__ = [
    "estimate_repair_cost",
    "COST_CALIBRATION_CONFIG",
    "SEVERITY_TIER_BOUNDS",
    "LOCATION_RATE_TABLE",
    "COST_ROUNDING_STEP",
    "COST_ESTIMATION_METHOD"
]
