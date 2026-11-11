"""
cost_config.py
--------------
Centralized configuration for NIRMAAN Rule-Based / Data-Driven Repair Cost Estimation Engine v1.

NOTE:
These figures represent calibrated baseline engineering ranges for municipal civic works in India (INR / ₹).
They are NOT claimed to be trained ML regression predictions or exact market contractor bids.
All parameters are centralized here for easy tuning and municipal rate table integration.
"""

# Base cost ranges (INR / ₹) per category and severity tier
COST_CALIBRATION_CONFIG = {
    "potholes": {
        "Low": {
            "min_base": 2500,
            "max_base": 4500,
            "work_scope": "Surface patch compaction, cold/hot bitumen mix filler, localized edge seal"
        },
        "Medium": {
            "min_base": 5000,
            "max_base": 9000,
            "work_scope": "Sub-base compaction, cold-mix asphalt overlay, and structural edge grading"
        },
        "High": {
            "min_base": 10000,
            "max_base": 18000,
            "work_scope": "Full-depth excavation, sub-grade stabilization, and multi-layer asphalt resurfacing"
        }
    },
    "streetlights": {
        "Low": {
            "min_base": 2000,
            "max_base": 4000,
            "work_scope": "Luminaire LED replacement, terminal fuse check, and dusk sensor calibration"
        },
        "Medium": {
            "min_base": 4000,
            "max_base": 9000,
            "work_scope": "Bracket arm overhaul, junction box rewiring, and surge protector restoration"
        },
        "High": {
            "min_base": 10000,
            "max_base": 19000,
            "work_scope": "Pole foundation re-anchoring, structural mast replacement, and underground cabling"
        }
    },
    "waterlogging": {
        "Low": {
            "min_base": 2500,
            "max_base": 5000,
            "work_scope": "Catch-basin desilting, storm-grate clearance, and surface gutter grading"
        },
        "Medium": {
            "min_base": 6000,
            "max_base": 12000,
            "work_scope": "Municipal suction tanker deployment, conduit rodding, and lateral drain flush"
        },
        "High": {
            "min_base": 14000,
            "max_base": 24000,
            "work_scope": "Heavy-duty diesel dewatering pumps, trunk line clearing, and culvert rebuild"
        }
    },
    "garbage": {
        "Low": {
            "min_base": 1500,
            "max_base": 3000,
            "work_scope": "Manual collection crew, bagged disposal, and localized disinfectant spray"
        },
        "Medium": {
            "min_base": 3000,
            "max_base": 6000,
            "work_scope": "Tipper truck transport, mechanized shovel collection, and ground bleaching"
        },
        "High": {
            "min_base": 6000,
            "max_base": 12000,
            "work_scope": "Multi-trip compactors, backhoe loader clearing, and public health bio-sanitization"
        }
    }
}

# Severity score normalizations per severity tier
# Standard calibration brackets from CategoryConditioned-ResNet18 severity model
SEVERITY_TIER_BOUNDS = {
    "Low": {"min_score": 1.0, "max_score": 3.8},
    "Medium": {"min_score": 3.5, "max_score": 6.8},
    "High": {"min_score": 6.5, "max_score": 9.9}
}

# Optional extensible location rate multiplier table (default 1.0 for neutral / unconfigured)
LOCATION_RATE_TABLE = {
    "default": 1.0,
    "metro": 1.15,
    "tier1": 1.10,
    "tier2": 1.00,
    "tier3_rural": 0.85
}

# Rounding increment for realistic municipal civic estimates (INR / ₹)
COST_ROUNDING_STEP = 100

# Cost method metadata identifier
COST_ESTIMATION_METHOD = "Rule-based / data-driven repair cost estimation engine using AI-derived category and severity outputs"
