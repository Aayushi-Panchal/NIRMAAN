"""
cost_estimator.py
-----------------
Rule-based / data-driven repair cost estimation engine for NIRMAAN civic issues.

PIPELINE:
  Uploaded Image
      ↓
  Existing ResNet-50 Recognition (Category)
      ↓
  Existing CategoryConditioned-ResNet18 Severity (Severity + Severity Score)
      ↓
  Visual/Damage Feature Extraction (damage_factor)
      ↓
  Cost Estimation Engine (intra-tier range calculation)
      ↓
  Estimated Repair Cost Range (INR / ₹)

NOTE:
This module does NOT pretend to be a trained ML regression model.
It is a transparent, deterministic, data-driven cost engine parameterized by
civil engineering repair baselines and fine-tuned by AI-derived category and severity outputs.
"""

import io
import numpy as np
from PIL import Image

try:
    from cost_config import (
        COST_CALIBRATION_CONFIG,
        SEVERITY_TIER_BOUNDS,
        LOCATION_RATE_TABLE,
        COST_ROUNDING_STEP,
        COST_ESTIMATION_METHOD,
    )
except ImportError:
    from .cost_config import (
        COST_CALIBRATION_CONFIG,
        SEVERITY_TIER_BOUNDS,
        LOCATION_RATE_TABLE,
        COST_ROUNDING_STEP,
        COST_ESTIMATION_METHOD,
    )


def _load_image(image_input):
    """Safely converts file path, bytes, BytesIO, or PIL Image into standard RGB Image."""
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        img = Image.open(io.BytesIO(content))
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def extract_visual_damage_features(image_input, category="potholes"):
    """
    Extracts measurable, deterministic visual/damage extent indicators from an image.
    Uses relative visual features, avoiding fabricated physical dimensions.

    Returns:
        dict: {
            "damage_factor": float in [0.0, 1.0],
            "features": dict of relative metric values
        }
    """
    img = _load_image(image_input)
    # Standard 224x224 RGB normalization for reproducible deterministic feature extraction
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    # Luminance channel
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    med_lum = float(np.median(lum))
    lum_var = float(np.var(lum))

    # Spatial luminance gradients
    grad_x = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
    grad_y = np.abs(np.diff(lum, axis=0, append=lum[-1:, :]))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_density = float(np.mean(grad_mag > 0.08))

    cat_key = category.lower() if category and category.lower() in COST_CALIBRATION_CONFIG else "potholes"

    if cat_key == "potholes":
        # Cavity area: pixels significantly darker than surrounding road median
        rel_cavity = float(np.mean(lum < (med_lum * 0.72)))
        # Deep cavity core in shadow
        dark_core = float(np.mean(lum < (med_lum * 0.45)))
        # Crack density along asphalt fracture lines
        crack_density = edge_density

        raw_factor = (
            0.40 * min(rel_cavity * 2.8, 1.0)
            + 0.35 * min(crack_density * 2.5, 1.0)
            + 0.25 * min(dark_core * 4.5, 1.0)
        )
        damage_factor = float(np.clip(raw_factor, 0.05, 0.95))
        features = {
            "relative_cavity_ratio": round(rel_cavity, 4),
            "deep_core_ratio": round(dark_core, 4),
            "crack_edge_density": round(crack_density, 4),
        }

    elif cat_key == "waterlogging":
        # Road plane in lower 65% of the frame
        road_lum = lum[80:, :]
        rlg_x = np.abs(np.diff(road_lum, axis=1, append=road_lum[:, -1:]))
        rlg_y = np.abs(np.diff(road_lum, axis=0, append=road_lum[-1:, :]))
        # Specular water reflection / standing water smoothness
        water_smoothness = float(np.mean((rlg_x < 0.035) & (rlg_y < 0.035)))
        # Deep ponding contrast against dry asphalt
        dark_ponding = float(np.mean(road_lum < (med_lum * 0.75)))

        raw_factor = (
            0.50 * min(water_smoothness * 1.6, 1.0)
            + 0.35 * min(dark_ponding * 2.0, 1.0)
            + 0.15 * min(lum_var * 7.0, 1.0)
        )
        damage_factor = float(np.clip(raw_factor, 0.05, 0.95))
        features = {
            "water_specular_coverage": round(water_smoothness, 4),
            "dark_ponding_ratio": round(dark_ponding, 4),
            "luminance_variance": round(lum_var, 4),
        }

    elif cat_key == "streetlights":
        # Structural damage: jagged edges and component fragmentation
        structural_damage = float(np.mean(grad_mag > 0.12))
        horiz_edges = float(np.mean(grad_x > 0.08))
        vert_edges = float(np.mean(grad_y > 0.08))
        # Deviation from vertical upright pole posture
        tilt_disorder = horiz_edges / (vert_edges + 1e-4)
        dark_coverage = float(np.mean(lum < 0.25))

        raw_factor = (
            0.45 * min(structural_damage * 2.6, 1.0)
            + 0.35 * min(tilt_disorder * 0.6, 1.0)
            + 0.20 * min(dark_coverage * 2.2, 1.0)
        )
        damage_factor = float(np.clip(raw_factor, 0.05, 0.95))
        features = {
            "structural_damage_density": round(structural_damage, 4),
            "tilt_disorder_ratio": round(tilt_disorder, 4),
            "darkness_ratio": round(dark_coverage, 4),
        }

    else:  # garbage
        hsv = img_resized.convert("HSV")
        sat = np.array(hsv, dtype=np.float32)[:, :, 1] / 255.0
        # Color diversity across mixed waste packaging
        color_entropy = float(np.var(sat) * 10.0 + np.mean(np.std(arr, axis=2)) * 3.0)
        # Clutter density
        clutter_density = float(np.mean(grad_mag > 0.07))
        # Ground spread in lower frame
        ground_spread = float(np.mean(arr[90:, :, :] > 0.10))

        raw_factor = (
            0.40 * min(clutter_density * 2.2, 1.0)
            + 0.35 * min(ground_spread * 1.2, 1.0)
            + 0.25 * min(color_entropy * 1.5, 1.0)
        )
        damage_factor = float(np.clip(raw_factor, 0.05, 0.95))
        features = {
            "clutter_density": round(clutter_density, 4),
            "ground_spread_ratio": round(ground_spread, 4),
            "color_entropy": round(color_entropy, 4),
        }

    return {
        "damage_factor": round(damage_factor, 4),
        "features": features,
    }


def estimate_repair_cost(
    image_input,
    category="potholes",
    severity="Medium",
    severity_score=5.0,
    location_data=None,
):
    """
    Computes a realistic, deterministic repair cost range for civic maintenance.

    Args:
        image_input: Image file path, bytes, BytesIO, or PIL Image.
        category (str): Problem category ('potholes', 'streetlights', 'waterlogging', 'garbage').
        severity (str): 'Low', 'Medium', or 'High'.
        severity_score (float): 0.0 to 10.0 score from the severity model.
        location_data (str|dict|float, optional): Optional location string or factor. Default: 1.0.

    Returns:
        dict: Full estimation breakdown adhering to API contract.
    """
    cat_key = category.lower() if category and category.lower() in COST_CALIBRATION_CONFIG else "potholes"
    sev_key = severity.capitalize() if severity and severity.capitalize() in ["Low", "Medium", "High"] else "Medium"

    # Step 1: Base Category & Severity Bracket
    cat_config = COST_CALIBRATION_CONFIG.get(cat_key, COST_CALIBRATION_CONFIG["potholes"])
    tier_info = cat_config.get(sev_key, cat_config["Medium"])
    min_base = float(tier_info["min_base"])
    max_base = float(tier_info["max_base"])
    work_scope = tier_info["work_scope"]
    span = max_base - min_base

    # Step 2: Location Factor (Neutral 1.0 default, extensible for real rates)
    loc_factor = 1.0
    if isinstance(location_data, (int, float)):
        loc_factor = float(location_data)
    elif isinstance(location_data, str):
        try:
            loc_factor = float(location_data.strip())
        except ValueError:
            loc_factor = LOCATION_RATE_TABLE.get(location_data.strip().lower(), 1.0)
    elif isinstance(location_data, dict):
        loc_factor = float(location_data.get("rate_factor", location_data.get("location_factor", 1.0)))

    # Step 3: Extract Visual Damage Extent
    vis_res = extract_visual_damage_features(image_input, category=cat_key)
    damage_factor = vis_res["damage_factor"]
    damage_features = vis_res["features"]

    # Step 4: Severity Model Score Intra-Tier Normalization
    bounds = SEVERITY_TIER_BOUNDS.get(sev_key, {"min_score": 3.5, "max_score": 6.8})
    score_clamped = min(max(float(severity_score), bounds["min_score"]), bounds["max_score"])
    sev_prog = (score_clamped - bounds["min_score"]) / max((bounds["max_score"] - bounds["min_score"]), 1e-4)
    sev_prog = float(np.clip(sev_prog, 0.0, 1.0))

    # Step 5: Controlled Blended Extent Indicator
    # 50% severity model continuous prediction + 50% direct visual extent metrics
    extent_factor = float(np.clip(0.50 * sev_prog + 0.50 * damage_factor, 0.05, 0.95))

    # Step 6: Target Cost & Spread Estimation
    target_mid = min_base + extent_factor * span

    # Half-width interval (proportional to repair scope, ~18-22% of bracket span)
    half_width = max(span * 0.20, 500.0)

    raw_min = (target_mid - half_width) * loc_factor
    raw_max = (target_mid + half_width) * loc_factor

    # Bound safety checks
    effective_min_base = min_base * loc_factor
    effective_max_base = max_base * loc_factor

    clamped_min = max(effective_min_base, raw_min)
    clamped_max = min(effective_max_base, raw_max)

    # Step 7: Realistic Civic Rounding (nearest ₹100)
    final_min = int(round(clamped_min / COST_ROUNDING_STEP) * COST_ROUNDING_STEP)
    final_max = int(round(clamped_max / COST_ROUNDING_STEP) * COST_ROUNDING_STEP)

    # Ensure minimum sensible range window (at least ₹500 span)
    if final_max <= final_min + 400:
        if final_max + 500 <= int(effective_max_base):
            final_max += 500
        elif final_min - 500 >= int(effective_min_base):
            final_min -= 500
        else:
            final_max = final_min + 500

    # Ensure sanity checks
    final_min = max(int(effective_min_base), final_min)
    final_max = max(final_min + 500, final_max)

    label = f"₹{final_min:,} – ₹{final_max:,}"

    return {
        "estimated_cost": {
            "min": final_min,
            "max": final_max,
            "currency": "INR",
            "label": label,
        },
        "category": cat_key,
        "severity": sev_key,
        "severity_score": round(float(severity_score), 1),
        "damage_factor": round(float(damage_factor), 3),
        "extent_factor": round(float(extent_factor), 3),
        "location_factor": round(float(loc_factor), 2),
        "cost_estimation_method": COST_ESTIMATION_METHOD,
        "work_scope": work_scope,
        "damage_features": damage_features,
    }
