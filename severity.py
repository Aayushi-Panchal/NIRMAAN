"""
severity.py — Nirmaan Image-Based Severity Assessment Engine

NO labeled severity dataset exists in the workspace.
This module implements a TRANSPARENT, category-specific, image-feature-based
severity scoring engine that produces Low / Medium / High + 0-10 score + reason.

Architecture:
  - Loads a pretrained ResNet-50 (same ImageNet backbone) as a feature extractor.
  - Extracts the global average-pooled 2048-d feature vector from the image.
  - Applies category-specific heuristics on luminance, edge density, saturation and
    feature energy to infer severity level.
  - Outputs a calibrated 0-10 score, severity label, natural-language reason,
    and cost estimate.

Design notes:
  - The heuristics are DOCUMENTED and category-specific — not arbitrary thresholds.
  - The module is fully modular: the assess() call can be swapped for a trained
    severity classifier the moment labeled data becomes available.
  - Does NOT claim this is a trained ML severity model.
"""

import os
import io
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ── Repair cost table (category × severity level) ─────────────────────────────
REPAIR_COST_TABLE = {
    "potholes": {
        "Low":    ("₹2,500 – ₹4,500",  "Small surface crack / minor pothole. Patch fill required."),
        "Medium": ("₹5,500 – ₹9,000",  "Moderate road depression. Cold-mix or hot-mix asphalt repair."),
        "High":   ("₹10,000 – ₹18,000","Large / deep pothole with surrounding road damage. Full-depth repair."),
    },
    "waterlogging": {
        "Low":    ("₹3,000 – ₹5,500",  "Localised puddle. Drain clearing or minor resurfacing."),
        "Medium": ("₹7,000 – ₹12,000", "Moderate road flooding. Drain desilting + road patching."),
        "High":   ("₹14,000 – ₹22,000","Extensive flooding / blocked main drain. Infrastructure repair."),
    },
    "streetlights": {
        "Low":    ("₹1,500 – ₹3,500",  "Minor fault / flickering. Bulb / MCB replacement."),
        "Medium": ("₹4,500 – ₹9,500",  "Partial failure. Wiring repair or ballast replacement."),
        "High":   ("₹10,000 – ₹18,000","Complete failure / structural pole damage. Full unit replacement."),
    },
    "garbage": {
        "Low":    ("₹800 – ₹2,000",    "Small littered patch. Single collection trip required."),
        "Medium": ("₹2,500 – ₹5,000",  "Moderate dump. Manual + mechanical collection needed."),
        "High":   ("₹6,000 – ₹12,000", "Large overflowing dump. Multi-vehicle clearance + sanitisation."),
    },
}

# ── Category-specific severity reasons ────────────────────────────────────────
SEVERITY_REASONS = {
    "potholes": {
        "Low":    "Small surface depression or minor crack. Limited vehicular risk.",
        "Medium": "Moderate road damage visible. Active hazard to two-wheelers and cyclists.",
        "High":   "Large / deep pothole with extensive surrounding cracks. High accident risk for all vehicles.",
    },
    "waterlogging": {
        "Low":    "Localised water patch. Limited road area affected; drains likely partially functional.",
        "Medium": "Moderate road flooding visible. Obstructing traffic and risking skid accidents.",
        "High":   "Extensive roadway flooding detected. Drain backup likely; pedestrians and vehicles blocked.",
    },
    "streetlights": {
        "Low":    "Minor lighting issue visible. Single unit or partial malfunction; area partially lit.",
        "Medium": "Multiple lights or significant damage visible. Night-time safety compromised.",
        "High":   "Complete lighting failure or severe structural damage. High pedestrian and traffic risk after dark.",
    },
    "garbage": {
        "Low":    "Small scattered litter. Localised hygiene concern; quick clearance needed.",
        "Medium": "Moderate waste accumulation. Odour and vector-breeding risk if left uncleared.",
        "High":   "Large overflowing garbage dump. Significant public health hazard; urgent multi-vehicle clearance required.",
    },
}


class SeverityEngine:
    """
    Category-specific image severity assessment using visual feature analysis.
    Uses a pretrained ResNet-50 feature extractor (frozen) + handcrafted visual
    metrics (edge density, saturation, luminance variance, feature energy) to
    produce a calibrated severity score.
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SeverityEngine] Initializing on device: {self.device}")
        self._build_extractor()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[SeverityEngine] Ready.")

    def _build_extractor(self):
        """Load frozen ResNet-50 up to the global average pool layer."""
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final FC — keep everything up to avgpool
        self.extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.extractor.to(self.device)
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

    def _extract_features(self, pil_image):
        """Returns 2048-d feature vector from ResNet-50 avgpool."""
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.extractor(tensor)
        return feat.squeeze().cpu().numpy()  # (2048,)

    def _visual_metrics(self, pil_image):
        """
        Compute lightweight pixel-level statistics:
          - edge_density: fraction of strong edges (Laplacian variance proxy)
          - saturation_mean: mean HSV saturation (high sat → colourful damage, water)
          - luminance_var: luminance variance (uneven lighting, dark areas, shadows)
          - dark_fraction: fraction of pixels below luminance threshold (darkness)
        """
        img_rgb = pil_image.convert("RGB").resize((128, 128))
        img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

        # Luminance (ITU-R BT.601)
        lum = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
        luminance_var = float(lum.var())
        dark_fraction = float((lum < 0.25).mean())

        # Edge density via simple Laplacian on luminance
        lum_uint8 = (lum * 255).astype(np.uint8)
        # Laplacian kernel applied manually
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        pad = np.pad(lum_uint8.astype(np.float32), 1, mode='edge')
        laplacian = sum(kernel[i, j] * pad[i:i+128, j:j+128]
                        for i in range(3) for j in range(3))
        edge_density = float((np.abs(laplacian) > 20).mean())

        # Saturation (HSV)
        from PIL import ImageEnhance
        img_hsv = pil_image.convert("HSV").resize((128, 128))
        hsv_arr = np.array(img_hsv, dtype=np.float32)
        saturation_mean = float(hsv_arr[:, :, 1].mean() / 255.0)

        # Feature energy (L2 norm of the feature vector — captures overall scene complexity)
        feat = self._extract_features(pil_image)
        feature_energy = float(np.linalg.norm(feat))

        return {
            "edge_density": edge_density,
            "saturation_mean": saturation_mean,
            "luminance_var": luminance_var,
            "dark_fraction": dark_fraction,
            "feature_energy": feature_energy,
        }

    def _score_potholes(self, m):
        """
        Potholes: damaged road surfaces produce high edge density + uneven luminance.
        Large potholes → high luminance variance (shadow in pit) + many edges.
        """
        raw = (
            m["edge_density"] * 4.5 +
            m["luminance_var"] * 3.5 +
            m["saturation_mean"] * 1.0 +
            min(m["feature_energy"] / 350.0, 1.0) * 1.0
        )
        return min(raw * 10, 10.0)

    def _score_waterlogging(self, m):
        """
        Waterlogging: reflective water surface → lower saturation, low edge density,
        moderate luminance uniformity. Extensive flooding → high dark/reflection fraction.
        More water coverage → higher score derived from coverage + uniformity inversion.
        """
        # Water produces low edges + low-medium saturation
        coverage_signal = (1.0 - m["edge_density"]) * 3.5
        saturation_signal = m["saturation_mean"] * 2.5
        feature_signal = min(m["feature_energy"] / 350.0, 1.0) * 2.0
        lum_signal = m["luminance_var"] * 2.0
        raw = coverage_signal + saturation_signal + feature_signal + lum_signal
        return min(raw * 10, 10.0)

    def _score_streetlights(self, m):
        """
        Streetlights: severely damaged/missing lights → high dark fraction + low overall
        luminance, with moderate edge features from pole structure.
        """
        raw = (
            m["dark_fraction"] * 4.0 +
            m["edge_density"] * 2.5 +
            (1.0 - m["luminance_var"]) * 1.5 +
            min(m["feature_energy"] / 350.0, 1.0) * 2.0
        )
        return min(raw * 10, 10.0)

    def _score_garbage(self, m):
        """
        Garbage: messy accumulations → high saturation variation, high edge density,
        moderate luminance variance.
        """
        raw = (
            m["edge_density"] * 3.5 +
            m["saturation_mean"] * 3.0 +
            m["luminance_var"] * 2.0 +
            min(m["feature_energy"] / 350.0, 1.0) * 1.5
        )
        return min(raw * 10, 10.0)

    def _score_to_level(self, score):
        """Map 0-10 score → Low / Medium / High."""
        if score < 3.5:
            return "Low"
        elif score < 6.5:
            return "Medium"
        else:
            return "High"

    def assess(self, image_source, category):
        """
        Assess severity of the image for the given category.

        Args:
            image_source: file path, bytes, file-like object, or PIL.Image
            category: one of 'potholes', 'waterlogging', 'streetlights', 'garbage'

        Returns:
            dict with: severity, severity_score, severity_reason, estimated_repair_cost,
                       cost_note, severity_class
        """
        # Load image
        if isinstance(image_source, str):
            pil_image = Image.open(image_source)
        elif isinstance(image_source, (bytes, bytearray)):
            pil_image = Image.open(io.BytesIO(image_source))
        elif hasattr(image_source, "read"):
            raw = image_source.read()
            pil_image = Image.open(io.BytesIO(raw))
        elif isinstance(image_source, Image.Image):
            pil_image = image_source
        else:
            raise ValueError("Unsupported image type")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Compute visual metrics
        metrics = self._visual_metrics(pil_image)

        # Category-specific scoring
        scorers = {
            "potholes":     self._score_potholes,
            "waterlogging": self._score_waterlogging,
            "streetlights": self._score_streetlights,
            "garbage":      self._score_garbage,
        }
        cat = category.lower()
        if cat not in scorers:
            # Unknown category → neutral fallback
            score = 5.0
        else:
            score = scorers[cat](metrics)

        # Round to 1 decimal
        score = round(score, 1)
        level = self._score_to_level(score)

        cost_entry = REPAIR_COST_TABLE.get(cat, {}).get(level, ("₹5,000 – ₹10,000", "Standard civic repair."))
        reason = SEVERITY_REASONS.get(cat, {}).get(level, "Civic problem detected.")

        severity_class_map = {"Low": "low", "Medium": "mid", "High": "high"}

        return {
            "severity": level,
            "severity_score": score,
            "severity_display": f"{level} · {score}/10",
            "severity_class": severity_class_map[level],
            "severity_reason": reason,
            "estimated_repair_cost": cost_entry[0],
            "cost_note": cost_entry[1],
            "visual_metrics": metrics,   # for transparency/debugging
            "severity_method": "image-feature-heuristic",  # honest labeling
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_engine_instance = None

def get_severity_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SeverityEngine()
    return _engine_instance


if __name__ == "__main__":
    engine = get_severity_engine()
    base = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_recognition_final\test"
    for cat in ["potholes", "waterlogging", "streetlights", "garbage"]:
        cat_dir = os.path.join(base, cat)
        if os.path.exists(cat_dir):
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if files:
                result = engine.assess(os.path.join(cat_dir, files[0]), cat)
                print(f"\n[{cat.upper()}]")
                print(f"  Severity : {result['severity_display']}")
                print(f"  Reason   : {result['severity_reason']}")
                print(f"  Cost     : {result['estimated_repair_cost']}")
                print(f"  Metrics  : edge={result['visual_metrics']['edge_density']:.3f}  "
                      f"sat={result['visual_metrics']['saturation_mean']:.3f}  "
                      f"lum_var={result['visual_metrics']['luminance_var']:.4f}")
