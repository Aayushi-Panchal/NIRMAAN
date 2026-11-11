"""
severity_predictor.py
---------------------
Category-Conditioned Inference wrapper for NIRMAAN Severity Assessment Model.

Loads nirmaan_severity_best.pth (CategoryConditioned-ResNet18) into memory.

Given an image and predicted category, executes:
  1. ImageNet validation transform (224x224 RGB)
  2. ResNet-18 shared backbone feature extraction (512-dim embedding)
  3. Routing through the dedicated category-specific classification head:
     (e.g., potholes_head, waterlogging_head, streetlights_head, garbage_head)
  4. Computes class softmax probabilities: P(Low), P(Medium), P(High)
  5. Calibrates continuous severity score (0.0 - 10.0):
     Low  ≈ 1.0 - 3.5
     Med  ≈ 4.0 - 6.0
     High ≈ 7.0 - 10.0
  6. Returns severity level, calibrated score, contextual reason, and cost estimate.
"""

import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CATEGORIES = ["potholes", "waterlogging", "streetlights", "garbage"]
SEVERITY_CLASSES = ["Low", "Medium", "High"]

REPAIR_COST_TABLE = {
    "potholes": {
        "Low":    ("₹2,500 – ₹4,500",  "Minor surface depression; single-patch asphalt fill required."),
        "Medium": ("₹5,500 – ₹9,000",  "Moderate crater; cold-mix compaction and sub-base grading."),
        "High":   ("₹10,000 – ₹18,000","Severe road crater with structural cracking; full-depth asphalt restoration."),
    },
    "waterlogging": {
        "Low":    ("₹3,000 – ₹5,500",  "Localized shallow puddle; catch-basin desilting required."),
        "Medium": ("₹7,000 – ₹12,000", "Moderate street ponding; stormwater line clearing and grade adjustment."),
        "High":   ("₹14,000 – ₹24,000","Extensive roadway submersion; heavy municipal dewatering and drain rebuild."),
    },
    "streetlights": {
        "Low":    ("₹1,500 – ₹3,500",  "Minor fixture wear or luminaire failure; LED bulb/fuse replacement."),
        "Medium": ("₹4,500 – ₹9,500",  "Broken luminaire or wiring fault; standard fixture overhaul required."),
        "High":   ("₹10,000 – ₹19,000","Damaged/fallen pole or severe electrical hazard; complete pole replacement."),
    },
    "garbage": {
        "Low":    ("₹1,000 – ₹2,500",  "Localized curb waste; single-operator manual cleanup required."),
        "Medium": ("₹3,000 – ₹6,000",  "Moderate roadside garbage dump; tipper vehicle and sanitization spray."),
        "High":   ("₹7,500 – ₹15,000", "Massive open overflow dump obstructing road; multi-trip loader deployment."),
    },
}

SEVERITY_REASONS = {
    "potholes": {
        "Low":    "Minor road depression with limited surrounding crack propagation. Low immediate vehicular hazard.",
        "Medium": "Noticeable asphalt breakdown posing direct risk to two-wheelers and suspension damage to vehicles.",
        "High":   "Extensive road surface crater with deep cavities and structural edge fracturing. Critical accident hazard.",
    },
    "waterlogging": {
        "Low":    "Localized shallow puddle. Minor curb-side pedestrian inconvenience with limited traffic obstruction.",
        "Medium": "Significant standing water across travel lane. Submerged surface hazards and potential skid danger.",
        "High":   "Widespread roadway submersion blocking vehicle passage and causing severe drain backup risk.",
    },
    "streetlights": {
        "Low":    "Isolated luminaire failure or low lux emission. Limited impact on overall corridor visibility.",
        "Medium": "Visible damaged/non-functional streetlight compromising pedestrian and vehicular safety after dusk.",
        "High":   "Severely bent/collapsed pole or exposed wiring posing severe nocturnal safety and electrical hazards.",
    },
    "garbage": {
        "Low":    "Contained litter accumulation. Manageable civic hygiene concern requiring scheduled collection.",
        "Medium": "Significant waste overflow encroaching sidewalk with evident decomposition and odor risks.",
        "High":   "Large uncontrolled municipal dump obstructing thoroughfare with critical public health and vector risks.",
    },
}

class CategoryConditionedSeverityNet(nn.Module):
    def __init__(self, categories=CATEGORIES, num_classes=3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, num_classes)
            ) for cat in categories
        })

    def forward(self, x, categories):
        features = self.backbone(x).flatten(1)
        logits = torch.zeros(x.size(0), 3, device=x.device)
        for i, cat in enumerate(categories):
            head = self.heads.get(cat, self.heads["potholes"])
            logits[i] = head(features[i:i+1])
        return logits

    def forward_single(self, x, category):
        features = self.backbone(x).flatten(1)
        head = self.heads[category] if category in self.heads else self.heads["potholes"]
        return head(features)


class NirmaanSeverityPredictor:
    def __init__(self, model_path=None, device=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_dir, "nirmaan_severity_best.pth")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SeverityPredictor] Loading model on: {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = CategoryConditionedSeverityNet(categories=CATEGORIES, num_classes=3)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[SeverityPredictor] Category-Conditioned checkpoint loaded successfully from: {self.model_path}")
        else:
            print(f"[SeverityPredictor] WARNING: Checkpoint not found at {self.model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_source, category="potholes"):
        """
        Runs category-conditioned severity inference on image.
        Args:
            image_source: file path, bytes, or file-like stream
            category: recognized category string
        Returns:
            dict containing severity, severity_score, reason, cost, probs
        """
        if isinstance(image_source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_source))
        elif hasattr(image_source, "read"):
            img = Image.open(image_source)
        elif isinstance(image_source, str):
            img = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            raise ValueError("Unsupported image source type")

        if img.mode != "RGB":
            img = img.convert("RGB")

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        cat_key = category.lower() if category and category.lower() in CATEGORIES else "potholes"

        with torch.no_grad():
            outputs = self.model.forward_single(tensor, cat_key)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        p_low = float(probs[0])
        p_med = float(probs[1])
        p_high = float(probs[2])

        # Argmax class
        class_idx = int(probs.argmax())
        severity_label = SEVERITY_CLASSES[class_idx]

        # Continuous severity score calibrated:
        # Low  ≈ 1.0 - 3.5
        # Med  ≈ 4.0 - 6.0
        # High ≈ 7.0 - 10.0
        score = p_low * 2.0 + p_med * 5.0 + p_high * 8.4
        severity_score = float(round(min(max(score, 1.0), 9.9), 1))

        # Retrieve category-specific cost and reason
        cost_info = REPAIR_COST_TABLE.get(cat_key, {}).get(severity_label, ("₹5,000 – ₹10,000", "Standard civic repair."))
        reason = SEVERITY_REASONS.get(cat_key, {}).get(severity_label, f"{severity_label} severity civic problem detected.")

        class_style_map = {"Low": "low", "Medium": "mid", "High": "high"}

        return {
            "severity": severity_label,
            "severity_score": severity_score,
            "severity_display": f"{severity_label} · {severity_score}/10",
            "severity_class": class_style_map.get(severity_label, "mid"),
            "severity_reason": reason,
            "estimated_repair_cost": cost_info[0],
            "cost_note": cost_info[1],
            "probabilities": {
                "Low": round(p_low, 4),
                "Medium": round(p_med, 4),
                "High": round(p_high, 4)
            },
            "severity_model": f"CategoryConditioned-ResNet18 ({cat_key}_head)",
            "device": str(self.device)
        }

_severity_singleton = None

def get_severity_predictor():
    global _severity_singleton
    if _severity_singleton is None:
        _severity_singleton = NirmaanSeverityPredictor()
    return _severity_singleton
