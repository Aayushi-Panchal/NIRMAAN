import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["garbage", "potholes", "streetlights", "waterlogging"]

CLASS_DISPLAY_NAMES = {
    "garbage": "Garbage — Uncollected Waste / Dump",
    "potholes": "Pothole — Road Surface Damage",
    "streetlights": "Streetlight — Non-Functional / Damaged Light",
    "waterlogging": "Waterlogging — Blocked Drainage / Street Flooding"
}

# Contextual metadata to keep Nirmaan UI cards rich and realistic
SEVERITY_METADATA = {
    "potholes": {
        "severity": "High · 8.4/10",
        "severityClass": "high",
        "reason": "Road surface depression posing immediate vehicular hazard and accident risk",
        "cost": "₹6,500 – ₹9,200",
        "matches": [
            ["Ramesh Road Contractors", "95% match · 1.2km"],
            ["PWD-Registered · V. Kumar", "89% match · 2.1km"],
            ["Sewa Bharti NGO", "81% match · civic partner"]
        ]
    },
    "waterlogging": {
        "severity": "High · 8.8/10",
        "severityClass": "high",
        "reason": "Standing water obstructing roadway, breeding vector hazard & drain backup",
        "cost": "₹10,500 – ₹13,000",
        "matches": [
            ["Municipal Drainage Works", "93% match · 1.5km"],
            ["Rajiv Colony RWA Response Team", "85% match · local"],
            ["Flood Response Taskforce", "80% match · priority"]
        ]
    },
    "streetlights": {
        "severity": "Medium · 5.6/10",
        "severityClass": "mid",
        "reason": "Pedestrian safety concern after dark, low night-time visibility area",
        "cost": "₹8,500 – ₹11,000",
        "matches": [
            ["Verma Electricals & Lighting", "92% match · 0.8km"],
            ["City Volunteer Electricians", "78% match · community"],
            ["Discom Certified Crew", "74% match · 2.4km"]
        ]
    },
    "garbage": {
        "severity": "Medium · 6.2/10",
        "severityClass": "mid",
        "reason": "Accumulated municipal solid waste posing public hygiene and odor concern",
        "cost": "₹3,500 – ₹5,000",
        "matches": [
            ["Swachh City Sanitation Unit", "96% match · 1.0km"],
            ["Green Clean Civic Volunteers", "88% match · community"],
            ["Zonal Waste Disposal Dep.", "82% match · municipal"]
        ]
    }
}

class NirmaanPredictor:
    def __init__(self, model_path=None, mapping_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "nirmaan_recognition_best.pth")
        if mapping_path is None:
            mapping_path = os.path.join(base_dir, "nirmaan_class_mapping.json")

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NirmaanPredictor] Initializing on device: {self.device}")

        # Load class mapping
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                data = json.load(f)
                self.classes = data.get("classes", CLASS_NAMES)
                idx_to_class_raw = data.get("idx_to_class", {})
                self.idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
        else:
            self.classes = CLASS_NAMES
            self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        # Load ResNet-50 architecture
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(self.classes))
        )

        # Load weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"[NirmaanPredictor] Checkpoint loaded successfully from: {self.model_path}")

        # Standard preprocessing matching training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_image(self, image_source):
        """
        Accepts:
            image_source: file path (str), PIL.Image, or file-like bytes buffer
        Returns:
            dict containing category, display_name, confidence, class_id, metadata, probabilities
        """
        if isinstance(image_source, str):
            image = Image.open(image_source)
        elif isinstance(image_source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_source))
        elif hasattr(image_source, "read"):
            image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError("Unsupported image input type")

        # Convert to RGB (handles RGBA, Grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probabilities, dim=0)

        class_id = int(top_idx.item())
        category = self.idx_to_class.get(class_id, "unknown")
        confidence = float(top_prob.item())

        all_probs = {
            self.idx_to_class[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(self.classes))
        }

        meta = SEVERITY_METADATA.get(category, {
            "severity": "Medium",
            "severityClass": "mid",
            "reason": "Civic problem detected by Nirmaan vision model",
            "cost": "₹5,000 – ₹10,000",
            "matches": []
        })

        return {
            "category": category,
            "display_name": CLASS_DISPLAY_NAMES.get(category, category.title()),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "class_id": class_id,
            "probabilities": all_probs,
            "metadata": meta,
            "device": str(self.device)
        }

# Global singleton helper
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = NirmaanPredictor()
    return _predictor_instance

if __name__ == "__main__":
    # Test on a test sample image
    predictor = get_predictor()
    test_dir = os.path.join(os.path.dirname(__file__), "nirmaan_recognition_final", "test")
    for category in ["potholes", "streetlights", "waterlogging", "garbage"]:
        cat_dir = os.path.join(test_dir, category)
        if os.path.exists(cat_dir):
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if files:
                sample_file = os.path.join(cat_dir, files[0])
                res = predictor.predict_image(sample_file)
                print(f"Test sample: {category}/{files[0]} -> Predicted: {res['category']} ({res['confidence_percentage']})")
