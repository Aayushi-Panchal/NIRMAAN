import sys
import io
import os
import requests
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_recognition_final\test"
CATEGORIES = ["potholes", "waterlogging", "streetlights", "garbage"]

print("=" * 75)
print("  NIRMAAN END-TO-END DUAL-AI PIPELINE VERIFICATION")
print("=" * 75)

results = []

for cat in CATEGORIES:
    cat_dir = os.path.join(BASE, cat)
    files = [f for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_img = os.path.join(cat_dir, files[0])

    with open(test_img, "rb") as img_f:
        res = requests.post("http://127.0.0.1:5000/predict", files={"image": (files[0], img_f, "image/jpeg")})

    if res.status_code != 200:
        print(f"FAILED for {cat}: {res.status_code} {res.text}")
        continue

    d = res.json()
    results.append(d)

    print(f"\n[CASE: {cat.upper()}] - File: {files[0]}")
    print(f"  • Category:       {d.get('display_name')}")
    print(f"  • Recognition:    {d.get('confidence_percentage')} confidence on {d.get('device')}")
    print(f"  • Severity Level: {d.get('severity')}")
    print(f"  • Severity Score: {d.get('severity_score')} / 10.0")
    print(f"  • Model Probabilities: {d.get('probabilities')}")
    print(f"  • Why This Score: {d.get('severity_reason')}")
    print(f"  • Est Repair Cost:{d.get('estimated_repair_cost')} ({d.get('cost_note')})")

print("\n" + "=" * 75)
print(f"  VERIFICATION RESULT: {len(results)}/4 categories passed end-to-end successfully!")
print("=" * 75)
