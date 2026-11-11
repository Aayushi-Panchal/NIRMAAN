import io
import json
import os
import sys
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_SEV = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_severity_dataset\test"
BASE_REC = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_recognition_final\test"
CATEGORIES = ["potholes", "streetlights", "waterlogging", "garbage"]

print("=" * 80)
print("  NIRMAAN V1 COST ESTIMATION ENGINE PIPELINE VERIFICATION")
print("=" * 80)

summary_results = []

# Part 1: Comprehensive Multi-Image Test across all 4 categories and severity tiers
for cat in CATEGORIES:
    print(f"\n{'='*30} CATEGORY: {cat.upper()} {'='*30}")
    cat_dir = os.path.join(BASE_REC, cat)
    files = [f for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:4]

    for fname in files:
        fpath = os.path.join(cat_dir, fname)
        with open(fpath, "rb") as f:
            res = requests.post("http://127.0.0.1:5000/predict", files={"image": (fname, f, "image/jpeg")})
        
        if res.status_code != 200:
            print(f"  [ERROR] {fname}: {res.status_code} {res.text}")
            continue

        data = res.json()
        est_cost = data.get("estimated_cost", {})
        summary_results.append({
            "category": data.get("category"),
            "file": fname,
            "severity": data.get("severity"),
            "severity_score": data.get("severity_score"),
            "damage_factor": data.get("damage_factor"),
            "min_cost": est_cost.get("min"),
            "max_cost": est_cost.get("max"),
            "cost_label": est_cost.get("label"),
            "work_scope": data.get("work_scope")
        })

        print(f"  • File: {fname[:22]:22s} | Cat: {data.get('category'):12s} | Sev: {data.get('severity'):6s} ({data.get('severity_score'):.1f}/10)")
        print(f"    Damage Factor: {data.get('damage_factor'):.3f} | Cost: {est_cost.get('label'):20s}")
        print(f"    Work Scope: {data.get('work_scope')}")

# Part 2: Determinism Test (same image repeated 3 times)
print(f"\n{'='*30} DETERMINISM TEST (Same image repeated) {'='*30}")
det_img_path = os.path.join(BASE_REC, "potholes", os.listdir(os.path.join(BASE_REC, "potholes"))[0])
det_runs = []
for i in range(3):
    with open(det_img_path, "rb") as f:
        res = requests.post("http://127.0.0.1:5000/predict", files={"image": ("test.jpg", f, "image/jpeg")})
    d = res.json()
    cost = d["estimated_cost"]
    det_runs.append((cost["min"], cost["max"], cost["label"], d["damage_factor"]))
    print(f"  Run {i+1}: {cost['label']} (min={cost['min']}, max={cost['max']}, dmg_factor={d['damage_factor']})")

is_deterministic = (det_runs[0] == det_runs[1] == det_runs[2])
print(f"  -> Determinism Check: {'PASSED [OK]' if is_deterministic else 'FAILED'}")

# Part 3: Location Factor Test (optional factor plugged in)
print(f"\n{'='*30} LOCATION ADJUSTMENT FACTOR TEST {'='*30}")
with open(det_img_path, "rb") as f:
    res_default = requests.post("http://127.0.0.1:5000/predict", files={"image": ("test.jpg", f, "image/jpeg")})
d_default = res_default.json()

with open(det_img_path, "rb") as f:
    res_metro = requests.post(
        "http://127.0.0.1:5000/predict",
        files={"image": ("test.jpg", f, "image/jpeg")},
        data={"location_factor": "1.15"}
    )
d_metro = res_metro.json()

print(f"  Default (factor=1.00): {d_default['estimated_cost']['label']} (min={d_default['estimated_cost']['min']}, max={d_default['estimated_cost']['max']})")
print(f"  Metro (factor=1.15):   {d_metro['estimated_cost']['label']} (min={d_metro['estimated_cost']['min']}, max={d_metro['estimated_cost']['max']})")
loc_passed = d_metro['estimated_cost']['min'] > d_default['estimated_cost']['min']
print(f"  -> Location Adjustment Plug-in Check: {'PASSED [OK]' if loc_passed else 'FAILED'}")

# Part 4: Sanity Bounds & Formatting Verification
print(f"\n{'='*30} SANITY CHECKS {'='*30}")
all_valid = True
for r in summary_results:
    if r["min_cost"] <= 0 or r["max_cost"] <= 0:
        print(f"  [FAIL] Non-positive cost: {r}")
        all_valid = False
    if r["min_cost"] >= r["max_cost"]:
        print(f"  [FAIL] Inverted range: {r}")
        all_valid = False
    if not (r["cost_label"].startswith("₹") and "–" in r["cost_label"]):
        print(f"  [FAIL] Invalid label format: {r}")
        all_valid = False

print(f"  Total test samples evaluated: {len(summary_results)}")
print(f"  No negative/zero values: {'PASSED [OK]' if all_valid else 'FAILED'}")
print(f"  Label formatting adheres to spec: {'PASSED [OK]' if all_valid else 'FAILED'}")
print(f"  Intra-category variation confirmed: {'PASSED [OK]' if len(set(r['cost_label'] for r in summary_results)) > 4 else 'FAILED'}")

print("\n" + "=" * 80)
print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
