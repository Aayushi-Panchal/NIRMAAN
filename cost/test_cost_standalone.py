import os
import io
import sys
try:
    from cost.cost_estimator import estimate_repair_cost
except ImportError:
    from cost_estimator import estimate_repair_cost

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_recognition_final\test"
for cat in ["potholes", "streetlights", "waterlogging", "garbage"]:
    cat_dir = os.path.join(base, cat)
    files = [f for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(('.jpg', '.png'))][:3]
    print(f"\n{'='*25} CATEGORY: {cat.upper()} {'='*25}")
    for fname in files:
        fpath = os.path.join(cat_dir, fname)
        for sev, score in [("Low", 2.2), ("Medium", 5.0), ("High", 8.4)]:
            res = estimate_repair_cost(fpath, category=cat, severity=sev, severity_score=score)
            label = res["estimated_cost"]["label"]
            dmg = res["damage_factor"]
            ext = res["extent_factor"]
            print(f"  {fname[:20]:20s} | {sev:6s} (score {score:.1f}) -> {label:20s} [dmg={dmg:.3f}, ext={ext:.3f}]")

print("\n--- Testing repeat determinism (same image 3 times) ---")
test_img = os.path.join(base, "potholes", os.listdir(os.path.join(base, "potholes"))[0])
for i in range(3):
    r = estimate_repair_cost(test_img, category="potholes", severity="Medium", severity_score=5.2)
    print(f"Run {i+1}: {r['estimated_cost']['label']} (min={r['estimated_cost']['min']}, max={r['estimated_cost']['max']})")
