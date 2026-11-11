import sys, io, os, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_severity_dataset\test"
CATS = ["potholes", "waterlogging", "streetlights", "garbage"]

print("=" * 78)
print("  EXTENDED MULTI-IMAGE TEST (2-3 per category across Low/Medium/High labels)")
print("=" * 78)

for cat in CATS:
    print(f"\n{'='*40} {cat.upper()} {'='*40}")
    for sev in ["Low", "Medium", "High"]:
        sev_dir = os.path.join(BASE, cat, sev)
        if not os.path.exists(sev_dir):
            continue
        files = [f for f in sorted(os.listdir(sev_dir)) if f.lower().endswith(('.jpg','.jpeg','.png'))][:2]
        for fname in files:
            fpath = os.path.join(sev_dir, fname)
            with open(fpath, 'rb') as f:
                r = requests.post("http://127.0.0.1:5000/predict", files={"image": (fname, f, "image/jpeg")})
            d = r.json()
            probs = d.get("probabilities", {})
            print(f"  [Pseudo-label:{sev:6s}] {fname[:35]:35s} -> ML Severity: {d['severity']:6s} ({d['severity_score']}/10)  Probs: L={probs.get('Low',0):.2f} M={probs.get('Medium',0):.2f} H={probs.get('High',0):.2f}")
