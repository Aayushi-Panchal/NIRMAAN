import urllib.request
import json

# Test the live Flask API
base = "http://127.0.0.1:5000"

# Test rewards for top user
url = f"{base}/api/rewards?user_id=usr_contractor_1"
try:
    resp = urllib.request.urlopen(url, timeout=5)
    data = json.loads(resp.read())
    print("=== /api/rewards?user_id=usr_contractor_1 ===")
    print(f"  success: {data.get('success')}")
    r = data.get('rewards', {})
    print(f"  user_name: {r.get('user_name')}")
    print(f"  tier: {r.get('tier')}")
    print(f"  points: {r.get('points')}")
    print(f"  rank: {r.get('rank')}")
    print(f"  completed_fixes: {r.get('completed_fixes')}")
    print(f"  problems_backed: {r.get('problems_backed')}")
    lb = data.get('leaderboard', [])
    print(f"\n  leaderboard entries returned: {len(lb)}")
    for e in lb[:5]:
        print(f"    #{e.get('rank'):02d} {e.get('user_name'):<28} {e.get('tier'):<10} {e.get('points')} pts")
except Exception as ex:
    print(f"ERROR: {ex}")
