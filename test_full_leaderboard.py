import urllib.request
import json

base = "http://127.0.0.1:5000"

# Test full leaderboard
url = f"{base}/api/rewards?user_id=usr_citizen_1"
resp = urllib.request.urlopen(url, timeout=5)
data = json.loads(resp.read().decode('utf-8'))

print("=== USER: Priya Malhotra (usr_citizen_1) ===")
r = data['rewards']
print(f"  Name: {r['user_name']}")
print(f"  Rank: #{r['rank']}")
print(f"  Tier: {r['tier']}")
print(f"  Points: {r['points']}")
print(f"  Role: {r['role']}")
print(f"  Contributed: Rs.{r['contributed_amount']:.0f}")
print(f"  Backed: {r['problems_backed']}")
print(f"  Fixes: {r['completed_fixes']}")

print()
print("=== FULL LEADERBOARD ===")
for e in data['leaderboard']:
    print(f"  #{e['rank']:02d} {e['user_name']:<30} {e['tier']:<10} {e['points']:>5} pts  fixes:{e['completed_fixes']}  backed:{e['problems_backed']}")
