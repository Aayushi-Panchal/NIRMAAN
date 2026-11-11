import urllib.request
import json

url = "http://127.0.0.1:5000/api/rewards?user_id=usr_contractor_1"
resp = urllib.request.urlopen(url, timeout=5)
data = json.loads(resp.read().decode('utf-8'))
print("=== rewards key ===")
print(json.dumps(data.get('rewards'), indent=2))
print()
print("=== leaderboard[0] ===")
lb = data.get('leaderboard', [])
if lb:
    print(json.dumps(lb[0], indent=2))
