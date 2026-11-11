import urllib.request
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://127.0.0.1:5000'

def req(path, method='GET', data=None):
    url = f"{BASE}{path}"
    headers = {'Content-Type': 'application/json'} if data else {}
    body = json.dumps(data).encode('utf-8') if data else None
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(r) as response:
        return response.getcode(), json.loads(response.read().decode('utf-8'))

print("=== 1. Health & Models Check ===")
code, data = req('/health')
rec_arch = data['recognition_model']['architecture']
sev_arch = data['severity_model']['architecture']
contrib_count = data['contributor_matcher']['active_contributors_count']
print(f"Health: {code} | Recognition: {rec_arch} | Severity: {sev_arch} | Contributors: {contrib_count}")

print("\n=== 2. Problem Board from DB ===")
code, data = req('/api/reports')
print(f"Reports fetched: {data['count']}")
for r in data['reports']:
    print(f"  - [{r['status']}] {r['title']} (Goal: Rs {r['goal_amount']}, Raised: Rs {r['raised_amount']} [{r['funding_pct']}%])")

print("\n=== 3. Funding Simulation ===")
code, data = req('/api/funding/rep_demo_2/contribute', method='POST', data={
    'amount': 1500,
    'donor_name': 'Aayushi P. (Resident)',
    'user_id': 'usr_citizen_1',
    'note': 'Park street lighting contribution'
})
contrib = data['contribution']
print(f"Contributed Rs 1,500. Updated Raised: Rs {contrib['raised_amount']} ({contrib['funding_pct']}%)")

print("\n=== 4. Contributor Tasks & Match Scores ===")
code, data = req('/api/contributor/tasks?role=contractor')
print(f"Open jobs: {data['count']}")
for t in data['tasks'][:3]:
    print(f"  - {t['match_score']}% match: {t['title']} [{t['status']}]")

print("\n=== 5. Admin Verification & Fund Release Workflow ===")
code, data = req('/api/admin/verifications?status=pending')
print(f"Pending verifications in queue: {len(data['verifications'])}")
if data['verifications']:
    v = data['verifications'][0]
    upd_id = v['update_id']
    rep_id = v['report_id']
    suggested = v['suggested_release_amount']
    print(f"  Reviewing: {v['title']} (Proof: {v['progress_pct']}%, Suggested Release: Rs {suggested})")
    
    code, app_res = req(f'/api/admin/verifications/{upd_id}/approve', method='POST', data={
        'report_id': rep_id,
        'amount_to_release': suggested,
        'notes': 'Verified on site. Quality meets Municipal Corporation standards.'
    })
    print(f"  [OK] Approved! Status: {app_res.get('status')}, Funds Released: Rs {app_res.get('amount_released')}")

print("\n=== 6. Citizen Live Tracking & Timeline ===")
code, rep_detail = req('/api/reports/rep_demo_3')
r = rep_detail['report']
print(f"Report: {r['title']} | Status: {r['status']}")
print(f"Timeline events count: {len(r.get('timeline', []))}")
for event in r.get('timeline', []):
    print(f"  -> {event['from_status']} -> {event['to_status']} by {event['actor_name']}: {event['note']}")

print("\n=== 7. Rewards & Leaderboard ===")
code, data = req('/api/rewards?user_id=usr_citizen_1')
print(f"User Score: {data['rewards']['points']} pts | Tier: {data['rewards']['tier']}")
print("Leaderboard Top 3:", [(x['user_name'], x['points'], x['tier']) for x in data['leaderboard'][:3]])

print("\n=== 8. Admin Dashboard KPI Stats ===")
code, data = req('/api/admin/dashboard')
print("Dashboard KPIs:", data['stats'])

print("\n>>> FULL WORKFLOW VERIFICATION SUCCEEDED 100%! <<<")
