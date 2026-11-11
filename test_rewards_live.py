import sys
sys.path.insert(0, 'database')
from database import compute_dynamic_rewards

print('=== LEADERBOARD (top 10) ===')
_, lb = compute_dynamic_rewards(limit=10)
for item in lb:
    print(f"#{item['rank']:02d} {item['user_name']:<32} {item['tier']:<10} {item['points']:>6} pts  fixes:{item['completed_fixes']}  backed:{item['problems_backed']}  contributed:Rs.{item['contributed_amount']:.0f}")

print()
if lb:
    uid = lb[0]['user_id']
    user_r, _ = compute_dynamic_rewards(target_user_id=uid, limit=1)
    print(f"=== USER REWARDS for top user ({uid}) ===")
    for k, v in user_r.items():
        print(f"  {k}: {v}")
