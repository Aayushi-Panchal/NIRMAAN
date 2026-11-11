import sys
sys.path.insert(0, 'database')
from database import compute_dynamic_rewards

uid = "usr_contractor_1"
user_r, lb = compute_dynamic_rewards(target_user_id=uid, limit=10)
print("user_rewards dict:")
print(user_r)
print()
print("leaderboard[0] dict:")
print(lb[0] if lb else "EMPTY")
