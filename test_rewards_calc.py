import sqlite3

def compute_rewards_and_leaderboard(target_user_id=None, limit=10):
    conn = sqlite3.connect('database/nirmaan_contributors.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Fetch Users
    cur.execute('SELECT id, name, role, organization FROM users')
    users = {r['id']: dict(r) for r in cur.fetchall()}

    # 2. Fetch Funding totals and campaign counts
    cur.execute('''
        SELECT user_id, donor_name, SUM(amount) as total_funded,
               COUNT(DISTINCT campaign_id) as campaigns_count,
               COUNT(*) as tx_count
        FROM funding_transactions
        GROUP BY user_id
    ''')
    funding = {r['user_id']: dict(r) for r in cur.fetchall()}

    # 3. Fetch Completed assignments
    cur.execute('''
        SELECT contributor_id, contributor_name, contributor_role,
               SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed_fixes,
               SUM(CASE WHEN status IN ('ASSIGNED', 'IN_PROGRESS') THEN 1 ELSE 0 END) as active_jobs,
               COUNT(*) as total_assignments
        FROM assignments
        GROUP BY contributor_id
    ''')
    assignments = {r['contributor_id']: dict(r) for r in cur.fetchall()}

    # 4. Fetch Reports filed per user
    cur.execute('''
        SELECT user_id, COUNT(*) as reports_count
        FROM reports
        GROUP BY user_id
    ''')
    reports = {r['user_id']: dict(r) for r in cur.fetchall()}

    # 5. Fetch Existing ledger baseline
    cur.execute('SELECT * FROM rewards_ledger')
    ledger = {r['user_id']: dict(r) for r in cur.fetchall()}

    # Merge all entities
    all_uids = set(users.keys()) | set(funding.keys()) | set(assignments.keys()) | set(ledger.keys())

    leaderboard = []
    for uid in all_uids:
        u = users.get(uid, {})
        f = funding.get(uid, {})
        a = assignments.get(uid, {})
        r = reports.get(uid, {})
        l = ledger.get(uid, {})

        name = u.get('name') or a.get('contributor_name') or f.get('donor_name') or l.get('user_name') or uid
        role = u.get('role') or a.get('contributor_role') or ('contractor' if 'contractor' in uid else 'citizen')

        funded = float(f.get('total_funded') or l.get('contributed_amount') or 0.0)
        fixes = int(a.get('completed_fixes') or 0)
        reps = int(r.get('reports_count') or 0)
        camps = int(f.get('campaigns_count') or 0)

        # Base / historical values from ledger
        base_backed = int(l.get('problems_backed') or 0)
        base_points = int(l.get('points') or 0)

        # Dynamic calculation
        funding_pts = int(funded / 10)
        fix_pts = fixes * 500
        rep_pts = reps * 150

        calc_points = funding_pts + fix_pts + rep_pts
        total_points = max(calc_points, base_points)

        total_backed = max(camps + fixes + reps, base_backed)

        tier = 'Bronze'
        if total_points >= 3000:
            tier = 'Platinum'
        elif total_points >= 1500:
            tier = 'Gold'
        elif total_points >= 500:
            tier = 'Silver'

        leaderboard.append({
            'user_id': uid,
            'user_name': name,
            'role': role,
            'points': total_points,
            'contributed_amount': funded,
            'problems_backed': total_backed,
            'completed_fixes': fixes,
            'reports_filed': reps,
            'tier': tier
        })

    # Sort descending by points, then contributed_amount
    leaderboard.sort(key=lambda x: (x['points'], x['contributed_amount'], x['problems_backed']), reverse=True)

    # Assign dynamic ranks
    for idx, item in enumerate(leaderboard, 1):
        item['rank'] = idx

    # If target_user_id requested, find their rewards or compute default
    user_rewards = None
    if target_user_id:
        for item in leaderboard:
            if item['user_id'] == target_user_id:
                user_rewards = item
                break

    if not user_rewards and target_user_id:
        user_rewards = {
            'user_id': target_user_id,
            'user_name': users.get(target_user_id, {}).get('name', 'Community Member'),
            'role': users.get(target_user_id, {}).get('role', 'citizen'),
            'points': 0,
            'contributed_amount': 0.0,
            'problems_backed': 0,
            'completed_fixes': 0,
            'reports_filed': 0,
            'tier': 'Bronze',
            'rank': len(leaderboard) + 1
        }

    conn.close()
    return user_rewards, leaderboard[:limit]

if __name__ == '__main__':
    user_rewards, lb = compute_rewards_and_leaderboard('usr_citizen_1', 10)
    print('User rewards for usr_citizen_1:')
    for k, v in user_rewards.items():
        print(f'  {k}: {v}')
    print('\nTop Leaderboard:')
    for item in lb:
        print(f"#{item['rank']:2d}: {item['user_name']:28s} ({item['role']:10s} - {item['tier']:8s}) | {item['points']:5d} pts | {item['completed_fixes']:2d} fixes | {item['contributed_amount']:8.0f} INR | {item['problems_backed']:2d} backed")
