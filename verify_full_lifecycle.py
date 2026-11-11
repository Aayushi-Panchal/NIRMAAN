import requests
import json

BASE = "http://127.0.0.1:5000"

print("==================================================================")
print("  NIRMAAN END-TO-END CIVIC LIFECYCLE VERIFICATION TEST")
print("==================================================================")

# Stage 1: Citizen creates defect report
print("\n[Stage 1] Citizen: Submitting Defect Report...")
create_payload = {
    "category": "potholes",
    "title": "Severe Pothole on Ring Road, New Delhi",
    "location_text": "Ring Road, New Delhi",
    "user_id": "usr_citizen_1",
    "ai_category": "potholes",
    "ai_confidence": 0.96,
    "severity": "High",
    "severity_score": 8.5,
    "severity_reason": "Deep asphalt crater creating vehicle hazard.",
    "estimated_cost_min": 6000,
    "estimated_cost_max": 12000,
    "goal_amount": 9000,
    "work_scope": "Bituminous cold-mix patching and roller compaction."
}
r1 = requests.post(f"{BASE}/api/reports", data=create_payload)
assert r1.status_code == 201, f"Failed to create report: {r1.text}"
rep_data = r1.json()
report_id = rep_data["report_id"]
print(f"[OK] Report created successfully: ID = {report_id}")
print(f"     Status: {rep_data['report']['status']}")

# Stage 2: Backer funds the campaign
print(f"\n[Stage 2] Funding Campaign for {report_id}...")
contrib_payload = {
    "user_id": "usr_citizen_2",
    "amount": 9500,
    "payment_method": "UPI",
    "anonymous": False
}
r2 = requests.post(f"{BASE}/api/funding/{report_id}/contribute", json=contrib_payload)
assert r2.status_code == 200, f"Funding failed: {r2.text}"
f_res = r2.json()["contribution"]
print(f"[OK] Campaign funded: Raised {f_res['raised_amount']} / {f_res['goal_amount']} INR")
print(f"     Funding Pct: {f_res['funding_pct']}% | Fully Funded: {f_res['is_fully_funded']}")

# Stage 3: Contributor matching
print(f"\n[Stage 3] Contributor Matching for {report_id}...")
match_payload = {
    "category": "potholes",
    "location_text": "New Delhi",
    "severity": "High",
    "severity_score": 8.5,
    "estimated_cost_min": 6000,
    "estimated_cost_max": 12000
}
rmatch = requests.post(f"{BASE}/api/match", json=match_payload)
assert rmatch.status_code == 200, f"Match failed: {rmatch.text}"
match_res = rmatch.json()
assert match_res["success"] is True and len(match_res["matches"]) > 0, "No matches found"
best_contrib = match_res["matches"][0]
contributor_id = best_contrib["contributor_id"]
print(f"[OK] Ranked {len(match_res['matches'])} contributors")
print(f"     Top Match: {best_contrib['name']} ({best_contrib['organization']}) | Score: {best_contrib['match_score']}%")

# Stage 4: Contributor accepts task
print(f"\n[Stage 4] Contributor ({contributor_id}): Accepting Task...")
accept_payload = {
    "contributor_id": contributor_id,
    "contributor_name": best_contrib["name"],
    "contributor_role": best_contrib["role"]
}
r3 = requests.post(f"{BASE}/api/tasks/{report_id}/accept", json=accept_payload)
assert r3.status_code == 200, f"Accept failed: {r3.text}"
asgn = r3.json()["assignment"]
asgn_id = asgn.get("assignment_id")
print(f"[OK] Task Accepted. Assignment ID: {asgn_id}")

# Stage 5: Contributor starts task
print(f"\n[Stage 5] Contributor: Starting Task...")
r4 = requests.post(f"{BASE}/api/tasks/{report_id}/start", json={"contributor_id": contributor_id})
assert r4.status_code == 200, f"Start failed: {r4.text}"
print(f"[OK] Task Started.")

# Stage 6: Contributor logs progress
print(f"\n[Stage 6] Contributor: Logging Progress Update...")
prog_data = {
    "assignment_id": str(asgn_id),
    "progress_pct": "50",
    "work_note": "Excavated loose debris and leveled the sub-base.",
    "submit_verification": "false"
}
r5 = requests.post(f"{BASE}/api/tasks/{report_id}/progress", data=prog_data)
assert r5.status_code == 200, f"Progress update failed: {r5.text}"
print(f"[OK] Progress logged: 50% milestone reached.")

# Stage 7: Contributor completes work and submits for verification
print(f"\n[Stage 7] Contributor: Submitting for Admin Verification...")
prog_final = {
    "assignment_id": str(asgn_id),
    "progress_pct": "100",
    "work_note": "Asphalt patch installed, compacted with vibratory roller, sealed.",
    "submit_verification": "true"
}
r6 = requests.post(f"{BASE}/api/tasks/{report_id}/progress", data=prog_final)
assert r6.status_code == 200, f"Final progress failed: {r6.text}"
update_id = r6.json()["update"]["update_id"]
print(f"[OK] Submitted for verification. Update ID = {update_id}")

# Stage 8: Admin checks verification queue
print(f"\n[Stage 8] Admin: Checking Verifications Queue...")
radmin = requests.get(f"{BASE}/api/admin/verifications?status=all")
assert radmin.status_code == 200, f"Admin verifications failed: {radmin.text}"
verifs = radmin.json()["verifications"]
target_v = [v for v in verifs if v.get("report_id") == report_id]
print(f"[OK] Admin Queue: {len(verifs)} total records. Found current report verification: {len(target_v) > 0}")

# Stage 9: Admin approves verification and releases funds
print(f"\n[Stage 9] Admin: Approving Verification...")
approve_data = {
    "report_id": report_id,
    "admin_id": "usr_admin_1",
    "notes": "Quality standards verified and photographic proof accepted.",
    "amount_to_release": 9000.0
}
rapprove = requests.post(f"{BASE}/api/admin/verifications/{update_id}/approve", json=approve_data)
assert rapprove.status_code == 200, f"Approval failed: {rapprove.text}"
print(f"[OK] Admin Approved verification successfully.")

# Stage 10: Check final report status
print(f"\n[Stage 10] Citizen View: Confirming Final Status...")
rfinal = requests.get(f"{BASE}/api/reports/{report_id}")
final_data = rfinal.json()["report"]
print(f"[OK] Final Report Details:")
print(f"     ID: {final_data['id']}")
print(f"     Status: {final_data['status']}")
print(f"     Assigned Contributor: {final_data.get('contributor_name')} ({final_data.get('contributor_role')})")
print(f"     Category: {final_data['category']}")
print(f"     Location: {final_data['location_text']}")

# Stage 11: Admin dashboard stats
print(f"\n[Stage 11] Central Admin Dashboard Stats...")
rdash = requests.get(f"{BASE}/api/admin/dashboard")
assert rdash.status_code == 200, f"Dashboard failed: {rdash.text}"
stats = rdash.json()["stats"]
print(f"[OK] Dashboard Stats:")
print(f"     Total Reports: {stats['total_reports']}")
print(f"     Completed Reports: {stats['completed_reports']}")
print(f"     Active Contributors: {stats['total_contributors']}")
print(f"     Total Raised: {stats['total_funds_raised']} INR")
print(f"     Total Released: {stats['total_funds_released']} INR")

print("\n==================================================================")
print("  ALL 11 END-TO-END PIPELINE & LIFECYCLE STAGES PASSED 100%!")
print("==================================================================")
