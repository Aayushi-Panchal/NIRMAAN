"""
Seed script for NIRMAAN Demo Database.
Populates standard demo users, realistic civic defect reports across all lifecycle statuses,
linked funding campaigns with multi-contributor transactions, assignments, progress proofs,
and verification records.
"""
import os
import shutil
from database import (
    init_db, upsert_user, create_report, add_funding_contribution,
    accept_task, start_task, submit_task_progress, admin_approve_verification,
    upsert_rewards, update_report_status, get_connection
)

def copy_sample_photos():
    """Ensures demo photos exist in uploads directories."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rep_dir = os.path.join(base_dir, "nirmaan-ui", "uploads", "reports")
    prog_dir = os.path.join(base_dir, "nirmaan-ui", "uploads", "progress")
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(prog_dir, exist_ok=True)

    categories = ["potholes", "streetlights", "waterlogging", "garbage"]
    for cat in categories:
        d = os.path.join(base_dir, "nirmaan dataset", "test", cat)
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                dst1 = os.path.join(rep_dir, f"sample_{cat}.jpg")
                if not os.path.exists(dst1):
                    shutil.copyfile(os.path.join(d, files[0]), dst1)
                if len(files) > 1:
                    dst2 = os.path.join(prog_dir, f"progress_{cat}.jpg")
                    if not os.path.exists(dst2):
                        shutil.copyfile(os.path.join(d, files[1]), dst2)

    print("[Seed] Sample images verified in uploads/ directory.")


def seed_data():
    print("[Seed] Initializing database schema...")
    init_db(drop_existing=False)

    copy_sample_photos()

    # Clear existing demo records so script is cleanly re-runnable
    conn, engine = get_connection()
    try:
        demo_ids = [
            "rep_demo_1", "rep_demo_2", "rep_demo_3", "rep_demo_4", "rep_demo_5"
        ]
        if engine == "mysql":
            with conn.cursor() as cur:
                format_strings = ','.join(['%s'] * len(demo_ids))
                cur.execute(f"DELETE FROM verification_records WHERE report_id IN ({format_strings});", demo_ids)
                cur.execute(f"DELETE FROM task_updates WHERE report_id IN ({format_strings});", demo_ids)
                cur.execute(f"DELETE FROM assignments WHERE report_id IN ({format_strings});", demo_ids)
                cur.execute(f"DELETE FROM status_history WHERE report_id IN ({format_strings});", demo_ids)
                cur.execute(f"DELETE FROM funding_transactions WHERE campaign_id IN (SELECT id FROM funding_campaigns WHERE report_id IN ({format_strings}));", demo_ids)
                cur.execute(f"DELETE FROM funding_campaigns WHERE report_id IN ({format_strings});", demo_ids)
                cur.execute(f"DELETE FROM reports WHERE id IN ({format_strings});", demo_ids)
        else:
            cur = conn.cursor()
            placeholders = ','.join(['?'] * len(demo_ids))
            cur.execute(f"DELETE FROM verification_records WHERE report_id IN ({placeholders});", demo_ids)
            cur.execute(f"DELETE FROM task_updates WHERE report_id IN ({placeholders});", demo_ids)
            cur.execute(f"DELETE FROM assignments WHERE report_id IN ({placeholders});", demo_ids)
            cur.execute(f"DELETE FROM status_history WHERE report_id IN ({placeholders});", demo_ids)
            cur.execute(f"DELETE FROM funding_transactions WHERE campaign_id IN (SELECT id FROM funding_campaigns WHERE report_id IN ({placeholders}));", demo_ids)
            cur.execute(f"DELETE FROM funding_campaigns WHERE report_id IN ({placeholders});", demo_ids)
            cur.execute(f"DELETE FROM reports WHERE id IN ({placeholders});", demo_ids)
            conn.commit()
    finally:
        conn.close()

    print("[Seed] Registering demo users across roles...")
    users = [
        {
            "id": "usr_citizen_1",
            "name": "Priya Malhotra",
            "email": "priya@example.com",
            "role": "citizen",
            "organization": "Resident, Sector 14",
            "phone": "+91 98120 44321",
            "avatar": "PM"
        },
        {
            "id": "usr_contractor_1",
            "name": "Ramesh Contractors",
            "email": "ramesh@example.com",
            "role": "contractor",
            "organization": "Ramesh Civil Infrastructure Ltd.",
            "phone": "+91 98111 22334",
            "avatar": "RC"
        },
        {
            "id": "usr_engineer_1",
            "name": "Vikrant Joshi",
            "email": "vikrant@example.com",
            "role": "engineer",
            "organization": "Certified Municipal Electrical & Drainage Engineer",
            "phone": "+91 98222 33445",
            "avatar": "VJ"
        },
        {
            "id": "usr_ngo_1",
            "name": "Sewa Bharti NGO",
            "email": "sewa@example.com",
            "role": "ngo",
            "organization": "Sewa Bharti Urban Habitat Mission",
            "phone": "+91 98333 44556",
            "avatar": "SB"
        },
        {
            "id": "usr_csr_1",
            "name": "Tata Community CSR",
            "email": "csr@tata.com",
            "role": "company",
            "organization": "Tata Trust Urban Renewal Fund",
            "phone": "+91 98444 55667",
            "avatar": "TC"
        },
        {
            "id": "usr_admin_1",
            "name": "Municipal Commissioner Sharma",
            "email": "admin@sonipat.gov.in",
            "role": "admin",
            "organization": "Municipal Corporation Sonipat",
            "phone": "+91 98555 66778",
            "avatar": "MS"
        }
    ]

    for u in users:
        upsert_user(u)

    print("[Seed] Seeding realistic civic reports across all workflow states...")

    # -------------------------------------------------------------
    # Report 1: Pothole -> Status: IN_PROGRESS (50% progress, 25% verified & paid)
    # -------------------------------------------------------------
    rep1 = {
        "id": "rep_demo_1",
        "user_id": "usr_citizen_1",
        "title": "Deep pothole, GT Road service lane",
        "category": "potholes",
        "location_text": "Sector 14, Sonipat",
        "latitude": 28.9930,
        "longitude": 77.0150,
        "image_path": "/uploads/reports/sample_potholes.jpg",
        "status": "IN_PROGRESS",
        "ai_category": "potholes",
        "ai_confidence": 0.984,
        "severity": "High",
        "severity_score": 8.4,
        "severity_reason": "Severe depth (~12cm) in active high-speed commercial lane with structural rim degradation.",
        "estimated_cost_min": 7500.0,
        "estimated_cost_max": 8500.0,
        "work_scope": "Excavate damaged asphalt edges, lay 50mm compacted aggregate base and apply cold mix bitumen patch.",
        "goal_amount": 8000.0,
        "eta_days": 4
    }
    create_report(rep1)

    # Fund Report 1 to 100%
    add_funding_contribution("fc_rep_demo_1", "usr_citizen_1", "Priya Malhotra (resident)", 500.0, "For safer daily commute")
    add_funding_contribution("fc_rep_demo_1", "usr_local_biz", "Anand Traders (local biz)", 2000.0, "Shopfront road improvement")
    add_funding_contribution("fc_rep_demo_1", "usr_citizen_pool", "21 Neighbourhood Residents", 5500.0, "Community crowd pool")

    # Contributor accepts and starts
    accept_task("rep_demo_1", "usr_contractor_1", "Ramesh Contractors", "contractor")
    start_task("rep_demo_1", "usr_contractor_1")

    # Milestone 1: 25% progress verified & funded
    upd1 = submit_task_progress(
        report_id="rep_demo_1",
        assignment_id="asgn_rep_demo_1",
        progress_pct=25,
        work_note="Subgrade excavated, edges squared off, 40mm crushed stone base vibratory rolled.",
        progress_image="/uploads/progress/progress_potholes.jpg",
        submit_verification=False
    )
    admin_approve_verification(
        update_id=upd1["update_id"],
        report_id="rep_demo_1",
        admin_id="usr_admin_1",
        notes="Excavation depth and base compaction inspected on site. Approved milestone 1.",
        amount_to_release=2000.0
    )

    # Milestone 2: 50% in progress
    submit_task_progress(
        report_id="rep_demo_1",
        assignment_id="asgn_rep_demo_1",
        progress_pct=50,
        work_note="Bituminous emulsion tack coat applied. Ready for dense graded asphalt wearing course.",
        progress_image="/uploads/progress/progress_potholes.jpg",
        submit_verification=False
    )
    update_report_status("rep_demo_1", "IN_PROGRESS", "Ramesh Contractors", "contractor", "Work progressing on asphalt application")

    # -------------------------------------------------------------
    # Report 2: Streetlights -> Status: FUNDING (Active campaign ~38% funded)
    # -------------------------------------------------------------
    rep2 = {
        "id": "rep_demo_2",
        "user_id": "usr_citizen_1",
        "title": "3 dead streetlights, Model Town park",
        "category": "streetlights",
        "location_text": "Model Town, Sonipat",
        "latitude": 28.9880,
        "longitude": 77.0220,
        "image_path": "/uploads/reports/sample_streetlights.jpg",
        "status": "FUNDING",
        "ai_category": "streetlights",
        "ai_confidence": 0.942,
        "severity": "Medium",
        "severity_score": 5.8,
        "severity_reason": "Multiple fixtures unlit creating safety vulnerability on park walking track.",
        "estimated_cost_min": 9500.0,
        "estimated_cost_max": 12000.0,
        "work_scope": "Replace 3 fused 60W LED luminaire fixtures, test driver capacitors, and secure underground armoring cable.",
        "goal_amount": 10800.0,
        "eta_days": 9
    }
    create_report(rep2)
    add_funding_contribution("fc_rep_demo_2", "usr_rwa", "Model Town RWA", 2500.0, "RWA infrastructure allocation")
    add_funding_contribution("fc_rep_demo_2", "usr_family", "Verma Family", 800.0, "Park safety backing")
    add_funding_contribution("fc_rep_demo_2", "usr_citizen_pool2", "9 Other Park Regulars", 800.0, "Morning walkers contribution")

    # -------------------------------------------------------------
    # Report 3: Drainage -> Status: SUBMITTED_FOR_VERIFICATION (100% completed, awaiting admin verification)
    # -------------------------------------------------------------
    rep3 = {
        "id": "rep_demo_3",
        "user_id": "usr_citizen_1",
        "title": "Blocked drain, overflow risk in monsoon",
        "category": "waterlogging",
        "location_text": "Rajiv Colony, Sonipat",
        "latitude": 29.0020,
        "longitude": 77.0310,
        "image_path": "/uploads/reports/sample_waterlogging.jpg",
        "status": "SUBMITTED_FOR_VERIFICATION",
        "ai_category": "waterlogging",
        "ai_confidence": 0.917,
        "severity": "High",
        "severity_score": 7.9,
        "severity_reason": "High risk of street flooding and foundation seepage due to silt clogged main canal.",
        "estimated_cost_min": 11000.0,
        "estimated_cost_max": 14000.0,
        "work_scope": "Clear 120m underground stormwater culvert, remove silt debris, and install heavy-duty reinforced iron grating.",
        "goal_amount": 12500.0,
        "eta_days": 14
    }
    create_report(rep3)
    add_funding_contribution("fc_rep_demo_3", "usr_ngo_1", "Sewa Bharti Urban Foundation", 5000.0, "Monsoon flood mitigation grant")
    add_funding_contribution("fc_rep_demo_3", "usr_rwa_rajiv", "Rajiv Colony Emergency Fund", 4500.0, "Emergency drain relief")
    add_funding_contribution("fc_rep_demo_3", "usr_citizen_pool3", "14 Colony Residents", 3000.0, "Community backing")

    accept_task("rep_demo_3", "usr_engineer_1", "Vikrant Joshi", "engineer")
    start_task("rep_demo_3", "usr_engineer_1")
    submit_task_progress(
        report_id="rep_demo_3",
        assignment_id="asgn_rep_demo_3",
        progress_pct=100,
        work_note="120m culvert cleaned using vacuum jetting. 2.4 tonnes of compacted silt removed. New rust-proof MS grating installed.",
        progress_image="/uploads/progress/progress_waterlogging.jpg",
        submit_verification=True
    )

    # -------------------------------------------------------------
    # Report 4: Green Cover / Plantation -> Status: ASSIGNED (100% funded, accepted by NGO)
    # -------------------------------------------------------------
    rep4 = {
        "id": "rep_demo_4",
        "user_id": "usr_citizen_1",
        "title": "Plantation drive, dry canal stretch",
        "category": "potholes",  # Category map
        "location_text": "Kundli Border Road, Sonipat",
        "latitude": 28.9750,
        "longitude": 77.0450,
        "image_path": "/uploads/reports/sample_potholes.jpg",
        "status": "ASSIGNED",
        "ai_category": "potholes",
        "ai_confidence": 0.890,
        "severity": "Low",
        "severity_score": 3.2,
        "severity_reason": "Erosion prone vacant stretch requiring ground cover and perimeter reinforcement.",
        "estimated_cost_min": 13500.0,
        "estimated_cost_max": 16500.0,
        "work_scope": "Plant 50 native shade trees with drip irrigation and tree guards along 300m canal berm.",
        "goal_amount": 15000.0,
        "eta_days": 1
    }
    create_report(rep4)
    add_funding_contribution("fc_rep_demo_4", "usr_ngo_1", "Sewa Bharti NGO", 6000.0, "Green Delhi-NCR initiative")
    add_funding_contribution("fc_rep_demo_4", "usr_green_sonipat", "GreenSonipat Collective", 4200.0, "Tree sapling sponsorship")
    add_funding_contribution("fc_rep_demo_4", "usr_citizen_pool4", "45 Civic Volunteers", 4800.0, "Individual citizen pledges")
    accept_task("rep_demo_4", "usr_ngo_1", "Sewa Bharti NGO", "ngo")

    # -------------------------------------------------------------
    # Report 5: Garbage dump -> Status: COMPLETED (Verified & 100% funds released)
    # -------------------------------------------------------------
    rep5 = {
        "id": "rep_demo_5",
        "user_id": "usr_citizen_1",
        "title": "Illegal dumping heap near Station Road",
        "category": "garbage",
        "location_text": "Station Road, Sonipat",
        "latitude": 29.0100,
        "longitude": 77.0120,
        "image_path": "/uploads/reports/sample_garbage.jpg",
        "status": "COMPLETED",
        "ai_category": "garbage",
        "ai_confidence": 0.961,
        "severity": "Medium",
        "severity_score": 6.2,
        "severity_reason": "Overflowing waste obstructing pedestrian sidewalk with sanitation risk.",
        "estimated_cost_min": 5500.0,
        "estimated_cost_max": 6500.0,
        "work_scope": "Clear 4 tonnes solid waste, sanitize ground with lime powder, and position twin 1100L segregated bins.",
        "goal_amount": 6000.0,
        "eta_days": 1
    }
    create_report(rep5)
    add_funding_contribution("fc_rep_demo_5", "usr_citizen_1", "Priya Malhotra", 1000.0, "Sanitation backing")
    add_funding_contribution("fc_rep_demo_5", "usr_station_market", "Station Market Association", 3500.0, "Clean market drive")
    add_funding_contribution("fc_rep_demo_5", "usr_citizen_pool5", "12 Commuters", 1500.0, "Commuter pledges")

    accept_task("rep_demo_5", "usr_contractor_1", "Ramesh Contractors", "contractor")
    start_task("rep_demo_5", "usr_contractor_1")
    upd5 = submit_task_progress(
        report_id="rep_demo_5",
        assignment_id="asgn_rep_demo_5",
        progress_pct=100,
        work_note="Waste cleared, ground disinfected with lime, 2 heavy-duty HDPE municipal dustbins placed.",
        progress_image="/uploads/progress/progress_garbage.jpg",
        submit_verification=True
    )
    admin_approve_verification(
        update_id=upd5["update_id"],
        report_id="rep_demo_5",
        admin_id="usr_admin_1",
        notes="Site inspected by Ward 7 sanitary inspector. Complete restoration verified.",
        amount_to_release=6000.0
    )

    print("[Seed] Seeding rewards leaderboard and user points...")
    upsert_rewards("usr_citizen_1", points=2140, contributed_amount=1850.0, problems_backed=6, user_name="Priya Malhotra")
    upsert_rewards("usr_contractor_1", points=5600, contributed_amount=0.0, problems_backed=12, user_name="Ramesh Contractors")
    upsert_rewards("usr_ngo_1", points=4920, contributed_amount=11000.0, problems_backed=9, user_name="Sewa Bharti NGO")
    upsert_rewards("usr_local_biz", points=1880, contributed_amount=2000.0, problems_backed=5, user_name="Anand Traders")
    upsert_rewards("usr_engineer_1", points=1250, contributed_amount=0.0, problems_backed=3, user_name="Vikrant Joshi")

    print("[Seed] Complete demo dataset successfully seeded!")


if __name__ == "__main__":
    seed_data()
