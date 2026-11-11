import os
import sys
import io
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from predict import get_predictor
from severity_predictor import get_severity_predictor
try:
    from cost.cost_estimator import estimate_repair_cost
except ImportError:
    from cost_estimator import estimate_repair_cost
from matching_engine import rank_contributors
from database import (
    get_contributor_count, get_all_reports, get_report_by_id, create_report,
    update_report_status, get_funding_campaign, add_funding_contribution,
    accept_task, start_task, submit_task_progress, get_admin_verifications,
    admin_approve_verification, admin_rework_verification, admin_release_funds,
    get_admin_dashboard_stats, get_user_rewards, get_leaderboard,
    get_all_users, get_user_by_id, init_db, get_global_status_history,
    delete_report, get_contributor_payments, compute_dynamic_rewards
)
from geocoding_service import resolve_location

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "nirmaan-ui")

app = Flask(__name__, static_folder=UI_DIR, static_url_path="")
CORS(app)

print("[Server] Loading Nirmaan Recognition Model (ResNet-50)...")
recognition_predictor = get_predictor()
print(f"[Server] Recognition Model ready. Device: {recognition_predictor.device}")

print("[Server] Loading Nirmaan Severity Model (ResNet-18)...")
severity_predictor = get_severity_predictor()
print(f"[Server] Severity Model ready. Device: {severity_predictor.device}")

# Initialise database at startup (triggers MySQL→SQLite fallback if needed)
print("[Server] Initialising database schema...")
init_db()
print("[Server] Database ready.")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def serve_ui():
    """Serves the main Nirmaan frontend index.html."""
    return send_from_directory(UI_DIR, "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "nirmaan-ai-pipeline",
        "recognition_model": {
            "architecture": "ResNet-50",
            "device": str(recognition_predictor.device),
            "classes": recognition_predictor.classes,
            "checkpoint": os.path.basename(recognition_predictor.model_path)
        },
        "severity_model": {
            "architecture": "ResNet-18",
            "device": str(severity_predictor.device),
            "classes": ["Low", "Medium", "High"],
            "checkpoint": os.path.basename(severity_predictor.model_path)
        },
        "cost_engine": {
            "version": "v1.0",
            "method": "Rule-based / data-driven repair cost estimation engine using AI-derived category and severity outputs",
            "supported_categories": ["potholes", "streetlights", "waterlogging", "garbage"]
        },
        "contributor_matcher": {
            "version": "v1.0",
            "method": "Transparent deterministic weighted feature scoring with geographic radius filtering",
            "active_contributors_count": get_contributor_count()
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts uploaded image file.
    Runs 3-stage NIRMAAN pipeline:
      1. ResNet-50 Recognition: Category + Confidence
      2. ResNet-18 Severity: Low / Medium / High + Calibrated Score (0-10)
      3. Cost Estimation Engine: Visual damage extraction + Category/Severity calibrated range
    """
    if "image" not in request.files and "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided in request. Use 'image' or 'file' key in FormData."
        }), 400

    file = request.files.get("image") or request.files.get("file")

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename provided."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file type. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        # Read raw image bytes once into memory
        image_bytes = file.stream.read()

        # Step 1: Run ResNet-50 Recognition Model
        recognition = recognition_predictor.predict_image(io.BytesIO(image_bytes))
        category = recognition.get("category", "potholes")

        # Step 2: Run ResNet-18 Severity Model
        severity_result = severity_predictor.predict_image(io.BytesIO(image_bytes), category=category)

        # Step 3: Run Cost Estimation Engine
        # Optional location parameter from request form/args (default neutral 1.0)
        location_data = (
            request.form.get("location_factor")
            or request.form.get("location")
            or request.args.get("location_factor")
            or 1.0
        )
        cost_result = estimate_repair_cost(
            image_input=io.BytesIO(image_bytes),
            category=category,
            severity=severity_result["severity"],
            severity_score=severity_result["severity_score"],
            location_data=location_data
        )

        # Build response adhering strictly to requirements
        response = {
            "success": True,
            # Recognition fields
            "category": recognition.get("category"),
            "display_name": recognition.get("display_name"),
            "class_id": recognition.get("class_id"),
            "confidence": recognition.get("confidence"),
            "confidence_percentage": recognition.get("confidence_percentage"),
            "device": recognition.get("device"),
            # Severity model fields
            "severity": severity_result["severity"],
            "severity_score": severity_result["severity_score"],
            "severity_display": severity_result["severity_display"],
            "severity_class": severity_result["severity_class"],
            "severity_reason": severity_result["severity_reason"],
            "probabilities": severity_result["probabilities"],
            "severity_model": severity_result["severity_model"],
            # Cost Estimation Engine fields (API contract)
            "estimated_cost": cost_result["estimated_cost"],
            "estimated_repair_cost": cost_result["estimated_cost"]["label"],
            "work_scope": cost_result["work_scope"],
            "cost_note": cost_result["work_scope"],
            "damage_factor": cost_result["damage_factor"],
            "extent_factor": cost_result["extent_factor"],
            "location_factor": cost_result["location_factor"],
            "cost_estimation_method": cost_result["cost_estimation_method"],
            "damage_features": cost_result["damage_features"],
            # Metadata block for frontend compatibility
            "metadata": {
                "severity": severity_result["severity_display"],
                "severityClass": severity_result["severity_class"],
                "reason": severity_result["severity_reason"],
                "cost": cost_result["estimated_cost"]["label"],
                "estimated_cost": cost_result["estimated_cost"],
                "matches": recognition.get("metadata", {}).get("matches", [])
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"[Error] Inference error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Failed to process image: {str(e)}"
        }), 500

@app.route("/api/match", methods=["POST"])
def match_contributors():
    """
    Ranks the top suitable registered contributors for a civic defect.
    Accepts:
      category (str): Problem category (e.g. 'potholes', 'streetlights', 'waterlogging', 'garbage')
      severity (str): 'Low' / 'Medium' / 'High'
      severity_score (float): 0 - 10
      estimated_cost_min (float): Minimum estimated repair cost in INR
      estimated_cost_max (float): Maximum estimated repair cost in INR
      work_scope (str): Suggested civic repair work scope
      location_text (str): Locality or city (e.g. 'Sector 21, Sonipat')
      latitude (float, optional): Exact latitude
      longitude (float, optional): Exact longitude
    Returns:
      Top 3 ranked contributors with transparent, deterministic ranking reasons.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({
            "success": False,
            "error": "Missing JSON request body or form payload."
        }), 400

    try:
        top_n = int(request.args.get("top_n", 3))
        result = rank_contributors(data, top_n=top_n)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Contributor matching failed: {str(e)}"
        }), 500


# =========================================================================
# FILE UPLOADS SERVING
# =========================================================================

@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    uploads_dir = os.path.join(UI_DIR, "uploads")
    return send_from_directory(uploads_dir, filename)


# =========================================================================
# AUTHENTICATION & ROLE-BASED ACCESS (DEMO READY)
# =========================================================================

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    role = data.get("role", "citizen").lower()
    user_id = data.get("user_id")

    user = None
    if user_id:
        user = get_user_by_id(user_id)

    if not user:
        demo_map = {
            "citizen": "usr_citizen_1",
            "contractor": "usr_contractor_1",
            "engineer": "usr_engineer_1",
            "ngo": "usr_ngo_1",
            "company": "usr_csr_1",
            "admin": "usr_admin_1"
        }
        uid = demo_map.get(role, "usr_citizen_1")
        user = get_user_by_id(uid)

    return jsonify({"success": True, "user": user})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user_id = request.args.get("user_id") or "usr_citizen_1"
    user = get_user_by_id(user_id)
    return jsonify({"success": True, "user": user})


@app.route("/api/users", methods=["GET"])
def list_users():
    users = get_all_users()
    return jsonify({"success": True, "users": users})


# =========================================================================
# REPORTS & DEFECT PIPELINE ENDPOINTS
# =========================================================================

@app.route("/api/reports", methods=["GET"])
def api_get_reports():
    cat = request.args.get("category")
    status = request.args.get("status")
    uid = request.args.get("user_id")
    reports = get_all_reports(category=cat, status=status, user_id=uid)
    return jsonify({"success": True, "reports": reports, "count": len(reports)})


@app.route("/api/reports/<report_id>", methods=["GET"])
def api_get_report_detail(report_id):
    rep = get_report_by_id(report_id)
    if not rep:
        return jsonify({"success": False, "error": f"Report not found: {report_id}"}), 404
    return jsonify({"success": True, "report": rep})


@app.route("/api/admin/reports/<report_id>", methods=["DELETE"])
@app.route("/api/reports/<report_id>", methods=["DELETE"])
def api_delete_report(report_id):
    """
    Permanently deletes a report and all its linked civic records
    (funding campaigns, transactions, assignments, updates, verifications).
    """
    try:
        success = delete_report(report_id)
        if not success:
            return jsonify({"success": False, "error": f"Report not found: {report_id}"}), 404
        return jsonify({
            "success": True,
            "message": f"Report {report_id} and all associated records deleted successfully."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reports", methods=["POST"])
def api_create_report():
    try:
        req_json = request.get_json(silent=True) or {}
        def get_field(k, default=None):
            if k in request.form:
                return request.form.get(k)
            return req_json.get(k, default)

        title = get_field("title")
        category = get_field("category", "potholes")
        loc_text = (get_field("location_text") or "").strip()
        if not loc_text:
            loc_text = "Reported Defect Location"
        user_id = get_field("user_id", "usr_citizen_1")

        try:
            eta_days = int(get_field("eta_days", 7))
        except (ValueError, TypeError):
            eta_days = 7

        goal = get_field("goal_amount")
        goal_amount = None
        if goal is not None and str(goal).strip() != "":
            try:
                goal_amount = float(goal)
            except (ValueError, TypeError):
                goal_amount = None

        lat_in = get_field("latitude")
        lng_in = get_field("longitude")

        # Resolve real coordinates across India via geocoding service
        loc_res = resolve_location(
            location_text=loc_text,
            latitude=lat_in,
            longitude=lng_in,
            enable_online=True
        )
        final_lat = loc_res.get("latitude")
        final_lng = loc_res.get("longitude")

        ai_cat = get_field("ai_category", category)
        try:
            ai_conf = float(get_field("ai_confidence", 0.95))
        except (ValueError, TypeError):
            ai_conf = 0.95

        severity = get_field("severity", "Medium")
        try:
            sev_score = float(get_field("severity_score", 5.0))
        except (ValueError, TypeError):
            sev_score = 5.0

        sev_reason = get_field("severity_reason", "AI detected civic defect.")
        try:
            cost_min = float(get_field("estimated_cost_min", 5000.0))
        except (ValueError, TypeError):
            cost_min = 5000.0
        try:
            cost_max = float(get_field("estimated_cost_max", 10000.0))
        except (ValueError, TypeError):
            cost_max = 10000.0

        work_scope = get_field("work_scope", "")

        # Save original image separately from progress proof
        img_url = f"/uploads/reports/sample_{category}.jpg"
        file = request.files.get("image") or request.files.get("file")
        if file and file.filename != "" and allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            fname = f"rep_{uuid.uuid4().hex[:12]}.{ext}"
            rep_dir = os.path.join(UI_DIR, "uploads", "reports")
            os.makedirs(rep_dir, exist_ok=True)
            file.save(os.path.join(rep_dir, fname))
            img_url = f"/uploads/reports/{fname}"
        elif get_field("image_path"):
            img_url = get_field("image_path")

        rep_data = {
            "user_id": user_id,
            "title": title or f"{category.capitalize()} repair, {loc_text}",
            "category": category,
            "location_text": loc_text,
            "latitude": final_lat,
            "longitude": final_lng,
            "image_path": img_url,
            "status": "FUNDING",
            "ai_category": ai_cat,
            "ai_confidence": ai_conf,
            "severity": severity,
            "severity_score": sev_score,
            "severity_reason": sev_reason,
            "estimated_cost_min": cost_min,
            "estimated_cost_max": cost_max,
            "work_scope": work_scope,
            "eta_days": eta_days,
            "goal_amount": goal_amount
        }
        new_id = create_report(rep_data)
        rep = get_report_by_id(new_id)
        return jsonify({"success": True, "report_id": new_id, "report": rep}), 201
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to create report: {str(e)}"}), 500


# =========================================================================
# FUNDING CAMPAIGN & TRANSACTION ENDPOINTS
# =========================================================================

@app.route("/api/funding/<report_id>", methods=["GET"])
def api_get_funding(report_id):
    camp = get_funding_campaign(report_id)
    if not camp:
        return jsonify({"success": False, "error": "Funding campaign not found"}), 404
    return jsonify({"success": True, "campaign": camp})


@app.route("/api/funding/<report_id>/contribute", methods=["POST"])
def api_contribute_funding(report_id):
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    amount = float(data.get("amount", 500.0))
    donor = data.get("donor_name", "Priya Malhotra (resident)")
    uid = data.get("user_id", "usr_citizen_1")
    note = data.get("note", "Demo community contribution")

    try:
        res = add_funding_contribution(report_id, uid, donor, amount, note)
        return jsonify({"success": True, "contribution": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =========================================================================
# CONTRIBUTOR PORTAL & TASK MANAGEMENT
# =========================================================================

@app.route("/api/contributor/tasks", methods=["GET"])
def api_contributor_tasks():
    role = request.args.get("role", "contractor").lower()
    reports = get_all_reports()

    tasks = []
    for r in reports:
        cat = r.get("category", "")
        # Recommendation score based on contributor fit
        base_match = 75
        if role in ("contractor", "engineer"):
            if cat == "potholes":
                base_match = 96 if role == "contractor" else 84
            elif cat == "streetlights":
                base_match = 92 if role == "engineer" else 78
            elif cat == "waterlogging":
                base_match = 89
            else:
                base_match = 72
        elif role == "ngo":
            base_match = 91 if cat in ("garbage", "potholes") else 74
        elif role == "company":
            base_match = 94 if cat in ("potholes", "streetlights") else 85

        tasks.append({
            **r,
            "match_score": base_match,
            "distance_km": round(1.2 + (len(r.get("title", "")) % 6) * 0.8, 1),
            "budget": r.get("goal_amount") or 8000.0
        })

    return jsonify({"success": True, "tasks": tasks, "count": len(tasks)})


@app.route("/api/contributor/payments", methods=["GET"])
def api_contributor_payments():
    """
    Returns actual fund disbursements and credits released to contributors.
    """
    role = request.args.get("role")
    cid = request.args.get("contributor_id")
    result = get_contributor_payments(role=role, contributor_id=cid)
    return jsonify({"success": True, **result})


@app.route("/api/tasks/<report_id>/accept", methods=["POST"])
def api_task_accept(report_id):
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        cid = data.get("contributor_id", "usr_contractor_1")
        name = data.get("contributor_name", "Ramesh Contractors")
        role = data.get("contributor_role", "contractor")

        res = accept_task(report_id, cid, name, role)
        return jsonify({"success": True, "assignment": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/tasks/<report_id>/start", methods=["POST"])
def api_task_start(report_id):
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        cid = data.get("contributor_id", "usr_contractor_1")
        res = start_task(report_id, cid)
        return jsonify({"success": True, "result": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/tasks/<report_id>/progress", methods=["POST"])
def api_task_progress(report_id):
    try:
        asgn_id = request.form.get("assignment_id")
        pct = int(request.form.get("progress_pct", 25))
        note = request.form.get("work_note", f"Work completed up to {pct}% milestone.")
        submit_v = request.form.get("submit_verification", "true").lower() in ("true", "1", "yes")

        # Progress image upload
        img_url = "/uploads/progress/progress_potholes.jpg"
        file = request.files.get("image") or request.files.get("file")
        if file and file.filename != "" and allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            fname = f"prog_{uuid.uuid4().hex[:12]}.{ext}"
            prog_dir = os.path.join(UI_DIR, "uploads", "progress")
            os.makedirs(prog_dir, exist_ok=True)
            file.save(os.path.join(prog_dir, fname))
            img_url = f"/uploads/progress/{fname}"
        elif request.form.get("progress_image"):
            img_url = request.form.get("progress_image")

        res = submit_task_progress(
            report_id=report_id,
            assignment_id=asgn_id,
            progress_pct=pct,
            work_note=note,
            progress_image=img_url,
            submit_verification=submit_v
        )
        return jsonify({"success": True, "update": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/tasks/<report_id>/submit-verification", methods=["POST"])
def api_task_submit_verification(report_id):
    try:
        update_report_status(report_id, "SUBMITTED_FOR_VERIFICATION", "Contributor", "contractor", "Submitted for municipal verification")
        return jsonify({"success": True, "status": "SUBMITTED_FOR_VERIFICATION"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =========================================================================
# ADMIN VERIFICATION & CENTRAL CONTROL
# =========================================================================

@app.route("/api/admin/dashboard", methods=["GET"])
def api_admin_dashboard():
    try:
        stats = get_admin_dashboard_stats()
        return jsonify({"success": True, "stats": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/reports", methods=["GET"])
def api_admin_reports():
    try:
        reports = get_all_reports()
        return jsonify({"success": True, "reports": reports})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/verifications", methods=["GET"])
def api_admin_verifications():
    try:
        status_filter = request.args.get("status", "pending")
        verifs = get_admin_verifications(status_filter=status_filter)
        return jsonify({"success": True, "verifications": verifs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/admin/verifications/<update_id>/approve", methods=["POST"])
def api_admin_approve_verification(update_id):
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        report_id = data.get("report_id")
        admin_id = data.get("admin_id", "usr_admin_1")
        notes = data.get("notes", "On-site photographic proof verified by Municipal Engineer.")
        release_amount = float(data.get("amount_to_release", 0.0))

        res = admin_approve_verification(update_id, report_id, admin_id, notes, release_amount)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/admin/verifications/<update_id>/rework", methods=["POST"])
def api_admin_rework_verification(update_id):
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        report_id = data.get("report_id")
        admin_id = data.get("admin_id", "usr_admin_1")
        notes = data.get("notes", "Work quality does not comply with engineering standards. Rework required.")

        res = admin_rework_verification(update_id, report_id, admin_id, notes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/admin/funds/release", methods=["POST"])
def api_admin_release_funds():
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        report_id = data.get("report_id")
        amount = float(data.get("amount", 0.0))
        admin_id = data.get("admin_id", "usr_admin_1")
        notes = data.get("notes", "Administrative milestone payout")

        res = admin_release_funds(report_id, amount, admin_id, notes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# =========================================================================
# ADMIN AUDIT HISTORY
# =========================================================================

@app.route("/api/admin/history", methods=["GET"])
def api_admin_history():
    """Returns recent status transitions from the audit log."""
    try:
        limit = int(request.args.get("limit", 30))
        rows = get_global_status_history(limit=limit)
        return jsonify({"success": True, "history": rows})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================================================
# REWARDS & CIVIC LEADERBOARD
# =========================================================================

@app.route("/api/rewards", methods=["GET"])
def api_rewards():
    try:
        uid = request.args.get("user_id") or "usr_citizen_1"
        # Single DB pass: compute both user rewards and top-10 leaderboard together
        user_rewards, leaderboard = compute_dynamic_rewards(target_user_id=uid, limit=10)
        if not user_rewards:
            user_rewards = {
                "user_id": uid,
                "user_name": "Active Citizen",
                "role": "citizen",
                "points": 0,
                "contributed_amount": 0.0,
                "problems_backed": 0,
                "completed_fixes": 0,
                "reports_filed": 0,
                "tier": "Bronze",
                "rank": 999
            }
        return jsonify({
            "success": True,
            "rewards": user_rewards,
            "leaderboard": leaderboard
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



# =========================================================================
# GLOBAL API ERROR HANDLERS (JSON ONLY FOR /api/ ROUTES)
# =========================================================================

@app.errorhandler(400)
def api_bad_request(e):
    if request.path.startswith("/api/") or request.path in ("/predict", "/health"):
        return jsonify({"success": False, "error": "Bad request"}), 400
    return e


@app.errorhandler(404)
def api_not_found(e):
    if request.path.startswith("/api/") or request.path in ("/predict", "/health"):
        return jsonify({"success": False, "error": f"Endpoint not found: {request.path}"}), 404
    return send_from_directory(UI_DIR, "index.html")


@app.errorhandler(405)
def api_method_not_allowed(e):
    if request.path.startswith("/api/") or request.path in ("/predict", "/health"):
        return jsonify({"success": False, "error": f"Method {request.method} not allowed for {request.path}"}), 405
    return e


@app.errorhandler(500)
def api_server_error(e):
    if request.path.startswith("/api/") or request.path in ("/predict", "/health"):
        return jsonify({"success": False, "error": "Internal server error"}), 500
    return e


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Initialise DB schema (creates tables if missing, resolves MySQL→SQLite fallback)
    print("[Server] Initialising database schema...")
    init_db()
    print("[Server] Database ready.")

    print(f"\n========================================================")
    print(f"  NIRMAAN TRIPLE-STAGE PIPELINE SERVER RUNNING")
    print(f"  Local URL: http://127.0.0.1:{port}")
    print(f"  Predict Endpoint: http://127.0.0.1:{port}/predict")
    print(f"========================================================\n")
    sys.stdout.flush()
    app.run(host="0.0.0.0", port=port, debug=False)
