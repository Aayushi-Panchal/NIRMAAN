"""
NIRMAAN Contributor Matching and Ranking Engine (V1).

A transparent, deterministic, feature-based recommendation system that scores
and ranks registered contributors against AI-detected civic infrastructure problems.

Matches problems based on:
1. Category & Skill Compatibility (30%)
2. Distance & Service Radius (20%)
3. Budget Compatibility (15%)
4. Severity Capability (10%)
5. Similar Past Work Track Record (10%)
6. Contributor Reliability & Verification (10%)
7. Operational Availability (5%)

Produces top 3 ranked contributors with transparent, deterministic reasons.
"""
import math
try:
    from db_config import (
        MATCH_WEIGHTS, SEVERITY_LEVELS, AVAILABILITY_SCORES, VERIFICATION_FACTORS
    )
except ImportError:
    from database.db_config import (
        MATCH_WEIGHTS, SEVERITY_LEVELS, AVAILABILITY_SCORES, VERIFICATION_FACTORS
    )
from geocoding_service import resolve_location
from database import get_all_active_contributors

# Category to relevant skills mapping for civic repairs
CATEGORY_SKILL_AFFINITY = {
    "potholes": {"road_repair", "civil_construction", "civil_inspection", "infrastructure_funding"},
    "streetlights": {"electrical", "street_safety", "infrastructure_funding", "civil_inspection"},
    "waterlogging": {"drainage", "civil_construction", "civil_inspection", "infrastructure_funding"},
    "garbage": {"waste_management", "community_cleanup", "community_programs", "volunteer_coordination"}
}


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculates great-circle distance between two coordinates in kilometers
    using the Haversine formula.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    # Earth radius in kilometers
    R = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


def calculate_category_skill_score(problem_category, problem_work_scope, contributor):
    """
    Evaluates category match, skill affinities, and preferred work scope.
    Returns score normalized 0.0 - 100.0.
    """
    cat = (problem_category or "").strip().lower()
    contributor_cats = [c.strip().lower() for c in contributor.get("categories", [])]
    contributor_skills = set(s.strip().lower() for s in contributor.get("skills", []))

    score = 0.0

    # 1. Direct Category Match (50 points)
    if cat in contributor_cats:
        score += 50.0

    # 2. Skill Affinity Match (up to 35 points)
    expected_skills = CATEGORY_SKILL_AFFINITY.get(cat, set())
    matching_skills = contributor_skills.intersection(expected_skills)
    if matching_skills:
        skill_ratio = min(1.0, len(matching_skills) / max(1, len(expected_skills) / 2))
        score += skill_ratio * 35.0
    elif len(contributor_skills) > 0:
        # Generic relevant engineering/cleanup capability
        score += 10.0

    # 3. Work scope overlap (up to 15 points)
    pref_scope = (contributor.get("preferred_work_scope") or "").lower().replace("_", " ")
    prob_scope = (problem_work_scope or "").lower().replace("_", " ")
    if pref_scope and prob_scope:
        pref_tokens = set(pref_scope.split())
        prob_tokens = set(prob_scope.split())
        common = pref_tokens.intersection(prob_tokens)
        if common:
            score += min(15.0, len(common) * 5.0)

    return min(100.0, max(0.0, score))


def calculate_distance_score(distance_km, service_radius_km):
    """
    Evaluates geographic proximity relative to contributor's service radius.
    Returns score normalized 0.0 - 100.0.
    """
    if distance_km is None:
        # Neutral score if location unresolved
        return 50.0

    radius = max(1.0, float(service_radius_km or 10.0))

    if distance_km <= radius:
        # Linearly scales from 100.0 (at 0 km) down to 55.0 (at service boundary)
        ratio = distance_km / radius
        score = 100.0 - (45.0 * ratio)
    else:
        # Graceful decay beyond service radius
        excess = distance_km - radius
        decay_factor = math.exp(-excess / (radius * 0.5))
        score = 50.0 * decay_factor

    return round(min(100.0, max(0.0, score)), 2)


def calculate_budget_score(problem_min, problem_max, contrib_min, contrib_max):
    """
    Evaluates overlap between AI estimated repair cost and contributor budget range.
    Returns score normalized 0.0 - 100.0.
    """
    try:
        p_min = float(problem_min)
        p_max = float(problem_max)
    except (ValueError, TypeError):
        return 50.0  # Neutral if problem cost unavailable

    c_min = float(contrib_min or 0.0)
    c_max = float(contrib_max or 0.0)

    if c_max <= 0:
        c_max = 500000.0

    # Calculate overlap interval
    overlap_start = max(p_min, c_min)
    overlap_end = min(p_max, c_max)
    overlap = max(0.0, overlap_end - overlap_start)

    problem_span = max(100.0, p_max - p_min)

    if overlap > 0.0:
        ratio = min(1.0, overlap / problem_span)
        # 75 to 100 score for valid overlap
        score = 75.0 + (25.0 * ratio)
    else:
        # Distance between non-overlapping intervals
        gap = min(abs(p_min - c_max), abs(c_min - p_max))
        relative_gap = gap / max(1000.0, p_max)
        score = max(5.0, 60.0 - (relative_gap * 40.0))

    return round(min(100.0, max(0.0, score)), 2)


def calculate_severity_score(problem_severity, problem_score, contrib_max_severity):
    """
    Evaluates capability to handle reported defect severity.
    Returns score normalized 0.0 - 100.0.
    """
    prob_sev = (problem_severity or "medium").strip().lower()
    cont_sev = (contrib_max_severity or "medium").strip().lower()

    p_level = SEVERITY_LEVELS.get(prob_sev, 2)
    c_level = SEVERITY_LEVELS.get(cont_sev, 2)

    if c_level == p_level:
        score = 100.0
    elif c_level > p_level:
        # Capable of higher severity, slightly reduced overqualification discount
        score = 92.0
    else:
        # Contributor max severity is lower than required
        if p_level - c_level == 1:
            score = 35.0
        else:
            score = 10.0

    return score


def calculate_past_work_score(problem_category, contributor_past_work, experience_years, completed_jobs):
    """
    Scores contributor's direct category experience, completed jobs, and years in field.
    Returns score normalized 0.0 - 100.0.
    """
    cat = (problem_category or "").strip().lower()
    category_jobs = int(contributor_past_work.get(cat, 0))

    # 1. Category specific track record (up to 55 points)
    # 25+ category-specific projects achieves full category points
    category_component = min(55.0, (category_jobs / 25.0) * 55.0)

    # 2. Industry experience in years (up to 25 points)
    # 10+ years achieves full tenure points
    years = max(0, int(experience_years or 0))
    experience_component = min(25.0, (years / 10.0) * 25.0)

    # 3. Overall completed projects volume (up to 20 points)
    # 100+ total jobs achieves full volume points
    total_jobs = max(0, int(completed_jobs or 0))
    volume_component = min(20.0, (total_jobs / 100.0) * 20.0)

    total = category_component + experience_component + volume_component
    return round(min(100.0, max(0.0, total)), 2)


def calculate_reliability_score(completion_rate, completed_jobs, verification_status):
    """
    Scores contributor reliability based on verified completion rate and status.
    Returns score normalized 0.0 - 100.0.
    """
    rate = float(completion_rate or 0.0)
    # Completion rate contributes up to 60 points
    rate_component = min(60.0, rate * 60.0)

    # Verification status contributes up to 25 points
    status = (verification_status or "unverified").strip().lower()
    status_factor = VERIFICATION_FACTORS.get(status, 0.70)
    verification_component = 25.0 * status_factor

    # Job count confidence factor (up to 15 points)
    jobs = max(0, int(completed_jobs or 0))
    confidence_component = min(15.0, (jobs / 50.0) * 15.0)

    total = rate_component + verification_component + confidence_component
    return round(min(100.0, max(0.0, total)), 2)


def calculate_availability_score(availability):
    """
    Scores contributor operational availability.
    Returns score normalized 0.0 - 100.0.
    """
    avail = (availability or "available").strip().lower()
    return AVAILABILITY_SCORES.get(avail, 50.0)


def generate_transparent_reasons(contributor, problem, scores, distance_km):
    """
    Synthesizes clear, verifiable, human-readable explanations detailing
    why the contributor was ranked in the top results.
    """
    reasons = []
    cat = (problem.get("category") or "").strip().lower()

    # Category & Skill reason
    matched_skills = set(contributor.get("skills", [])).intersection(
        CATEGORY_SKILL_AFFINITY.get(cat, set())
    )
    if matched_skills:
        skill_str = ", ".join(sorted(list(matched_skills)))
        reasons.append(f"Possesses specialized capabilities: {skill_str}")
    elif cat in [c.lower() for c in contributor.get("categories", [])]:
        reasons.append(f"Formally registered and verified for {cat} civic repairs")

    # Distance reason
    radius = contributor.get("service_radius_km", 10.0)
    if distance_km is not None:
        if distance_km <= radius:
            reasons.append(f"Located {distance_km} km away, comfortably within service radius ({radius:.0f} km)")
        else:
            reasons.append(f"Located {distance_km} km away in nearby municipal zone")
    else:
        reasons.append(f"Serves municipal zone with an operational radius of {radius:.0f} km")

    # Budget reason
    p_min = problem.get("estimated_cost_min", 0)
    p_max = problem.get("estimated_cost_max", 0)
    c_min = contributor.get("budget_min", 0)
    c_max = contributor.get("budget_max", 0)
    if scores["budget"] >= 75.0:
        reasons.append(f"Budget tier (₹{int(c_min):,} - ₹{int(c_max):,}) aligns with estimated repair cost (₹{int(p_min):,} - ₹{int(p_max):,})")

    # Past work reason
    cat_jobs = contributor.get("past_work", {}).get(cat, 0)
    if cat_jobs > 0:
        reasons.append(f"Proven track record with {cat_jobs} completed {cat} civic assignments")
    elif contributor.get("experience_years", 0) > 0:
        reasons.append(f"Extensive industry experience ({contributor.get('experience_years')} years, {contributor.get('completed_jobs', 0)} total projects)")

    # Reliability reason
    completion_pct = round(contributor.get("completion_rate", 0.0) * 100, 1)
    status = contributor.get("verification_status", "unverified").capitalize()
    reasons.append(f"{status} contributor with a high job completion rate of {completion_pct}%")

    # Availability reason
    avail = contributor.get("availability", "available")
    if avail == "available":
        reasons.append("Currently active and available for immediate task assignment")
    elif avail == "limited":
        reasons.append("Available with limited scheduling capacity")

    return reasons[:5]


def rank_contributors(problem_data, top_n=3):
    """
    Main ranking pipeline:
      1. Resolves geocoordinates from input (if needed)
      2. Loads active contributors from database
      3. Applies hard filters (active, category mismatch, extreme radius, extreme severity)
      4. Computes feature scores using transparent sub-scorers
      5. Calculates deterministic weighted final score
      6. Sorts and returns top N contributors with transparent reasons.
    """
    # Step 1: Resolve location
    location_res = resolve_location(
        location_text=problem_data.get("location_text"),
        latitude=problem_data.get("latitude"),
        longitude=problem_data.get("longitude")
    )
    user_lat = location_res["latitude"]
    user_lon = location_res["longitude"]

    # Step 2: Load contributors from MySQL/DB
    category = (problem_data.get("category") or "").strip().lower()
    raw_contributors = get_all_active_contributors(filter_category=category if category else None)

    # If category filter returned empty, fallback to all active contributors
    if not raw_contributors:
        raw_contributors = get_all_active_contributors()

    problem_severity = (problem_data.get("severity") or "medium").strip().lower()
    problem_sev_level = SEVERITY_LEVELS.get(problem_severity, 2)
    p_min = float(problem_data.get("estimated_cost_min") or 0.0)
    p_max = float(problem_data.get("estimated_cost_max") or 0.0)
    work_scope = problem_data.get("work_scope", "")
    sev_score = float(problem_data.get("severity_score") or 5.0)

    evaluated_candidates = []

    # Step 3 & 4: Filter and Score
    for c in raw_contributors:
        # Hard Filter 1: Inactive
        if not c.get("active", True):
            continue

        # Hard Filter 2: Explicitly unavailable
        if c.get("availability", "").lower() == "unavailable":
            continue

        # Calculate distance
        dist_km = haversine_distance_km(user_lat, user_lon, c.get("latitude"), c.get("longitude"))

        # Hard Filter 3: Extreme distance beyond service radius
        # Allow up to 2.0x service radius boundary before dropping completely
        service_rad = float(c.get("service_radius_km") or 10.0)
        if dist_km is not None and dist_km > (service_rad * 2.2):
            continue

        # Hard Filter 4: Severe underqualification on High severity defects
        # If problem is High (level 3) and contributor can only do Low (level 1)
        cont_sev_level = SEVERITY_LEVELS.get(c.get("max_severity", "medium").lower(), 2)
        if problem_sev_level == 3 and cont_sev_level == 1:
            continue

        # Compute feature scores
        cat_score = calculate_category_skill_score(category, work_scope, c)
        dist_score = calculate_distance_score(dist_km, service_rad)
        budget_score = calculate_budget_score(p_min, p_max, c.get("budget_min"), c.get("budget_max"))
        sev_score_val = calculate_severity_score(problem_severity, sev_score, c.get("max_severity"))
        past_score = calculate_past_work_score(category, c.get("past_work", {}), c.get("experience_years"), c.get("completed_jobs"))
        rel_score = calculate_reliability_score(c.get("completion_rate"), c.get("completed_jobs"), c.get("verification_status"))
        avail_score = calculate_availability_score(c.get("availability"))

        feature_scores = {
            "category_skill": cat_score,
            "distance": dist_score,
            "budget": budget_score,
            "severity": sev_score_val,
            "past_work": past_score,
            "reliability": rel_score,
            "availability": avail_score
        }

        # Weighted final score (0 - 100)
        final_score = (
            cat_score * MATCH_WEIGHTS["category_skill"] +
            dist_score * MATCH_WEIGHTS["distance"] +
            budget_score * MATCH_WEIGHTS["budget"] +
            sev_score_val * MATCH_WEIGHTS["severity"] +
            past_score * MATCH_WEIGHTS["past_work"] +
            rel_score * MATCH_WEIGHTS["reliability"] +
            avail_score * MATCH_WEIGHTS["availability"]
        )
        final_score = round(final_score, 1)

        reasons = generate_transparent_reasons(c, problem_data, feature_scores, dist_km)

        evaluated_candidates.append({
            "contributor_id": c["id"],
            "name": c["name"],
            "organization": c["organization"],
            "role": c["role"],
            "city": c["city"],
            "match_score": final_score,
            "distance_km": dist_km,
            "service_radius_km": service_rad,
            "feature_scores": feature_scores,
            "reasons": reasons,
            "_sort_dist": dist_km if dist_km is not None else 9999.0,
            "_completed_jobs": c.get("completed_jobs", 0),
            "_id": c["id"]
        })

    # Step 5: Deterministic Sort
    # Primary: match_score DESC
    # Secondary: distance_km ASC
    # Tertiary: completed_jobs DESC
    # Deterministic Tie-breaker: contributor_id ASC
    evaluated_candidates.sort(
        key=lambda x: (-x["match_score"], x["_sort_dist"], -x["_completed_jobs"], x["_id"])
    )

    top_matches = evaluated_candidates[:top_n]

    # Clean internal sort helper keys
    clean_matches = []
    for m in top_matches:
        item = {
            "contributor_id": m["contributor_id"],
            "name": m["name"],
            "organization": m["organization"],
            "role": m["role"],
            "city": m["city"],
            "match_score": m["match_score"],
            "distance_km": m["distance_km"],
            "reasons": m["reasons"],
            "feature_breakdown": m["feature_scores"]
        }
        clean_matches.append(item)

    return {
        "success": True,
        "total_evaluated": len(evaluated_candidates),
        "location_resolved": {
            "location_text": location_res["location_text"],
            "latitude": location_res["latitude"],
            "longitude": location_res["longitude"],
            "source": location_res["source"]
        },
        "weights_used": MATCH_WEIGHTS,
        "matches": clean_matches
    }
