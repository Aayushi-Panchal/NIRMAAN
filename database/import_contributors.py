"""
XML Import Utility for NIRMAAN Contributor Matching System.
Parses nirmaan_contributors_dataset.xml and imports contributors into the database.
Idempotent and safe to run multiple times without creating duplicate records.
"""
import os
import sys
try:
    from database import init_db, save_contributor, get_contributor_count
except ImportError:
    from .database import init_db, save_contributor, get_contributor_count

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "database" else BASE_DIR
DEFAULT_XML_PATH = os.path.join(PROJECT_ROOT, "nirmaan_contributors_dataset.xml")
if not os.path.exists(DEFAULT_XML_PATH):
    DEFAULT_XML_PATH = os.path.join(BASE_DIR, "nirmaan_contributors_dataset.xml")


def parse_contributor_node(node):
    """Parses a single <contributor> XML Element into a standard Python dictionary."""
    cid = node.attrib.get("id") or node.findtext("id", "").strip()

    name = node.findtext("name", "").strip()
    organization = node.findtext("organization", "").strip()
    role = node.findtext("role", "").strip()
    city = node.findtext("city", "").strip()

    try:
        lat = float(node.findtext("latitude", "0").strip())
    except ValueError:
        lat = None

    try:
        lon = float(node.findtext("longitude", "0").strip())
    except ValueError:
        lon = None

    try:
        service_radius = float(node.findtext("service_radius_km", "10").strip())
    except ValueError:
        service_radius = 10.0

    # Skills / Capabilities
    skills = []
    caps_node = node.find("capabilities")
    if caps_node is not None:
        for skill_elem in caps_node.findall("skill"):
            if skill_elem.text and skill_elem.text.strip():
                skills.append(skill_elem.text.strip().lower())

    # Supported Categories
    categories = []
    cats_node = node.find("supported_categories")
    if cats_node is not None:
        for cat_elem in cats_node.findall("category"):
            if cat_elem.text and cat_elem.text.strip():
                categories.append(cat_elem.text.strip().lower())

    max_severity = (node.findtext("max_severity", "medium") or "medium").strip().lower()

    # Budget range
    budget_node = node.find("budget_range_inr")
    if budget_node is not None:
        try:
            b_min = float(budget_node.findtext("min", "0").strip())
        except ValueError:
            b_min = 0.0
        try:
            b_max = float(budget_node.findtext("max", "0").strip())
        except ValueError:
            b_max = 0.0
    else:
        b_min = 0.0
        b_max = 0.0

    availability = (node.findtext("availability", "available") or "available").strip().lower()

    try:
        exp_years = int(node.findtext("experience_years", "0").strip())
    except ValueError:
        exp_years = 0

    try:
        completed_jobs = int(node.findtext("completed_jobs", "0").strip())
    except ValueError:
        completed_jobs = 0

    try:
        completion_rate = float(node.findtext("completion_rate", "0").strip())
    except ValueError:
        completion_rate = 0.0

    verification_status = (node.findtext("verification_status", "unverified") or "unverified").strip().lower()

    active_raw = (node.findtext("active", "true") or "true").strip().lower()
    active = active_raw in ("true", "1", "yes")

    # Past work
    past_work = {}
    pw_node = node.find("past_work")
    if pw_node is not None:
        for exp_elem in pw_node.findall("category_experience"):
            cat_name = exp_elem.attrib.get("category", "").strip().lower()
            try:
                cnt = int(exp_elem.text.strip())
            except (ValueError, AttributeError):
                cnt = 0
            if cat_name:
                past_work[cat_name] = cnt

    preferred_work_scope = (node.findtext("preferred_work_scope", "") or "").strip()

    return {
        "id": cid,
        "name": name,
        "organization": organization,
        "role": role,
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "service_radius_km": service_radius,
        "skills": skills,
        "categories": categories,
        "max_severity": max_severity,
        "budget_min": b_min,
        "budget_max": b_max,
        "availability": availability,
        "experience_years": exp_years,
        "completed_jobs": completed_jobs,
        "completion_rate": completion_rate,
        "verification_status": verification_status,
        "active": active,
        "past_work": past_work,
        "preferred_work_scope": preferred_work_scope
    }


def import_contributors_from_xml(xml_path=DEFAULT_XML_PATH, drop_first=False):
    """
    Parses the contributor XML file and stores all records in the database.
    Returns the count of successfully imported records.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML dataset file not found at: {xml_path}")

    print(f"[Import] Initializing database schema (drop_first={drop_first})...")
    init_db(drop_existing=drop_first)

    print(f"[Import] Parsing XML: {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    contributor_nodes = root.findall(".//contributor")
    total_found = len(contributor_nodes)
    print(f"[Import] Found {total_found} contributor records in XML.")

    imported_count = 0
    total_skills = 0
    total_categories = 0
    total_past_work = 0

    for idx, node in enumerate(contributor_nodes, start=1):
        data = parse_contributor_node(node)
        save_contributor(data)
        imported_count += 1
        total_skills += len(data["skills"])
        total_categories += len(data["categories"])
        total_past_work += len(data["past_work"])

        if idx % 200 == 0 or idx == total_found:
            print(f"[Import] Progress: {idx}/{total_found} records processed...")

    total_in_db = get_contributor_count()
    print(f"\n========================================================")
    print(f"  NIRMAAN CONTRIBUTOR IMPORT COMPLETED")
    print(f"  Records Parsed & Saved : {imported_count}")
    print(f"  Skills Linked          : {total_skills}")
    print(f"  Categories Linked      : {total_categories}")
    print(f"  Past Work Entries      : {total_past_work}")
    print(f"  Total Contributors in DB: {total_in_db}")
    print(f"========================================================\n")

    return imported_count


if __name__ == "__main__":
    drop = "--drop" in sys.argv
    path = DEFAULT_XML_PATH
    for arg in sys.argv[1:]:
        if arg.endswith(".xml"):
            path = arg
    import_contributors_from_xml(xml_path=path, drop_first=drop)
