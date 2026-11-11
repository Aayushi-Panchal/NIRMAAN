# NIRMAAN Contributor Matching System (V1 Architecture & Design)

## 1. Overview
The NIRMAAN Contributor Matching System is an explainable, deterministic backend recommendation engine that connects detected civic infrastructure issues (potholes, streetlights, waterlogging, garbage) with the most qualified, nearby, and available local contractors, civil engineers, and CSR organizations.

The system ingests the outputs from the existing NIRMAAN AI Pipeline:
1. **Problem Category** (from ResNet-50 Recognition model)
2. **Defect Severity & Severity Score** (from ResNet-18 Severity model)
3. **Estimated Repair Cost Range & Work Scope** (from Rule-based Cost Estimation Engine)
4. **User Location** (Address / Locality / GPS coordinates)

It then filters, evaluates, and ranks all registered contributors to return the **Top 3 contributors** with transparent, auditable justifications for each ranking.

---

## 2. Contributor Dataset
- **Source**: `nirmaan_contributors_dataset.xml`
- **Volume**: 1,000 synthetic pilot records representing the Sonipat, Delhi NCR, and Haryana municipal regions.
- **Fields Captured**:
  - `id`: Unique contributor identifier (`NRM-00001` to `NRM-01000`)
  - `name`, `organization`, `role` (contractor, civil engineer, company CSR, NGO)
  - `city`, `latitude`, `longitude`, `service_radius_km`
  - `capabilities` / `skills` (`road_repair`, `civil_construction`, `drainage`, `electrical`, `waste_management`, etc.)
  - `supported_categories` (`potholes`, `streetlights`, `waterlogging`, `garbage`)
  - `max_severity` (`low`, `medium`, `high`)
  - `budget_range_inr` (`min`, `max`)
  - `availability` (`available`, `busy`, `limited`, `unavailable`)
  - `experience_years`, `completed_jobs`, `completion_rate`
  - `verification_status` (`verified`, `pending`, `unverified`)
  - `active` (`true`, `false`)
  - `past_work`: Direct count of completed jobs per civic category
  - `preferred_work_scope`

---

## 3. Database Architecture & Schema
The data is persisted in a relational schema. The system connects primarily to **MySQL** (configured via `.env` / environment variables) and includes a resilient dual-mode SQLite fallback for local developer setups.

### Relational Schema:
1. **`contributors`**:
   - `id` VARCHAR(32) PRIMARY KEY
   - `name` VARCHAR(150), `organization` VARCHAR(200), `role` VARCHAR(64), `city` VARCHAR(100)
   - `latitude` DOUBLE, `longitude` DOUBLE, `service_radius_km` DOUBLE
   - `max_severity` VARCHAR(20), `budget_min` DOUBLE, `budget_max` DOUBLE
   - `availability` VARCHAR(30), `experience_years` INT, `completed_jobs` INT
   - `completion_rate` DOUBLE, `verification_status` VARCHAR(30), `active` TINYINT(1)
   - `preferred_work_scope` VARCHAR(150)
   - `created_at`, `updated_at`

2. **`contributor_skills`**:
   - `id` INT AUTO_INCREMENT PRIMARY KEY
   - `contributor_id` VARCHAR(32) FOREIGN KEY -> `contributors(id)` ON DELETE CASCADE
   - `skill` VARCHAR(100)
   - `UNIQUE(contributor_id, skill)`

3. **`contributor_categories`**:
   - `id` INT AUTO_INCREMENT PRIMARY KEY
   - `contributor_id` VARCHAR(32) FOREIGN KEY -> `contributors(id)` ON DELETE CASCADE
   - `category` VARCHAR(100)
   - `UNIQUE(contributor_id, category)`

4. **`contributor_past_work`**:
   - `id` INT AUTO_INCREMENT PRIMARY KEY
   - `contributor_id` VARCHAR(32) FOREIGN KEY -> `contributors(id)` ON DELETE CASCADE
   - `category` VARCHAR(100), `job_count` INT
   - `UNIQUE(contributor_id, category)`

### Idempotent XML Importer (`import_contributors.py`):
```bash
python import_contributors.py
```
- Performs `INSERT ... ON DUPLICATE KEY UPDATE` (or `ON CONFLICT DO UPDATE`).
- Re-running the script is completely safe and updates existing records without duplicate rows or primary key conflicts.

---

## 4. Location & Geocoding Service (`geocoding_service.py`)
Users are never forced to manually input latitude and longitude coordinates.
- **Regional Fast Lookup**: High-precision offline coordinates for Sonipat sectors (Sector 14, 15, 21, Model Town, Murthal, Kundli, Ganaur) and NCR cities (Delhi, Rohini, Bawana, Narela, Gurugram, Rohtak, Panipat, Bahadurgarh).
- **Keyword & Substring Extraction**: Extracts locality tokens from free-form user text (e.g., `"Near Market, Sector 21, Sonipat"` -> `28.9855, 77.0260`).
- **Graceful Failure Handling**: If coordinates cannot be resolved, the system defaults to `(None, None)` and assigns a neutral proximity score rather than throwing an unhandled exception.

---

## 5. Distance Calculation
Geographic distance between problem coordinates and contributor base is computed using the spherical **Haversine formula**:
$$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$
where $R = 6371.0\text{ km}$.
- Distance is evaluated relative to the contributor's individual `service_radius_km`.
- Within the service radius, the distance score scales from 100.0 (0 km) down to 55.0 (at boundary).
- Beyond the service radius, an exponential decay penalty is applied.

---

## 6. Hard Filters
Before ranking, non-viable candidates are eliminated:
1. **Inactive Filter**: Contributor marked `active == False` is excluded.
2. **Availability Filter**: Contributors with `availability == 'unavailable'` are excluded.
3. **Severe Radius Filter**: Excludes contributors located beyond 2.2x their maximum operational service radius.
4. **Severity Underqualification Filter**: Excludes contributors with `max_severity == 'low'` from being assigned dangerous `High` severity infrastructure issues.

---

## 7. Feature Scoring & Configurable Weights
Scoring functions calculate normalized sub-scores from $0.0$ to $100.0$:

| Feature Sub-Score | Weight | Key Determinants |
|---|---|---|
| **Category & Skill Compatibility** | **30%** | Category registration, domain skills (`road_repair`, `drainage`, `electrical`, `waste_management`), work scope keyword overlap. |
| **Distance & Service Radius** | **20%** | Haversine distance relative to operational service radius. |
| **Budget Compatibility** | **15%** | Intersecting range overlap between AI estimated repair cost `[min, max]` and contributor budget tier `[min, max]`. |
| **Severity Capability** | **10%** | Defect severity rating (Low, Medium, High) mapped to contributor capability. |
| **Similar Past Work Experience** | **10%** | Verified completed job count specifically within the reported category, plus overall career tenure. |
| **Reliability & Verification** | **10%** | Historical completion rate percentage, verification tier (`verified`, `pending`, `unverified`), and project volume confidence. |
| **Availability** | **5%** | Operational readiness (`available` = 100, `limited` = 65, `busy` = 30). |

Configured in `db_config.py`:
```python
MATCH_WEIGHTS = {
    "category_skill": 0.30,
    "distance": 0.20,
    "budget": 0.15,
    "severity": 0.10,
    "past_work": 0.10,
    "reliability": 0.10,
    "availability": 0.05
}
```

---

## 8. Deterministic Ranking & Tie-Breaking
The system guarantees that the exact same problem input against the same database state produces the identical ranking and scores every time:
- No non-deterministic random numbers or stochastic temperature factors.
- Deterministic multi-tier sorting key:
  1. `match_score` (Descending)
  2. `distance_km` (Ascending)
  3. `completed_jobs` (Descending)
  4. `contributor_id` (Ascending, deterministic tie-breaker)

---

## 9. Backend API (`POST /api/match`)

### Endpoint:
`POST /api/match`

### Request Payload:
```json
{
  "category": "potholes",
  "severity": "High",
  "severity_score": 6.7,
  "estimated_cost_min": 12100,
  "estimated_cost_max": 15300,
  "work_scope": "Road surface repair",
  "location_text": "Sector 21, Sonipat",
  "latitude": 28.9855,
  "longitude": 77.0260
}
```

### Example Response:
```json
{
  "success": true,
  "total_evaluated": 231,
  "location_resolved": {
    "location_text": "Sector 21, Sonipat",
    "latitude": 28.9855,
    "longitude": 77.026,
    "source": "provided_coordinates"
  },
  "weights_used": {
    "category_skill": 0.3,
    "distance": 0.2,
    "budget": 0.15,
    "severity": 0.1,
    "past_work": 0.1,
    "reliability": 0.1,
    "availability": 0.05
  },
  "matches": [
    {
      "contributor_id": "NRM-00483",
      "name": "Naveen Singhal",
      "organization": "Singhal Technical Services",
      "role": "engineer",
      "city": "Sonipat",
      "match_score": 91.8,
      "distance_km": 1.98,
      "reasons": [
        "Possesses specialized capabilities: civil_construction, road_repair",
        "Located 1.98 km away, comfortably within service radius (21 km)",
        "Budget tier (₹12,500 - ₹34,500) aligns with estimated repair cost (₹12,100 - ₹15,300)",
        "Proven track record with 29 completed potholes civic assignments",
        "Verified contributor with a high job completion rate of 95.8%"
      ],
      "feature_breakdown": {
        "category_skill": 100.0,
        "distance": 95.76,
        "budget": 100.0,
        "severity": 100.0,
        "past_work": 78.5,
        "reliability": 86.48,
        "availability": 100.0
      }
    }
  ]
}
```

---

## 10. Why Matching V1 is Rule/Feature-Based Instead of ML
**Crucial Architectural Decision**:
The initial contributor dataset (`nirmaan_contributors_dataset.xml`) contains profile attributes and past volume statistics, but does **not** contain historical assignment outcomes, such as:
- Which contributor was dispatched for a specific complaint?
- Did the contributor accept or reject the job?
- What was the response turnaround time?
- What was the citizen satisfaction rating or municipal audit score?

Without real ground-truth label pairs `(defect, contributor, outcome_quality)`, fitting a machine learning classifier or deep neural network would be mathematically ungrounded and would represent a **"fake AI"** hallucination.

Instead, Matching V1 implements an **industry-standard, auditable, deterministic Multi-Criteria Decision Analysis (MCDA) weighted feature ranking system**. Every point in the final match score is directly traceable to concrete physical and operational attributes.

---

## 11. Limitations & Roadmap to Matching V2
### Current Limitations:
- Real-time traffic and road network topology are not factored into distance calculations (Euclidean Haversine distance is used).
- Real-time work in progress (WIP) load balancing across active crews is simplified to static availability flags.

### Matching V2 Roadmap (Learning-to-Rank):
Once NIRMAAN processes live reports with verified contractor dispatches:
1. Log telemetry on contractor acceptance rates, completion times, and citizen satisfaction ratings.
2. Upgrade from weighted feature ranking to a **Learning-to-Rank (LTR)** model (e.g. LightGBM Ranker / LambdaMART).
3. Continually tune feature importance weights based on observed empirical assignment success rates.
