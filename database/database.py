"""
Database connection and repository module for NIRMAAN Contributor Matching & Civic Workflow System.
Handles schema initialization, idempotent contributor storage, structured retrieval,
and end-to-end civic lifecycle tracking:
  REPORT -> AI ASSESSMENT -> FUNDING -> JOB -> PROGRESS -> ADMIN VERIFICATION -> FUND RELEASE -> COMPLETED

Supports MySQL (primary) and an automated SQLite fallback for resilience.
"""
import os
import uuid
import datetime
import sqlite3
import pymysql
from pymysql.cursors import DictCursor
try:
    from db_config import (
        MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    )
except ImportError:
    from .db_config import (
        MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    )

# SQLite fallback path - resolve to the populated database file
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) if os.path.basename(_CURRENT_DIR) == "database" else _CURRENT_DIR

_DB_IN_FOLDER = os.path.join(_CURRENT_DIR, "nirmaan_contributors.db")
_DB_IN_ROOT = os.path.join(_PROJECT_ROOT, "nirmaan_contributors.db")

# Prefer existing populated database
if os.path.exists(_DB_IN_FOLDER) and os.path.getsize(_DB_IN_FOLDER) > 4096:
    SQLITE_DB_PATH = _DB_IN_FOLDER
elif os.path.exists(_DB_IN_ROOT) and os.path.getsize(_DB_IN_ROOT) > 4096:
    SQLITE_DB_PATH = _DB_IN_ROOT
elif os.path.exists(_DB_IN_FOLDER):
    SQLITE_DB_PATH = _DB_IN_FOLDER
else:
    SQLITE_DB_PATH = _DB_IN_ROOT

_USE_SQLITE = False


def check_mysql_connection():
    """Attempts a test connection to MySQL server."""
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connect_timeout=3
        )
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def get_connection():
    """
    Returns an open database connection and a string indicating 'mysql' or 'sqlite'.
    """
    global _USE_SQLITE
    if not _USE_SQLITE:
        try:
            # First ensure database exists
            server_conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                connect_timeout=3
            )
            with server_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            server_conn.commit()
            server_conn.close()

            # Connect to the specific database
            conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                cursorclass=DictCursor,
                autocommit=True
            )
            return conn, "mysql"
        except Exception as e:
            print(f"[Database] MySQL connection failed ({e}). Falling back to SQLite for resilience.")
            _USE_SQLITE = True

    # SQLite fallback
    conn = sqlite3.connect(SQLITE_DB_PATH, timeout=60.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 60000;")
    except Exception:
        pass
    return conn, "sqlite"


def init_db(drop_existing=False):
    """
    Creates all required relational tables for NIRMAAN:
      - contributors, contributor_skills, contributor_categories, contributor_past_work
      - users
      - reports
      - funding_campaigns
      - funding_transactions
      - assignments
      - task_updates
      - verification_records
      - status_history
      - rewards_ledger
    """
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                if drop_existing:
                    cur.execute("SET FOREIGN_KEY_CHECKS = 0;")
                    cur.execute("DROP TABLE IF EXISTS rewards_ledger;")
                    cur.execute("DROP TABLE IF EXISTS status_history;")
                    cur.execute("DROP TABLE IF EXISTS verification_records;")
                    cur.execute("DROP TABLE IF EXISTS task_updates;")
                    cur.execute("DROP TABLE IF EXISTS assignments;")
                    cur.execute("DROP TABLE IF EXISTS funding_transactions;")
                    cur.execute("DROP TABLE IF EXISTS funding_campaigns;")
                    cur.execute("DROP TABLE IF EXISTS reports;")
                    cur.execute("DROP TABLE IF EXISTS users;")
                    cur.execute("DROP TABLE IF EXISTS contributor_past_work;")
                    cur.execute("DROP TABLE IF EXISTS contributor_categories;")
                    cur.execute("DROP TABLE IF EXISTS contributor_skills;")
                    cur.execute("DROP TABLE IF EXISTS contributors;")
                    cur.execute("SET FOREIGN_KEY_CHECKS = 1;")

                # 1. Existing contributors tables
                cur.execute("""
                CREATE TABLE IF NOT EXISTS contributors (
                    id VARCHAR(32) PRIMARY KEY,
                    name VARCHAR(150) NOT NULL,
                    organization VARCHAR(200) NOT NULL,
                    role VARCHAR(64) NOT NULL,
                    city VARCHAR(100) NOT NULL,
                    latitude DOUBLE,
                    longitude DOUBLE,
                    service_radius_km DOUBLE NOT NULL DEFAULT 10.0,
                    max_severity VARCHAR(20) NOT NULL DEFAULT 'medium',
                    budget_min DOUBLE NOT NULL DEFAULT 0.0,
                    budget_max DOUBLE NOT NULL DEFAULT 0.0,
                    availability VARCHAR(30) NOT NULL DEFAULT 'available',
                    experience_years INT NOT NULL DEFAULT 0,
                    completed_jobs INT NOT NULL DEFAULT 0,
                    completion_rate DOUBLE NOT NULL DEFAULT 0.0,
                    verification_status VARCHAR(30) NOT NULL DEFAULT 'unverified',
                    active TINYINT(1) NOT NULL DEFAULT 1,
                    preferred_work_scope VARCHAR(150),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_active (active),
                    INDEX idx_city (city),
                    INDEX idx_max_severity (max_severity)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                cur.execute("""
                CREATE TABLE IF NOT EXISTS contributor_skills (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    contributor_id VARCHAR(32) NOT NULL,
                    skill VARCHAR(100) NOT NULL,
                    UNIQUE KEY uq_contrib_skill (contributor_id, skill),
                    FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE,
                    INDEX idx_skill (skill)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                cur.execute("""
                CREATE TABLE IF NOT EXISTS contributor_categories (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    contributor_id VARCHAR(32) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    UNIQUE KEY uq_contrib_category (contributor_id, category),
                    FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE,
                    INDEX idx_category (category)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                cur.execute("""
                CREATE TABLE IF NOT EXISTS contributor_past_work (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    contributor_id VARCHAR(32) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    job_count INT NOT NULL DEFAULT 0,
                    UNIQUE KEY uq_contrib_past_work (contributor_id, category),
                    FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE,
                    INDEX idx_past_cat (category)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 2. Users Table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(150) NOT NULL,
                    email VARCHAR(150) UNIQUE NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    organization VARCHAR(150),
                    phone VARCHAR(50),
                    avatar VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_role (role)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 3. Reports Table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64),
                    title VARCHAR(255) NOT NULL,
                    category VARCHAR(64) NOT NULL,
                    location_text VARCHAR(255) NOT NULL,
                    latitude DOUBLE,
                    longitude DOUBLE,
                    image_path VARCHAR(255) NOT NULL,
                    status VARCHAR(64) NOT NULL DEFAULT 'REPORT',
                    ai_category VARCHAR(64),
                    ai_confidence DOUBLE,
                    severity VARCHAR(32),
                    severity_score DOUBLE,
                    severity_reason TEXT,
                    estimated_cost_min DOUBLE,
                    estimated_cost_max DOUBLE,
                    work_scope TEXT,
                    eta_days INT DEFAULT 7,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_rep_status (status),
                    INDEX idx_rep_cat (category),
                    INDEX idx_rep_user (user_id),
                    INDEX idx_rep_loc (location_text(100))
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 4. Funding Campaigns
                cur.execute("""
                CREATE TABLE IF NOT EXISTS funding_campaigns (
                    id VARCHAR(64) PRIMARY KEY,
                    report_id VARCHAR(64) UNIQUE NOT NULL,
                    goal_amount DOUBLE NOT NULL,
                    raised_amount DOUBLE NOT NULL DEFAULT 0.0,
                    released_amount DOUBLE NOT NULL DEFAULT 0.0,
                    available_amount DOUBLE NOT NULL DEFAULT 0.0,
                    funder_count INT NOT NULL DEFAULT 0,
                    status VARCHAR(64) NOT NULL DEFAULT 'ACTIVE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_fc_status (status),
                    INDEX idx_fc_rep (report_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 5. Funding Transactions (Ledger)
                cur.execute("""
                CREATE TABLE IF NOT EXISTS funding_transactions (
                    id VARCHAR(64) PRIMARY KEY,
                    campaign_id VARCHAR(64) NOT NULL,
                    user_id VARCHAR(64),
                    donor_name VARCHAR(150) NOT NULL,
                    amount DOUBLE NOT NULL,
                    transaction_type VARCHAR(50) DEFAULT 'CONTRIBUTION',
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_ft_camp (campaign_id),
                    INDEX idx_ft_user (user_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 6. Assignments
                cur.execute("""
                CREATE TABLE IF NOT EXISTS assignments (
                    id VARCHAR(64) PRIMARY KEY,
                    report_id VARCHAR(64) NOT NULL,
                    contributor_id VARCHAR(64) NOT NULL,
                    contributor_name VARCHAR(150),
                    contributor_role VARCHAR(64),
                    status VARCHAR(64) NOT NULL DEFAULT 'ASSIGNED',
                    accepted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP NULL,
                    completed_at TIMESTAMP NULL,
                    INDEX idx_asgn_rep (report_id),
                    INDEX idx_asgn_contrib (contributor_id),
                    INDEX idx_asgn_stat (status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 7. Task Updates
                cur.execute("""
                CREATE TABLE IF NOT EXISTS task_updates (
                    id VARCHAR(64) PRIMARY KEY,
                    assignment_id VARCHAR(64) NOT NULL,
                    report_id VARCHAR(64) NOT NULL,
                    progress_pct INT NOT NULL,
                    work_note TEXT,
                    progress_image VARCHAR(255),
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified TINYINT(1) DEFAULT 0,
                    verified_at TIMESTAMP NULL,
                    verified_by VARCHAR(64) NULL,
                    INDEX idx_tu_rep (report_id),
                    INDEX idx_tu_asgn (assignment_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 8. Verification Records
                cur.execute("""
                CREATE TABLE IF NOT EXISTS verification_records (
                    id VARCHAR(64) PRIMARY KEY,
                    task_update_id VARCHAR(64) NOT NULL,
                    report_id VARCHAR(64) NOT NULL,
                    admin_id VARCHAR(64) NOT NULL,
                    action VARCHAR(64) NOT NULL,
                    notes TEXT,
                    amount_released DOUBLE DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_vr_rep (report_id),
                    INDEX idx_vr_tu (task_update_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 9. Status History
                cur.execute("""
                CREATE TABLE IF NOT EXISTS status_history (
                    id VARCHAR(64) PRIMARY KEY,
                    report_id VARCHAR(64) NOT NULL,
                    from_status VARCHAR(64),
                    to_status VARCHAR(64) NOT NULL,
                    actor_name VARCHAR(150),
                    actor_role VARCHAR(64),
                    note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_sh_rep (report_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

                # 10. Rewards Ledger
                cur.execute("""
                CREATE TABLE IF NOT EXISTS rewards_ledger (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id VARCHAR(64) UNIQUE NOT NULL,
                    user_name VARCHAR(150),
                    points INT NOT NULL DEFAULT 0,
                    contributed_amount DOUBLE NOT NULL DEFAULT 0.0,
                    problems_backed INT NOT NULL DEFAULT 0,
                    tier VARCHAR(32) DEFAULT 'Bronze',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_rew_points (points DESC)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

        else:
            # SQLite syntax
            cur = conn.cursor()
            if drop_existing:
                cur.execute("DROP TABLE IF EXISTS rewards_ledger;")
                cur.execute("DROP TABLE IF EXISTS status_history;")
                cur.execute("DROP TABLE IF EXISTS verification_records;")
                cur.execute("DROP TABLE IF EXISTS task_updates;")
                cur.execute("DROP TABLE IF EXISTS assignments;")
                cur.execute("DROP TABLE IF EXISTS funding_transactions;")
                cur.execute("DROP TABLE IF EXISTS funding_campaigns;")
                cur.execute("DROP TABLE IF EXISTS reports;")
                cur.execute("DROP TABLE IF EXISTS users;")
                cur.execute("DROP TABLE IF EXISTS contributor_past_work;")
                cur.execute("DROP TABLE IF EXISTS contributor_categories;")
                cur.execute("DROP TABLE IF EXISTS contributor_skills;")
                cur.execute("DROP TABLE IF EXISTS contributors;")

            # 1. Existing contributors tables
            cur.execute("""
            CREATE TABLE IF NOT EXISTS contributors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                organization TEXT NOT NULL,
                role TEXT NOT NULL,
                city TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                service_radius_km REAL NOT NULL DEFAULT 10.0,
                max_severity TEXT NOT NULL DEFAULT 'medium',
                budget_min REAL NOT NULL DEFAULT 0.0,
                budget_max REAL NOT NULL DEFAULT 0.0,
                availability TEXT NOT NULL DEFAULT 'available',
                experience_years INTEGER NOT NULL DEFAULT 0,
                completed_jobs INTEGER NOT NULL DEFAULT 0,
                completion_rate REAL NOT NULL DEFAULT 0.0,
                verification_status TEXT NOT NULL DEFAULT 'unverified',
                active INTEGER NOT NULL DEFAULT 1,
                preferred_work_scope TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS contributor_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contributor_id TEXT NOT NULL,
                skill TEXT NOT NULL,
                UNIQUE (contributor_id, skill),
                FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS contributor_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contributor_id TEXT NOT NULL,
                category TEXT NOT NULL,
                UNIQUE (contributor_id, category),
                FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS contributor_past_work (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contributor_id TEXT NOT NULL,
                category TEXT NOT NULL,
                job_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE (contributor_id, category),
                FOREIGN KEY (contributor_id) REFERENCES contributors(id) ON DELETE CASCADE
            );
            """)

            # 2. Users Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                organization TEXT,
                phone TEXT,
                avatar TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 3. Reports Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                location_text TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                image_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'REPORT',
                ai_category TEXT,
                ai_confidence REAL,
                severity TEXT,
                severity_score REAL,
                severity_reason TEXT,
                estimated_cost_min REAL,
                estimated_cost_max REAL,
                work_scope TEXT,
                eta_days INTEGER DEFAULT 7,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. Funding Campaigns
            cur.execute("""
            CREATE TABLE IF NOT EXISTS funding_campaigns (
                id TEXT PRIMARY KEY,
                report_id TEXT UNIQUE NOT NULL,
                goal_amount REAL NOT NULL,
                raised_amount REAL NOT NULL DEFAULT 0.0,
                released_amount REAL NOT NULL DEFAULT 0.0,
                available_amount REAL NOT NULL DEFAULT 0.0,
                funder_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 5. Funding Transactions
            cur.execute("""
            CREATE TABLE IF NOT EXISTS funding_transactions (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                user_id TEXT,
                donor_name TEXT NOT NULL,
                amount REAL NOT NULL,
                transaction_type TEXT DEFAULT 'CONTRIBUTION',
                note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 6. Assignments
            cur.execute("""
            CREATE TABLE IF NOT EXISTS assignments (
                id TEXT PRIMARY KEY,
                report_id TEXT NOT NULL,
                contributor_id TEXT NOT NULL,
                contributor_name TEXT,
                contributor_role TEXT,
                status TEXT NOT NULL DEFAULT 'ASSIGNED',
                accepted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                completed_at TEXT
            );
            """)

            # 7. Task Updates
            cur.execute("""
            CREATE TABLE IF NOT EXISTS task_updates (
                id TEXT PRIMARY KEY,
                assignment_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                progress_pct INTEGER NOT NULL,
                work_note TEXT,
                progress_image TEXT,
                submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                verified INTEGER DEFAULT 0,
                verified_at TEXT,
                verified_by TEXT
            );
            """)

            # 8. Verification Records
            cur.execute("""
            CREATE TABLE IF NOT EXISTS verification_records (
                id TEXT PRIMARY KEY,
                task_update_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                admin_id TEXT NOT NULL,
                action TEXT NOT NULL,
                notes TEXT,
                amount_released REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 9. Status History
            cur.execute("""
            CREATE TABLE IF NOT EXISTS status_history (
                id TEXT PRIMARY KEY,
                report_id TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                actor_name TEXT,
                actor_role TEXT,
                note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 10. Rewards Ledger
            cur.execute("""
            CREATE TABLE IF NOT EXISTS rewards_ledger (
                id TEXT PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL,
                user_name TEXT,
                points INTEGER NOT NULL DEFAULT 0,
                contributed_amount REAL NOT NULL DEFAULT 0.0,
                problems_backed INTEGER NOT NULL DEFAULT 0,
                tier TEXT DEFAULT 'Bronze',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # SQLite indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rep_status ON reports (status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rep_cat ON reports (category);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fc_status ON funding_campaigns (status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_asgn_rep ON assignments (report_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tu_rep ON task_updates (report_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sh_rep ON status_history (report_id);")
            conn.commit()

        print(f"[Database] NIRMAAN schema verified and ready on engine: {engine}")
    finally:
        conn.close()


def save_contributor(data):
    """
    Idempotently saves or updates a contributor record along with skills,
    supported categories, and past work history.
    """
    conn, engine = get_connection()
    try:
        cid = data["id"]
        active_val = 1 if data.get("active", True) else 0

        if engine == "mysql":
            with conn.cursor() as cur:
                # Upsert contributor main record
                sql = """
                INSERT INTO contributors (
                    id, name, organization, role, city, latitude, longitude,
                    service_radius_km, max_severity, budget_min, budget_max,
                    availability, experience_years, completed_jobs, completion_rate,
                    verification_status, active, preferred_work_scope
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                ) ON DUPLICATE KEY UPDATE
                    name=VALUES(name), organization=VALUES(organization), role=VALUES(role),
                    city=VALUES(city), latitude=VALUES(latitude), longitude=VALUES(longitude),
                    service_radius_km=VALUES(service_radius_km), max_severity=VALUES(max_severity),
                    budget_min=VALUES(budget_min), budget_max=VALUES(budget_max),
                    availability=VALUES(availability), experience_years=VALUES(experience_years),
                    completed_jobs=VALUES(completed_jobs), completion_rate=VALUES(completion_rate),
                    verification_status=VALUES(verification_status), active=VALUES(active),
                    preferred_work_scope=VALUES(preferred_work_scope);
                """
                cur.execute(sql, (
                    cid, data["name"], data["organization"], data["role"], data["city"],
                    data["latitude"], data["longitude"], data["service_radius_km"],
                    data["max_severity"], data["budget_min"], data["budget_max"],
                    data["availability"], data["experience_years"], data["completed_jobs"],
                    data["completion_rate"], data["verification_status"], active_val,
                    data.get("preferred_work_scope", "")
                ))

                # Skills
                cur.execute("DELETE FROM contributor_skills WHERE contributor_id = %s;", (cid,))
                for skill in data.get("skills", []):
                    cur.execute(
                        "INSERT IGNORE INTO contributor_skills (contributor_id, skill) VALUES (%s, %s);",
                        (cid, skill.strip())
                    )

                # Categories
                cur.execute("DELETE FROM contributor_categories WHERE contributor_id = %s;", (cid,))
                for cat in data.get("categories", []):
                    cur.execute(
                        "INSERT IGNORE INTO contributor_categories (contributor_id, category) VALUES (%s, %s);",
                        (cid, cat.strip())
                    )

                # Past Work
                cur.execute("DELETE FROM contributor_past_work WHERE contributor_id = %s;", (cid,))
                for cat, count in data.get("past_work", {}).items():
                    cur.execute(
                        "INSERT IGNORE INTO contributor_past_work (contributor_id, category, job_count) VALUES (%s, %s, %s);",
                        (cid, cat.strip(), int(count))
                    )

        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO contributors (
                id, name, organization, role, city, latitude, longitude,
                service_radius_km, max_severity, budget_min, budget_max,
                availability, experience_years, completed_jobs, completion_rate,
                verification_status, active, preferred_work_scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, organization=excluded.organization, role=excluded.role,
                city=excluded.city, latitude=excluded.latitude, longitude=excluded.longitude,
                service_radius_km=excluded.service_radius_km, max_severity=excluded.max_severity,
                budget_min=excluded.budget_min, budget_max=excluded.budget_max,
                availability=excluded.availability, experience_years=excluded.experience_years,
                completed_jobs=excluded.completed_jobs, completion_rate=excluded.completion_rate,
                verification_status=excluded.verification_status, active=excluded.active,
                preferred_work_scope=excluded.preferred_work_scope;
            """, (
                cid, data["name"], data["organization"], data["role"], data["city"],
                data["latitude"], data["longitude"], data["service_radius_km"],
                data["max_severity"], data["budget_min"], data["budget_max"],
                data["availability"], data["experience_years"], data["completed_jobs"],
                data["completion_rate"], data["verification_status"], active_val,
                data.get("preferred_work_scope", "")
            ))

            cur.execute("DELETE FROM contributor_skills WHERE contributor_id = ?;", (cid,))
            for skill in data.get("skills", []):
                cur.execute(
                    "INSERT OR IGNORE INTO contributor_skills (contributor_id, skill) VALUES (?, ?);",
                    (cid, skill.strip())
                )

            cur.execute("DELETE FROM contributor_categories WHERE contributor_id = ?;", (cid,))
            for cat in data.get("categories", []):
                cur.execute(
                    "INSERT OR IGNORE INTO contributor_categories (contributor_id, category) VALUES (?, ?);",
                    (cid, cat.strip())
                )

            cur.execute("DELETE FROM contributor_past_work WHERE contributor_id = ?;", (cid,))
            for cat, count in data.get("past_work", {}).items():
                cur.execute(
                    "INSERT OR IGNORE INTO contributor_past_work (contributor_id, category, job_count) VALUES (?, ?, ?);",
                    (cid, cat.strip(), int(count))
                )
            conn.commit()

    finally:
        conn.close()


def get_all_active_contributors(filter_category=None):
    """
    Retrieves all active contributors along with their skills,
    supported categories, and past work from the database.
    """
    conn, engine = get_connection()
    try:
        contributors = {}
        if engine == "mysql":
            with conn.cursor() as cur:
                if filter_category:
                    sql = """
                    SELECT DISTINCT c.* FROM contributors c
                    JOIN contributor_categories cc ON c.id = cc.contributor_id
                    WHERE c.active = 1 AND cc.category = %s;
                    """
                    cur.execute(sql, (filter_category,))
                else:
                    cur.execute("SELECT * FROM contributors WHERE active = 1;")
                rows = cur.fetchall()

                for r in rows:
                    cid = r["id"]
                    contributors[cid] = dict(r)
                    contributors[cid]["skills"] = []
                    contributors[cid]["categories"] = []
                    contributors[cid]["past_work"] = {}

                if not contributors:
                    return []

                cids = list(contributors.keys())
                format_strings = ','.join(['%s'] * len(cids))

                # Fetch skills
                cur.execute(f"SELECT contributor_id, skill FROM contributor_skills WHERE contributor_id IN ({format_strings});", cids)
                for s in cur.fetchall():
                    contributors[s["contributor_id"]]["skills"].append(s["skill"])

                # Fetch categories
                cur.execute(f"SELECT contributor_id, category FROM contributor_categories WHERE contributor_id IN ({format_strings});", cids)
                for cat in cur.fetchall():
                    contributors[cat["contributor_id"]]["categories"].append(cat["category"])

                # Fetch past work
                cur.execute(f"SELECT contributor_id, category, job_count FROM contributor_past_work WHERE contributor_id IN ({format_strings});", cids)
                for pw in cur.fetchall():
                    contributors[pw["contributor_id"]]["past_work"][pw["category"]] = pw["job_count"]

        else:
            cur = conn.cursor()
            if filter_category:
                cur.execute("""
                SELECT DISTINCT c.* FROM contributors c
                JOIN contributor_categories cc ON c.id = cc.contributor_id
                WHERE c.active = 1 AND cc.category = ?;
                """, (filter_category,))
            else:
                cur.execute("SELECT * FROM contributors WHERE active = 1;")
            rows = [dict(r) for r in cur.fetchall()]

            for r in rows:
                cid = r["id"]
                contributors[cid] = r
                contributors[cid]["skills"] = []
                contributors[cid]["categories"] = []
                contributors[cid]["past_work"] = {}

            if not contributors:
                return []

            cids = list(contributors.keys())
            placeholders = ','.join(['?'] * len(cids))

            cur.execute(f"SELECT contributor_id, skill FROM contributor_skills WHERE contributor_id IN ({placeholders});", cids)
            for s in [dict(r) for r in cur.fetchall()]:
                contributors[s["contributor_id"]]["skills"].append(s["skill"])

            cur.execute(f"SELECT contributor_id, category FROM contributor_categories WHERE contributor_id IN ({placeholders});", cids)
            for cat in [dict(r) for r in cur.fetchall()]:
                contributors[cat["contributor_id"]]["categories"].append(cat["category"])

            cur.execute(f"SELECT contributor_id, category, job_count FROM contributor_past_work WHERE contributor_id IN ({placeholders});", cids)
            for pw in [dict(r) for r in cur.fetchall()]:
                contributors[pw["contributor_id"]]["past_work"][pw["category"]] = pw["job_count"]

        return list(contributors.values())
    finally:
        conn.close()


def get_contributor_count():
    """Returns the total number of contributors registered in the database."""
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total FROM contributors;")
                res = cur.fetchone()
                return res["total"] if res else 0
        else:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS total FROM contributors;")
            res = cur.fetchone()
            return res[0] if res else 0
    finally:
        conn.close()


# =========================================================================
# USERS & REWARDS MANAGEMENT
# =========================================================================

def upsert_user(user_data):
    """Inserts or updates a user profile."""
    conn, engine = get_connection()
    try:
        uid = user_data.get("id") or f"usr_{uuid.uuid4().hex[:10]}"
        name = user_data["name"]
        email = user_data["email"]
        role = user_data.get("role", "citizen")
        org = user_data.get("organization", "")
        phone = user_data.get("phone", "")
        avatar = user_data.get("avatar", "")

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO users (id, name, email, role, organization, phone, avatar)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    name=VALUES(name), role=VALUES(role), organization=VALUES(organization),
                    phone=VALUES(phone), avatar=VALUES(avatar);
                """, (uid, name, email, role, org, phone, avatar))
        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO users (id, name, email, role, organization, phone, avatar)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, role=excluded.role, organization=excluded.organization,
                phone=excluded.phone, avatar=excluded.avatar;
            """, (uid, name, email, role, org, phone, avatar))
            conn.commit()

        return uid
    finally:
        conn.close()


def get_all_users():
    """Returns all registered users."""
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users ORDER BY created_at ASC;")
                return cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users ORDER BY created_at ASC;")
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_user_by_id(user_id):
    """Retrieves a single user by ID."""
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE id = %s;", (user_id,))
                return cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ?;", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def upsert_rewards(user_id, points, contributed_amount, problems_backed, user_name=None):
    """Updates rewards ledger for a user."""
    conn, engine = get_connection()
    try:
        rid = f"rew_{user_id}"
        tier = "Bronze"
        if points >= 3000:
            tier = "Platinum"
        elif points >= 1500:
            tier = "Gold"
        elif points >= 500:
            tier = "Silver"

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO rewards_ledger (id, user_id, user_name, points, contributed_amount, problems_backed, tier)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    user_name=COALESCE(VALUES(user_name), user_name),
                    points=VALUES(points),
                    contributed_amount=VALUES(contributed_amount),
                    problems_backed=VALUES(problems_backed),
                    tier=VALUES(tier);
                """, (rid, user_id, user_name, points, contributed_amount, problems_backed, tier))
        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO rewards_ledger (id, user_id, user_name, points, contributed_amount, problems_backed, tier)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                user_name=COALESCE(excluded.user_name, rewards_ledger.user_name),
                points=excluded.points,
                contributed_amount=excluded.contributed_amount,
                problems_backed=excluded.problems_backed,
                tier=excluded.tier;
            """, (rid, user_id, user_name, points, contributed_amount, problems_backed, tier))
            conn.commit()
    finally:
        conn.close()


def compute_dynamic_rewards(target_user_id=None, limit=15):
    """
    Computes fully dynamic, database-driven rewards and leaderboard rankings
    by aggregating across live tables:
      - funding_transactions (real contributions & backed problem counts)
      - assignments & verification_records (real completed & verified jobs)
      - reports (real civic defect reports filed)
      - users & contributors (real roles, names, organizations)
      - rewards_ledger (historical baseline points)
    Updates rewards_ledger with latest calculated figures.
    """
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                # 1. Fetch Users
                cur.execute("SELECT id, name, role, organization FROM users;")
                users = {r["id"]: dict(r) for r in cur.fetchall()}

                # 2. Fetch Funding totals and campaign counts
                cur.execute("""
                    SELECT user_id, donor_name, SUM(amount) AS total_funded,
                           COUNT(DISTINCT campaign_id) AS campaigns_count,
                           COUNT(*) AS tx_count
                    FROM funding_transactions
                    GROUP BY user_id;
                """)
                funding = {r["user_id"]: dict(r) for r in cur.fetchall()}

                # 3. Fetch Completed assignments
                cur.execute("""
                    SELECT contributor_id, contributor_name, contributor_role,
                           SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_fixes,
                           SUM(CASE WHEN status IN ('ASSIGNED', 'IN_PROGRESS') THEN 1 ELSE 0 END) AS active_jobs,
                           COUNT(*) AS total_assignments
                    FROM assignments
                    GROUP BY contributor_id;
                """)
                assignments = {r["contributor_id"]: dict(r) for r in cur.fetchall()}

                # 4. Fetch Verified task updates
                cur.execute("""
                    SELECT a.contributor_id, COUNT(*) AS verified_count
                    FROM verification_records vr
                    JOIN assignments a ON vr.report_id = a.report_id
                    WHERE vr.action = 'APPROVE'
                    GROUP BY a.contributor_id;
                """)
                verified = {r["contributor_id"]: dict(r) for r in cur.fetchall()}

                # 5. Fetch Reports filed per user
                cur.execute("SELECT user_id, COUNT(*) AS reports_count FROM reports GROUP BY user_id;")
                reports = {r["user_id"]: dict(r) for r in cur.fetchall()}

                # 6. Fetch Existing ledger baseline
                cur.execute("SELECT * FROM rewards_ledger;")
                ledger = {r["user_id"]: dict(r) for r in cur.fetchall()}
        else:
            cur = conn.cursor()
            # 1. Fetch Users
            cur.execute("SELECT id, name, role, organization FROM users;")
            users = {r[0]: {"id": r[0], "name": r[1], "role": r[2], "organization": r[3]} for r in cur.fetchall()}

            # 2. Fetch Funding totals and campaign counts
            cur.execute("""
                SELECT user_id, donor_name, SUM(amount) AS total_funded,
                       COUNT(DISTINCT campaign_id) AS campaigns_count,
                       COUNT(*) AS tx_count
                FROM funding_transactions
                GROUP BY user_id;
            """)
            funding = {r[0]: {"user_id": r[0], "donor_name": r[1], "total_funded": r[2], "campaigns_count": r[3], "tx_count": r[4]} for r in cur.fetchall()}

            # 3. Fetch Completed assignments
            cur.execute("""
                SELECT contributor_id, contributor_name, contributor_role,
                       SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_fixes,
                       SUM(CASE WHEN status IN ('ASSIGNED', 'IN_PROGRESS') THEN 1 ELSE 0 END) AS active_jobs,
                       COUNT(*) AS total_assignments
                FROM assignments
                GROUP BY contributor_id;
            """)
            assignments = {r[0]: {"contributor_id": r[0], "contributor_name": r[1], "contributor_role": r[2], "completed_fixes": r[3], "active_jobs": r[4], "total_assignments": r[5]} for r in cur.fetchall()}

            # 4. Fetch Verified task updates
            cur.execute("""
                SELECT a.contributor_id, COUNT(*) AS verified_count
                FROM verification_records vr
                JOIN assignments a ON vr.report_id = a.report_id
                WHERE vr.action = 'APPROVE'
                GROUP BY a.contributor_id;
            """)
            verified = {r[0]: {"contributor_id": r[0], "verified_count": r[1]} for r in cur.fetchall()}

            # 5. Fetch Reports filed per user
            cur.execute("SELECT user_id, COUNT(*) AS reports_count FROM reports GROUP BY user_id;")
            reports = {r[0]: {"user_id": r[0], "reports_count": r[1]} for r in cur.fetchall()}

            # 6. Fetch Existing ledger baseline
            cur.execute("SELECT * FROM rewards_ledger;")
            ledger = {r["user_id"]: dict(r) for r in cur.fetchall()}

        # Merge all entities
        all_uids = set(users.keys()) | set(funding.keys()) | set(assignments.keys()) | set(ledger.keys())

        leaderboard = []
        for uid in all_uids:
            u = users.get(uid, {})
            f = funding.get(uid, {})
            a = assignments.get(uid, {})
            r = reports.get(uid, {})
            l = ledger.get(uid, {})
            v = verified.get(uid, {})

            name = u.get("name") or a.get("contributor_name") or f.get("donor_name") or l.get("user_name") or uid
            role = u.get("role") or a.get("contributor_role") or ("contractor" if "contractor" in uid else ("engineer" if "engineer" in uid else "citizen"))

            funded = float(f.get("total_funded") or l.get("contributed_amount") or 0.0)
            fixes = max(int(a.get("completed_fixes") or 0), int(v.get("verified_count") or 0))
            reps = int(r.get("reports_count") or 0)
            camps = int(f.get("campaigns_count") or 0)

            base_backed = int(l.get("problems_backed") or 0)
            base_points = int(l.get("points") or 0)

            # Dynamic formula: 1 pt per 10 INR funded + 500 pts per completed fix + 150 pts per reported defect
            calc_points = int(funded / 10.0) + (fixes * 500) + (reps * 150)
            total_points = max(calc_points, base_points)
            total_backed = max(camps + fixes + reps, base_backed)

            tier = "Bronze"
            if total_points >= 3000:
                tier = "Platinum"
            elif total_points >= 1500:
                tier = "Gold"
            elif total_points >= 500:
                tier = "Silver"

            leaderboard.append({
                "user_id": uid,
                "user_name": name,
                "role": role,
                "points": total_points,
                "contributed_amount": funded,
                "problems_backed": total_backed,
                "completed_fixes": fixes,
                "reports_filed": reps,
                "tier": tier
            })

        # Sort descending by points, then contributed_amount, then problems_backed
        leaderboard.sort(key=lambda x: (x["points"], x["contributed_amount"], x["problems_backed"]), reverse=True)

        # Assign ranks
        for idx, item in enumerate(leaderboard, 1):
            item["rank"] = idx

        # Find target user
        user_rewards = None
        if target_user_id:
            for item in leaderboard:
                if item["user_id"] == target_user_id:
                    user_rewards = dict(item)
                    break

        if not user_rewards and target_user_id:
            u_info = users.get(target_user_id, {})
            user_rewards = {
                "user_id": target_user_id,
                "user_name": u_info.get("name", "Active Citizen"),
                "role": u_info.get("role", "citizen"),
                "points": 0,
                "contributed_amount": 0.0,
                "problems_backed": 0,
                "completed_fixes": 0,
                "reports_filed": 0,
                "tier": "Bronze",
                "rank": len(leaderboard) + 1
            }

        return user_rewards, leaderboard[:limit]
    finally:
        conn.close()


def get_user_rewards(user_id):
    """Retrieves dynamically computed rewards details for a user."""
    # Use limit=500 to rank ALL users so the target user's rank is always correct
    user_rewards, _ = compute_dynamic_rewards(target_user_id=user_id, limit=500)
    return user_rewards or {
        "user_id": user_id,
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


def get_leaderboard(limit=10):
    """Returns top dynamically ranked contributors and citizens."""
    _, leaderboard = compute_dynamic_rewards(limit=limit)
    return leaderboard


# =========================================================================
# REPORTS & CIVIC LIFECYCLE MANAGEMENT
# =========================================================================

def add_status_history(report_id, from_status, to_status, actor_name="System", actor_role="system", note="", existing_conn=None, existing_engine=None):
    """Appends an event to the status audit trail."""
    close_when_done = False
    if existing_conn is not None:
        conn = existing_conn
        engine = existing_engine or ("mysql" if hasattr(conn, "cursorclass") else "sqlite")
    else:
        conn, engine = get_connection()
        close_when_done = True
    try:
        hid = f"sh_{uuid.uuid4().hex[:12]}"
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO status_history (id, report_id, from_status, to_status, actor_name, actor_role, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """, (hid, report_id, from_status, to_status, actor_name, actor_role, note))
        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO status_history (id, report_id, from_status, to_status, actor_name, actor_role, note)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (hid, report_id, from_status, to_status, actor_name, actor_role, note))
            conn.commit()
    finally:
        if close_when_done:
            conn.close()


def get_status_history(report_id):
    """Returns the ordered status audit trail for a report."""
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM status_history WHERE report_id = %s ORDER BY created_at ASC;", (report_id,))
                return cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM status_history WHERE report_id = ? ORDER BY created_at ASC;", (report_id,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def create_report(data):
    """
    Creates a new citizen civic defect report, automatically initialises
    a corresponding funding campaign, and logs the initial status event.
    """
    conn, engine = get_connection()
    try:
        rep_id = data.get("id") or f"rep_{uuid.uuid4().hex[:10]}"
        user_id = data.get("user_id", "usr_citizen_1")
        title = data.get("title") or f"{data.get('category', 'Civic Defect').capitalize()} repair"
        category = data.get("category", "potholes")
        loc_text = data.get("location_text") or "Reported Location"
        lat = data.get("latitude")
        lng = data.get("longitude")
        img_path = data.get("image_path", "")
        status = data.get("status", "FUNDING")

        ai_cat = data.get("ai_category", category)
        ai_conf = float(data.get("ai_confidence", 0.95))
        severity = data.get("severity", "Medium")
        sev_score = float(data.get("severity_score", 5.0))
        sev_reason = data.get("severity_reason", "Civic hazard detected by AI vision model.")

        cost_min = float(data.get("estimated_cost_min", 5000.0))
        cost_max = float(data.get("estimated_cost_max", 10000.0))
        work_scope = data.get("work_scope", "Repair and restore civil infrastructure.")
        eta_days = int(data.get("eta_days", 7))

        goal_amount = float(data.get("goal_amount") or round((cost_min + cost_max) / 2))
        if goal_amount <= 0:
            goal_amount = 8000.0

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO reports (
                    id, user_id, title, category, location_text, latitude, longitude,
                    image_path, status, ai_category, ai_confidence, severity,
                    severity_score, severity_reason, estimated_cost_min, estimated_cost_max,
                    work_scope, eta_days
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s
                );
                """, (
                    rep_id, user_id, title, category, loc_text, lat, lng,
                    img_path, status, ai_cat, ai_conf, severity,
                    sev_score, sev_reason, cost_min, cost_max,
                    work_scope, eta_days
                ))

                # Create linked funding campaign
                camp_id = f"fc_{rep_id}"
                cur.execute("""
                INSERT INTO funding_campaigns (
                    id, report_id, goal_amount, raised_amount, released_amount,
                    available_amount, funder_count, status
                ) VALUES (%s, %s, %s, 0.0, 0.0, 0.0, 0, 'ACTIVE')
                ON DUPLICATE KEY UPDATE goal_amount=VALUES(goal_amount);
                """, (camp_id, rep_id, goal_amount))
        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO reports (
                id, user_id, title, category, location_text, latitude, longitude,
                image_path, status, ai_category, ai_confidence, severity,
                severity_score, severity_reason, estimated_cost_min, estimated_cost_max,
                work_scope, eta_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                rep_id, user_id, title, category, loc_text, lat, lng,
                img_path, status, ai_cat, ai_conf, severity,
                sev_score, sev_reason, cost_min, cost_max,
                work_scope, eta_days
            ))

            camp_id = f"fc_{rep_id}"
            cur.execute("""
            INSERT INTO funding_campaigns (
                id, report_id, goal_amount, raised_amount, released_amount,
                available_amount, funder_count, status
            ) VALUES (?, ?, ?, 0.0, 0.0, 0.0, 0, 'ACTIVE')
            ON CONFLICT(report_id) DO UPDATE SET goal_amount=excluded.goal_amount;
            """, (camp_id, rep_id, goal_amount))
            conn.commit()

        # Audit history
        add_status_history(rep_id, None, "REPORT", "Citizen", "citizen", "Problem reported with AI assessment", existing_conn=conn, existing_engine=engine)
        if status != "REPORT":
            add_status_history(rep_id, "REPORT", status, "System", "system", "Funding campaign created", existing_conn=conn, existing_engine=engine)

        return rep_id
    finally:
        conn.close()


def update_report_status(report_id, new_status, actor_name="System", actor_role="system", note="", existing_conn=None, existing_engine=None):
    """Transitions a report to a new status and logs to audit history."""
    close_when_done = False
    if existing_conn is not None:
        conn = existing_conn
        engine = existing_engine or ("mysql" if hasattr(conn, "cursorclass") else "sqlite")
    else:
        conn, engine = get_connection()
        close_when_done = True
    try:
        old_status = None
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM reports WHERE id = %s;", (report_id,))
                res = cur.fetchone()
                if res:
                    old_status = res["status"]
                cur.execute("UPDATE reports SET status = %s WHERE id = %s;", (new_status, report_id))
        else:
            cur = conn.cursor()
            cur.execute("SELECT status FROM reports WHERE id = ?;", (report_id,))
            res = cur.fetchone()
            if res:
                old_status = res[0]
            cur.execute("UPDATE reports SET status = ? WHERE id = ?;", (new_status, report_id))
            conn.commit()

        if old_status != new_status:
            add_status_history(report_id, old_status, new_status, actor_name, actor_role, note, existing_conn=conn, existing_engine=engine)
        return True
    finally:
        if close_when_done:
            conn.close()


def get_all_reports(category=None, status=None, user_id=None):
    """
    Retrieves all reports joined with their live funding campaign,
    assigned contributor, and latest work progress.
    """
    conn, engine = get_connection()
    try:
        sql = """
        SELECT
            r.*,
            fc.goal_amount,
            fc.raised_amount,
            fc.released_amount,
            fc.available_amount,
            fc.funder_count,
            fc.status AS funding_status,
            a.contributor_name,
            a.contributor_role,
            a.status AS assignment_status,
            (SELECT progress_pct FROM task_updates WHERE report_id = r.id ORDER BY submitted_at DESC LIMIT 1) AS latest_progress_pct,
            (SELECT progress_image FROM task_updates WHERE report_id = r.id ORDER BY submitted_at DESC LIMIT 1) AS latest_progress_image
        FROM reports r
        LEFT JOIN funding_campaigns fc ON r.id = fc.report_id
        LEFT JOIN assignments a ON r.id = a.report_id
        WHERE 1=1
        """
        params = []
        if category and category.lower() != "all":
            sql += " AND r.category = " + ("%s" if engine == "mysql" else "?")
            params.append(category.lower())
        if status and status.lower() != "all":
            sql += " AND r.status = " + ("%s" if engine == "mysql" else "?")
            params.append(status.upper())
        if user_id:
            sql += " AND r.user_id = " + ("%s" if engine == "mysql" else "?")
            params.append(user_id)

        sql += " ORDER BY r.created_at DESC;"

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = [dict(r) for r in cur.fetchall()]

        # Calculate percentages and clean up defaults
        for r in rows:
            goal = float(r.get("goal_amount") or 8000.0)
            raised = float(r.get("raised_amount") or 0.0)
            pct = round((raised / goal * 100), 1) if goal > 0 else 0.0
            r["funding_pct"] = min(100.0, pct)
            r["remaining_amount"] = max(0.0, goal - raised)
            r["latest_progress_pct"] = r.get("latest_progress_pct") or 0

        return rows
    finally:
        conn.close()


def get_report_by_id(report_id):
    """
    Retrieves full details for a single report:
      - Report info + AI results
      - Funding campaign + ledger transactions
      - Assigned contributor + assignments
      - Task updates + progress photos
      - Verification records
      - Complete chronological status history
    """
    conn, engine = get_connection()
    try:
        # Base report
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                SELECT r.*,
                       fc.id AS campaign_id, fc.goal_amount, fc.raised_amount,
                       fc.released_amount, fc.available_amount, fc.funder_count,
                       fc.status AS funding_status,
                       a.id AS assignment_id, a.contributor_id, a.contributor_name,
                       a.contributor_role, a.status AS assignment_status
                FROM reports r
                LEFT JOIN funding_campaigns fc ON r.id = fc.report_id
                LEFT JOIN assignments a ON r.id = a.report_id
                WHERE r.id = %s;
                """, (report_id,))
                report = cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("""
            SELECT r.*,
                   fc.id AS campaign_id, fc.goal_amount, fc.raised_amount,
                   fc.released_amount, fc.available_amount, fc.funder_count,
                   fc.status AS funding_status,
                   a.id AS assignment_id, a.contributor_id, a.contributor_name,
                   a.contributor_role, a.status AS assignment_status
            FROM reports r
            LEFT JOIN funding_campaigns fc ON r.id = fc.report_id
            LEFT JOIN assignments a ON r.id = a.report_id
            WHERE r.id = ?;
            """, (report_id,))
            row = cur.fetchone()
            report = dict(row) if row else None

        if not report:
            return None

        # Clean funding numbers
        goal = float(report.get("goal_amount") or 8000.0)
        raised = float(report.get("raised_amount") or 0.0)
        released = float(report.get("released_amount") or 0.0)
        report["goal_amount"] = goal
        report["raised_amount"] = raised
        report["released_amount"] = released
        report["available_amount"] = max(0.0, raised - released)
        report["funding_pct"] = round(min(100.0, (raised / goal * 100)), 1) if goal > 0 else 0.0
        report["remaining_amount"] = max(0.0, goal - raised)

        # Funding transactions
        camp_id = report.get("campaign_id") or f"fc_{report_id}"
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM funding_transactions WHERE campaign_id = %s ORDER BY created_at DESC;", (camp_id,))
                report["transactions"] = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM funding_transactions WHERE campaign_id = ? ORDER BY created_at DESC;", (camp_id,))
            report["transactions"] = [dict(r) for r in cur.fetchall()]

        # Task updates & progress proof
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM task_updates WHERE report_id = %s ORDER BY submitted_at ASC;", (report_id,))
                report["task_updates"] = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM task_updates WHERE report_id = ? ORDER BY submitted_at ASC;", (report_id,))
            report["task_updates"] = [dict(r) for r in cur.fetchall()]

        # Latest progress percentage
        if report["task_updates"]:
            report["latest_progress_pct"] = report["task_updates"][-1]["progress_pct"]
        else:
            report["latest_progress_pct"] = 0

        # Verification records
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM verification_records WHERE report_id = %s ORDER BY created_at DESC;", (report_id,))
                report["verifications"] = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM verification_records WHERE report_id = ? ORDER BY created_at DESC;", (report_id,))
            report["verifications"] = [dict(r) for r in cur.fetchall()]

        # Complete status history timeline
        report["timeline"] = get_status_history(report_id)

        return report
    finally:
        conn.close()


# =========================================================================
# FUNDING SYSTEM & TRANSACTIONS
# =========================================================================

def get_funding_campaign(report_id):
    """Retrieves funding campaign status and ledger for a report."""
    conn, engine = get_connection()
    try:
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM funding_campaigns WHERE report_id = %s;", (report_id,))
                camp = cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM funding_campaigns WHERE report_id = ?;", (report_id,))
            row = cur.fetchone()
            camp = dict(row) if row else None

        if not camp:
            return None

        goal = float(camp.get("goal_amount") or 8000.0)
        raised = float(camp.get("raised_amount") or 0.0)
        released = float(camp.get("released_amount") or 0.0)
        camp["funding_pct"] = round(min(100.0, (raised / goal * 100)), 1) if goal > 0 else 0.0
        camp["remaining_amount"] = max(0.0, goal - raised)
        camp["available_amount"] = max(0.0, raised - released)

        # Get transactions
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM funding_transactions WHERE campaign_id = %s ORDER BY created_at DESC;", (camp["id"],))
                camp["transactions"] = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM funding_transactions WHERE campaign_id = ? ORDER BY created_at DESC;", (camp["id"],))
            camp["transactions"] = [dict(r) for r in cur.fetchall()]

        return camp
    finally:
        conn.close()


def add_funding_contribution(campaign_id, user_id, donor_name, amount, note=""):
    """
    Adds a demo contribution, updates raised and available balances,
    updates funder count, checks if campaign reaches 100%, and updates rewards.
    """
    conn, engine = get_connection()
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Contribution amount must be positive.")

        tx_id = f"tx_{uuid.uuid4().hex[:12]}"

        if engine == "mysql":
            with conn.cursor() as cur:
                # 1. Fetch current campaign
                cur.execute("SELECT * FROM funding_campaigns WHERE id = %s OR report_id = %s;", (campaign_id, campaign_id))
                camp = cur.fetchone()
                if not camp:
                    raise ValueError(f"Campaign not found: {campaign_id}")

                real_camp_id = camp["id"]
                report_id = camp["report_id"]
                goal = float(camp["goal_amount"])
                new_raised = float(camp["raised_amount"]) + amount
                new_released = float(camp["released_amount"])
                new_available = new_raised - new_released
                new_funder_count = int(camp["funder_count"]) + 1

                new_camp_status = "FULLY_FUNDED" if new_raised >= goal else "ACTIVE"

                # 2. Insert transaction
                cur.execute("""
                INSERT INTO funding_transactions (id, campaign_id, user_id, donor_name, amount, transaction_type, note)
                VALUES (%s, %s, %s, %s, %s, 'CONTRIBUTION', %s);
                """, (tx_id, real_camp_id, user_id, donor_name, amount, note))

                # 3. Update campaign totals
                cur.execute("""
                UPDATE funding_campaigns
                SET raised_amount = %s, available_amount = %s, funder_count = %s, status = %s
                WHERE id = %s;
                """, (new_raised, new_available, new_funder_count, new_camp_status, real_camp_id))

                # 4. Check if report reaches FUNDED
                cur.execute("SELECT status FROM reports WHERE id = %s;", (report_id,))
                rep = cur.fetchone()
                rep_status = rep["status"] if rep else "FUNDING"

                if new_raised >= goal and rep_status in ("REPORT", "FUNDING"):
                    cur.execute("UPDATE reports SET status = 'FUNDED' WHERE id = %s;", (report_id,))
                    add_status_history(report_id, rep_status, "FUNDED", donor_name, "donor", f"Campaign fully funded (₹{new_raised:,.0f} / ₹{goal:,.0f})", existing_conn=conn, existing_engine=engine)
                else:
                    add_status_history(report_id, rep_status, rep_status, donor_name, "donor", f"Received ₹{amount:,.0f} contribution", existing_conn=conn, existing_engine=engine)
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM funding_campaigns WHERE id = ? OR report_id = ?;", (campaign_id, campaign_id))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Campaign not found: {campaign_id}")
            camp = dict(row)

            real_camp_id = camp["id"]
            report_id = camp["report_id"]
            goal = float(camp["goal_amount"])
            new_raised = float(camp["raised_amount"]) + amount
            new_released = float(camp["released_amount"])
            new_available = new_raised - new_released
            new_funder_count = int(camp["funder_count"]) + 1

            new_camp_status = "FULLY_FUNDED" if new_raised >= goal else "ACTIVE"

            cur.execute("""
            INSERT INTO funding_transactions (id, campaign_id, user_id, donor_name, amount, transaction_type, note)
            VALUES (?, ?, ?, ?, ?, 'CONTRIBUTION', ?);
            """, (tx_id, real_camp_id, user_id, donor_name, amount, note))

            cur.execute("""
            UPDATE funding_campaigns
            SET raised_amount = ?, available_amount = ?, funder_count = ?, status = ?
            WHERE id = ?;
            """, (new_raised, new_available, new_funder_count, new_camp_status, real_camp_id))

            cur.execute("SELECT status FROM reports WHERE id = ?;", (report_id,))
            rep_row = cur.fetchone()
            rep_status = rep_row[0] if rep_row else "FUNDING"

            if new_raised >= goal and rep_status in ("REPORT", "FUNDING"):
                cur.execute("UPDATE reports SET status = 'FUNDED' WHERE id = ?;", (report_id,))
                add_status_history(report_id, rep_status, "FUNDED", donor_name, "donor", f"Campaign fully funded (₹{new_raised:,.0f} / ₹{goal:,.0f})", existing_conn=conn, existing_engine=engine)
            else:
                add_status_history(report_id, rep_status, rep_status, donor_name, "donor", f"Received ₹{amount:,.0f} contribution", existing_conn=conn, existing_engine=engine)
            conn.commit()
    finally:
        conn.close()

    # Update rewards for donor after connection closed
    if user_id:
        try:
            curr_rewards = get_user_rewards(user_id)
            new_points = curr_rewards["points"] + int(amount / 10)
            new_contributed = curr_rewards["contributed_amount"] + amount
            new_backed = curr_rewards["problems_backed"] + 1
            upsert_rewards(user_id, new_points, new_contributed, new_backed, donor_name)
        except Exception as e:
            print(f"[Rewards] Error updating rewards: {e}")

    return {
        "transaction_id": tx_id,
        "campaign_id": real_camp_id,
        "report_id": report_id,
        "amount": amount,
        "raised_amount": new_raised,
        "goal_amount": goal,
        "funding_pct": round(min(100.0, (new_raised / goal * 100)), 1) if goal > 0 else 0.0,
        "available_amount": new_available,
        "is_fully_funded": new_raised >= goal
    }


# =========================================================================
# CONTRIBUTOR WORKFLOW & TASK EXECUTION
# =========================================================================

def accept_task(report_id, contributor_id, contributor_name, contributor_role):
    """
    Contributor accepts an open civic job.
    Transitions report from FUNDED/REPORT to ASSIGNED.
    """
    conn, engine = get_connection()
    try:
        asgn_id = f"asgn_{uuid.uuid4().hex[:10]}"
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO assignments (id, report_id, contributor_id, contributor_name, contributor_role, status)
                VALUES (%s, %s, %s, %s, %s, 'ASSIGNED')
                ON DUPLICATE KEY UPDATE
                    contributor_id=VALUES(contributor_id),
                    contributor_name=VALUES(contributor_name),
                    contributor_role=VALUES(contributor_role),
                    status='ASSIGNED';
                """, (asgn_id, report_id, contributor_id, contributor_name, contributor_role))
        else:
            cur = conn.cursor()
            # Delete any previous assignment for this report to keep single active assignment
            cur.execute("DELETE FROM assignments WHERE report_id = ?;", (report_id,))
            cur.execute("""
            INSERT INTO assignments (id, report_id, contributor_id, contributor_name, contributor_role, status)
            VALUES (?, ?, ?, ?, ?, 'ASSIGNED');
            """, (asgn_id, report_id, contributor_id, contributor_name, contributor_role))
            conn.commit()

        update_report_status(report_id, "ASSIGNED", contributor_name, contributor_role, f"Accepted by {contributor_name} ({contributor_role})")
        return {"assignment_id": asgn_id, "report_id": report_id, "status": "ASSIGNED"}
    finally:
        conn.close()


def start_task(report_id, contributor_id):
    """
    Contributor marks work as started on site.
    Transitions ASSIGNED -> IN_PROGRESS.
    """
    conn, engine = get_connection()
    try:
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE assignments SET status = 'IN_PROGRESS', started_at = %s
                WHERE report_id = %s;
                """, (now_ts, report_id))
        else:
            cur = conn.cursor()
            cur.execute("""
            UPDATE assignments SET status = 'IN_PROGRESS', started_at = ?
            WHERE report_id = ?;
            """, (now_ts, report_id))
            conn.commit()

        update_report_status(report_id, "IN_PROGRESS", "Contributor", "contractor", "Ground work commenced on site")
        return {"report_id": report_id, "status": "IN_PROGRESS"}
    finally:
        conn.close()


def submit_task_progress(report_id, assignment_id, progress_pct, work_note, progress_image, submit_verification=True):
    """
    Records an on-site progress update with proof photo (stored in uploads/progress/).
    If submitted for verification, transitions status to SUBMITTED_FOR_VERIFICATION.
    """
    conn, engine = get_connection()
    try:
        update_id = f"upd_{uuid.uuid4().hex[:10]}"
        progress_pct = int(progress_pct)

        if not assignment_id:
            # Look up assignment
            if engine == "mysql":
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM assignments WHERE report_id = %s;", (report_id,))
                    row = cur.fetchone()
                    assignment_id = row["id"] if row else f"asgn_{report_id}"
            else:
                cur = conn.cursor()
                cur.execute("SELECT id FROM assignments WHERE report_id = ?;", (report_id,))
                row = cur.fetchone()
                assignment_id = row[0] if row else f"asgn_{report_id}"

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO task_updates (id, assignment_id, report_id, progress_pct, work_note, progress_image)
                VALUES (%s, %s, %s, %s, %s, %s);
                """, (update_id, assignment_id, report_id, progress_pct, work_note, progress_image))
        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO task_updates (id, assignment_id, report_id, progress_pct, work_note, progress_image)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (update_id, assignment_id, report_id, progress_pct, work_note, progress_image))
            conn.commit()

        if submit_verification:
            update_report_status(
                report_id,
                "SUBMITTED_FOR_VERIFICATION",
                "Contributor",
                "contractor",
                f"Work milestone {progress_pct}% reached. Proof uploaded for municipal admin verification."
            )
        else:
            add_status_history(report_id, "IN_PROGRESS", "IN_PROGRESS", "Contributor", "contractor", f"Milestone progress update: {progress_pct}%")

        return {
            "update_id": update_id,
            "report_id": report_id,
            "progress_pct": progress_pct,
            "status": "SUBMITTED_FOR_VERIFICATION" if submit_verification else "IN_PROGRESS"
        }
    finally:
        conn.close()


# =========================================================================
# ADMIN VERIFICATION & FUND RELEASE WORKFLOW
# =========================================================================

def get_admin_verifications(status_filter=None):
    """
    Returns pending verification requests with defect context,
    uploaded proof images, AI scores, and eligible release funds.
    """
    conn, engine = get_connection()
    try:
        sql = """
        SELECT
            tu.id AS update_id,
            tu.assignment_id,
            tu.report_id,
            tu.progress_pct,
            tu.work_note,
            tu.progress_image,
            tu.submitted_at,
            tu.verified,
            tu.verified_at,
            tu.verified_by,
            r.title,
            r.category,
            r.location_text,
            r.severity,
            r.severity_score,
            r.image_path AS original_image,
            r.status AS report_status,
            fc.goal_amount,
            fc.raised_amount,
            fc.released_amount,
            fc.available_amount,
            a.contributor_name,
            a.contributor_role
        FROM task_updates tu
        JOIN reports r ON tu.report_id = r.id
        LEFT JOIN funding_campaigns fc ON r.id = fc.report_id
        LEFT JOIN assignments a ON tu.assignment_id = a.id
        WHERE 1=1
        """
        params = []
        if status_filter == "pending":
            sql += " AND tu.verified = 0 AND r.status = 'SUBMITTED_FOR_VERIFICATION'"
        elif status_filter == "verified":
            sql += " AND tu.verified = 1"

        sql += " ORDER BY tu.submitted_at DESC;"

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = [dict(r) for r in cur.fetchall()]

        for r in rows:
            goal = float(r.get("goal_amount") or 8000.0)
            pct = int(r.get("progress_pct") or 100)
            # Standard milestone release amount (e.g. 25% -> 25% of goal)
            r["suggested_release_amount"] = round((goal * pct / 100.0) - float(r.get("released_amount") or 0.0), 2)
            if r["suggested_release_amount"] < 0:
                r["suggested_release_amount"] = 0.0

        return rows
    finally:
        conn.close()


def admin_approve_verification(update_id, report_id, admin_id="admin_1", notes="Work proof approved.", amount_to_release=0.0):
    """
    Admin verifies work proof.
    If amount_to_release > 0, releases funds to contributor.
    Updates report status to VERIFIED or COMPLETED (if 100%).
    """
    conn, engine = get_connection()
    try:
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vrec_id = f"vr_{uuid.uuid4().hex[:10]}"
        amount_to_release = float(amount_to_release)

        # 1. Check update record to get progress_pct
        pct = 100
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT progress_pct FROM task_updates WHERE id = %s;", (update_id,))
                row = cur.fetchone()
                if row:
                    pct = int(row["progress_pct"])
        else:
            cur = conn.cursor()
            cur.execute("SELECT progress_pct FROM task_updates WHERE id = ?;", (update_id,))
            row = cur.fetchone()
            if row:
                pct = int(row[0])

        # 2. Mark update as verified
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE task_updates
                SET verified = 1, verified_at = %s, verified_by = %s
                WHERE id = %s;
                """, (now_ts, admin_id, update_id))

                cur.execute("""
                INSERT INTO verification_records (id, task_update_id, report_id, admin_id, action, notes, amount_released)
                VALUES (%s, %s, %s, %s, 'APPROVE', %s, %s);
                """, (vrec_id, update_id, report_id, admin_id, notes, amount_to_release))
        else:
            cur = conn.cursor()
            cur.execute("""
            UPDATE task_updates
            SET verified = 1, verified_at = ?, verified_by = ?
            WHERE id = ?;
            """, (now_ts, admin_id, update_id))

            cur.execute("""
            INSERT INTO verification_records (id, task_update_id, report_id, admin_id, action, notes, amount_released)
            VALUES (?, ?, ?, ?, 'APPROVE', ?, ?);
            """, (vrec_id, update_id, report_id, admin_id, notes, amount_to_release))
            conn.commit()

        # 3. Disburse funds if specified
        if amount_to_release > 0:
            if engine == "mysql":
                with conn.cursor() as cur:
                    cur.execute("""
                    UPDATE funding_campaigns
                    SET released_amount = released_amount + %s,
                        available_amount = available_amount - %s
                    WHERE report_id = %s;
                    """, (amount_to_release, amount_to_release, report_id))
            else:
                cur = conn.cursor()
                cur.execute("""
                UPDATE funding_campaigns
                SET released_amount = released_amount + ?,
                    available_amount = available_amount - ?
                WHERE report_id = ?;
                """, (amount_to_release, amount_to_release, report_id))
                conn.commit()

        # 4. Determine final report status
        new_status = "COMPLETED" if pct >= 100 else "VERIFIED"
        status_note = f"Admin verified {pct}% progress proof. Disbursed ₹{amount_to_release:,.0f}."
        if new_status == "COMPLETED":
            status_note += " Defect repair successfully finished and closed."
            # Also mark assignment as completed
            if engine == "mysql":
                with conn.cursor() as cur:
                    cur.execute("UPDATE assignments SET status = 'COMPLETED', completed_at = %s WHERE report_id = %s;", (now_ts, report_id))
            else:
                cur = conn.cursor()
                cur.execute("UPDATE assignments SET status = 'COMPLETED', completed_at = ? WHERE report_id = ?;", (now_ts, report_id))
                conn.commit()

        update_report_status(report_id, new_status, "City Engineer Admin", "admin", status_note)

        return {
            "success": True,
            "verification_id": vrec_id,
            "report_id": report_id,
            "status": new_status,
            "amount_released": amount_to_release
        }
    finally:
        conn.close()


def admin_rework_verification(update_id, report_id, admin_id="admin_1", notes="Work proof does not meet civic quality standards."):
    """Admin rejects work proof and flags report for rework with contractor."""
    conn, engine = get_connection()
    try:
        vrec_id = f"vr_{uuid.uuid4().hex[:10]}"
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO verification_records (id, task_update_id, report_id, admin_id, action, notes, amount_released)
                VALUES (%s, %s, %s, %s, 'REWORK', %s, 0.0);
                """, (vrec_id, update_id, report_id, admin_id, notes))
        else:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO verification_records (id, task_update_id, report_id, admin_id, action, notes, amount_released)
            VALUES (?, ?, ?, ?, 'REWORK', ?, 0.0);
            """, (vrec_id, update_id, report_id, admin_id, notes))
            conn.commit()

        update_report_status(report_id, "REWORK", "City Engineer Admin", "admin", f"Quality verification failed: {notes}")
        return {"success": True, "verification_id": vrec_id, "status": "REWORK"}
    finally:
        conn.close()


def admin_release_funds(report_id, amount, admin_id="admin_1", notes="Manual fund disbursement"):
    """Releases available campaign funds directly to contributor."""
    conn, engine = get_connection()
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")

        vrec_id = f"vr_{uuid.uuid4().hex[:10]}"
        dummy_upd_id = f"manual_{report_id}"

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("""
                UPDATE funding_campaigns
                SET released_amount = released_amount + %s,
                    available_amount = available_amount - %s
                WHERE report_id = %s;
                """, (amount, amount, report_id))

                cur.execute("""
                INSERT INTO verification_records (id, task_update_id, report_id, admin_id, action, notes, amount_released)
                VALUES (%s, %s, %s, %s, 'APPROVE', %s, %s);
                """, (vrec_id, dummy_upd_id, report_id, admin_id, notes, amount))
        else:
            cur = conn.cursor()
            cur.execute("""
            UPDATE funding_campaigns
            SET released_amount = released_amount + ?,
                available_amount = available_amount - ?
            WHERE report_id = ?;
            """, (amount, amount, report_id))

            cur.execute("""
            INSERT INTO verification_records (id, task_update_id, report_id, admin_id, action, notes, amount_released)
            VALUES (?, ?, ?, ?, 'APPROVE', ?, ?);
            """, (vrec_id, dummy_upd_id, report_id, admin_id, notes, amount))
            conn.commit()

        add_status_history(report_id, None, "FUNDS_RELEASED", "City Engineer Admin", "admin", f"Released ₹{amount:,.0f} from campaign funds")
        return {"success": True, "amount_released": amount}
    finally:
        conn.close()


def get_admin_dashboard_stats():
    """Aggregates high-level city KPI metrics for the Municipal Admin Console."""
    conn, engine = get_connection()
    try:
        stats = {
            "total_reports": 0,
            "funding_reports": 0,
            "funded_reports": 0,
            "assigned_jobs": 0,
            "in_progress_jobs": 0,
            "pending_verification": 0,
            "completed_reports": 0,
            "total_funds_raised": 0.0,
            "total_funds_released": 0.0,
            "total_contributors": get_contributor_count()
        }

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("SELECT status, COUNT(*) AS count FROM reports GROUP BY status;")
                for r in cur.fetchall():
                    st = r["status"]
                    cnt = r["count"]
                    stats["total_reports"] += cnt
                    if st in ("REPORT", "FUNDING"):
                        stats["funding_reports"] += cnt
                    elif st == "FUNDED":
                        stats["funded_reports"] += cnt
                    elif st == "ASSIGNED":
                        stats["assigned_jobs"] += cnt
                    elif st == "IN_PROGRESS":
                        stats["in_progress_jobs"] += cnt
                    elif st == "SUBMITTED_FOR_VERIFICATION":
                        stats["pending_verification"] += cnt
                    elif st in ("VERIFIED", "COMPLETED"):
                        stats["completed_reports"] += cnt

                cur.execute("SELECT SUM(raised_amount) AS total_raised, SUM(released_amount) AS total_released FROM funding_campaigns;")
                f_res = cur.fetchone()
                if f_res:
                    stats["total_funds_raised"] = float(f_res["total_raised"] or 0.0)
                    stats["total_funds_released"] = float(f_res["total_released"] or 0.0)
        else:
            cur = conn.cursor()
            cur.execute("SELECT status, COUNT(*) AS count FROM reports GROUP BY status;")
            for r in cur.fetchall():
                st = r[0]
                cnt = r[1]
                stats["total_reports"] += cnt
                if st in ("REPORT", "FUNDING"):
                    stats["funding_reports"] += cnt
                elif st == "FUNDED":
                    stats["funded_reports"] += cnt
                elif st == "ASSIGNED":
                    stats["assigned_jobs"] += cnt
                elif st == "IN_PROGRESS":
                    stats["in_progress_jobs"] += cnt
                elif st == "SUBMITTED_FOR_VERIFICATION":
                    stats["pending_verification"] += cnt
                elif st in ("VERIFIED", "COMPLETED"):
                    stats["completed_reports"] += cnt

            cur.execute("SELECT SUM(raised_amount), SUM(released_amount) FROM funding_campaigns;")
            f_res = cur.fetchone()
            if f_res:
                stats["total_funds_raised"] = float(f_res[0] or 0.0)
                stats["total_funds_released"] = float(f_res[1] or 0.0)

        return stats
    finally:
        conn.close()


def get_global_status_history(limit=30):
    """
    Returns recent status transitions from status_history joined with report titles.
    """
    conn, db_type = get_connection()
    try:
        if db_type == "sqlite":
            cur = conn.execute(
                """
                SELECT sh.id, sh.report_id, sh.from_status, sh.to_status,
                       sh.actor_name, sh.actor_role, sh.note, sh.created_at,
                       r.title AS report_title, r.location_text
                FROM status_history sh
                LEFT JOIN reports r ON r.id = sh.report_id
                ORDER BY sh.created_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = [dict(r) for r in cur.fetchall()]
        else:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT sh.id, sh.report_id, sh.from_status, sh.to_status,
                           sh.actor_name, sh.actor_role, sh.note, sh.created_at,
                           r.title AS report_title, r.location_text
                    FROM status_history sh
                    LEFT JOIN reports r ON r.id = sh.report_id
                    ORDER BY sh.created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                rows = cur.fetchall()
        return rows
    finally:
        conn.close()


def delete_report(report_id):
    """
    Permanently deletes a report and all its linked civic lifecycle records:
      - funding transactions & campaign
      - verification records
      - task progress updates
      - contributor assignments
      - status audit history
      - the base report record
    Does NOT delete master contributor profiles.
    """
    conn, engine = get_connection()
    try:
        camp_id = f"fc_{report_id}"
        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute("DELETE FROM funding_transactions WHERE campaign_id = %s OR campaign_id = %s;", (camp_id, report_id))
                cur.execute("DELETE FROM funding_campaigns WHERE report_id = %s OR id = %s;", (report_id, camp_id))
                cur.execute("DELETE FROM verification_records WHERE report_id = %s;", (report_id,))
                cur.execute("DELETE FROM task_updates WHERE report_id = %s;", (report_id,))
                cur.execute("DELETE FROM assignments WHERE report_id = %s;", (report_id,))
                cur.execute("DELETE FROM status_history WHERE report_id = %s;", (report_id,))
                cur.execute("DELETE FROM reports WHERE id = %s;", (report_id,))
                row_cnt = cur.rowcount
            conn.commit()
        else:
            cur = conn.cursor()
            cur.execute("DELETE FROM funding_transactions WHERE campaign_id = ? OR campaign_id = ?;", (camp_id, report_id))
            cur.execute("DELETE FROM funding_campaigns WHERE report_id = ? OR id = ?;", (report_id, camp_id))
            cur.execute("DELETE FROM verification_records WHERE report_id = ?;", (report_id,))
            cur.execute("DELETE FROM task_updates WHERE report_id = ?;", (report_id,))
            cur.execute("DELETE FROM assignments WHERE report_id = ?;", (report_id,))
            cur.execute("DELETE FROM status_history WHERE report_id = ?;", (report_id,))
            cur.execute("DELETE FROM reports WHERE id = ?;", (report_id,))
            row_cnt = cur.rowcount
            conn.commit()
        return row_cnt > 0
    finally:
        conn.close()


def get_contributor_payments(role=None, contributor_id=None):
    """
    Retrieves all verified milestone fund disbursements credited to contributors.
    Joined with report details and assignments.
    """
    conn, engine = get_connection()
    try:
        sql = """
        SELECT
            vr.id,
            vr.report_id,
            vr.action,
            vr.amount_released,
            vr.notes,
            vr.created_at,
            r.title AS report_title,
            r.category AS report_category,
            r.location_text,
            a.contributor_id,
            a.contributor_name,
            a.contributor_role
        FROM verification_records vr
        JOIN reports r ON vr.report_id = r.id
        LEFT JOIN assignments a ON vr.report_id = a.report_id
        WHERE vr.action = 'APPROVE' AND vr.amount_released > 0
        """
        params = []
        if role and role.lower() != "all":
            sql += " AND (a.contributor_role = " + ("%s" if engine == "mysql" else "?") + " OR a.contributor_role IS NULL)"
            params.append(role.lower())
        if contributor_id:
            sql += " AND a.contributor_id = " + ("%s" if engine == "mysql" else "?")
            params.append(contributor_id)

        sql += " ORDER BY vr.created_at DESC;"

        if engine == "mysql":
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = [dict(r) for r in cur.fetchall()]

        total_earned = sum(float(r.get("amount_released") or 0) for r in rows)
        return {
            "payments": rows,
            "total_earned": total_earned,
            "count": len(rows)
        }
    finally:
        conn.close()

