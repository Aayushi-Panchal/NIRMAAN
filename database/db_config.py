"""
Database configuration for the NIRMAAN Contributor Matching System.
Reads connection parameters from environment variables with sensible defaults.
Supports .env file loading if python-dotenv is present.
"""
import os

# Load .env file if present
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Database Connection Settings
MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "nirmaan_db")

# Matching Weights Configuration (Normalized 0.0 - 1.0)
MATCH_WEIGHTS = {
    "category_skill": 0.30,
    "distance": 0.20,
    "budget": 0.15,
    "severity": 0.10,
    "past_work": 0.10,
    "reliability": 0.10,
    "availability": 0.05
}

# Verification Status Multipliers
VERIFICATION_FACTORS = {
    "verified": 1.00,
    "pending": 0.85,
    "unverified": 0.70
}

# Availability Scores
AVAILABILITY_SCORES = {
    "available": 100.0,
    "limited": 65.0,
    "busy": 30.0,
    "unavailable": 0.0
}

# Severity Rank Mapping
SEVERITY_LEVELS = {
    "low": 1,
    "medium": 2,
    "high": 3
}
