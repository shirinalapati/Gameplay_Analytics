"""
Shared configuration, paths, database helpers, and NHL API utilities.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SQL_DIR = PROJECT_ROOT / "sql"
DEFAULT_DB_PATH = DATA_PROCESSED / "nhl_playback.db"

# --- Season scope (2025–2026 NHL regular season) ---
SEASON_NUMERIC = 20252026
SEASON_LABEL = "2025-2026"
# NHL Web API (api-web.nhle.com): 1 = preseason, 2 = regular season, 3 = playoffs.
# (Do not use 1 here — that only pulls preseason and explains a Sept–Oct-heavy DB.)
GAME_TYPE_REGULAR = 2

# Schedule `gameState`: completed games are usually OFF (official); FINAL appears in some feeds.
SCHEDULE_COMPLETED_STATES = frozenset({"FINAL", "OFF"})

# --- NHL public web API ---
NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date}"
NHL_PLAY_BY_PLAY_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

# Full 2025–2026 regular season window (schedule scan; early Oct + late April included)
REGULAR_SEASON_START = "2025-09-24"
REGULAR_SEASON_END = "2026-04-22"

# League schedule: 32 teams × 82 GP / 2 = unique regular-season games
EXPECTED_REGULAR_SEASON_GAME_TOTAL = 1312

REQUEST_TIMEOUT_SEC = 30


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Return a SQLite connection with row factory for dict-like rows.

    ``check_same_thread=False`` is required for Streamlit: reruns and widgets can
    execute on different threads than the one that opened the connection. The app
    treats the DB as read-mostly during a session; ingest/clean still run single-threaded.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection, schema_path: Optional[Path] = None) -> None:
    """Apply sql/schema.sql to create tables and indexes."""
    sp = schema_path or (SQL_DIR / "schema.sql")
    sql_text = sp.read_text(encoding="utf-8")
    conn.executescript(sql_text)
    conn.commit()


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_time_in_period(time_str: str) -> int:
    """Parse MM:SS clock into seconds elapsed within the period."""
    if not time_str:
        return 0
    parts = str(time_str).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def game_seconds(
    period: int,
    time_in_period: str,
    max_regulation_periods: int = 3,
    ot_period_seconds: int = 300,
) -> float:
    """
    Monotonic game clock in seconds from puck drop (period 1, 0:00).
    Regulation periods are 20 minutes (1200s). OT periods are 5 minutes (300s) each.
    """
    sec_in_period = parse_time_in_period(time_in_period)
    if period <= max_regulation_periods:
        return (period - 1) * 1200.0 + float(sec_in_period)
    # Overtime / shootout frames beyond regulation
    base_reg = max_regulation_periods * 1200.0
    ot_index = period - max_regulation_periods - 1
    return base_reg + ot_index * float(ot_period_seconds) + float(sec_in_period)


def format_situation(code: str) -> str:
    """
    Convert NHL 4-digit situation code to readable manpower label.

    Structure: [away_goalie][away_skaters][home_skaters][home_goalie]
    """
    raw = str(code or "").strip()
    digit_str = "".join(ch for ch in raw if ch.isdigit())
    if not digit_str:
        return "unknown"
    digits = digit_str[-4:].zfill(4)
    away_goalie = int(digits[0])
    away_skaters = int(digits[1])
    home_skaters = int(digits[2])
    home_goalie = int(digits[3])

    base = f"{away_skaters}v{home_skaters}"
    labels: list[str] = []

    # Empty-net context (preferred over PP/PK tags for concise output).
    if away_goalie == 0 and home_goalie == 0:
        labels.append("both empty net")
    elif away_goalie == 0:
        labels.append("away empty net")
    elif home_goalie == 0:
        labels.append("home empty net")
    else:
        if away_skaters == home_skaters:
            if away_skaters == 5:
                labels.append("even strength")
            elif away_skaters == 4:
                labels.append("4-on-4")
            elif away_skaters == 3:
                labels.append("3-on-3")
            else:
                labels.append(f"{away_skaters}-on-{home_skaters}")
        else:
            away_adv = away_skaters > home_skaters
            adv_side = "away" if away_adv else "home"
            high = max(away_skaters, home_skaters)
            low = min(away_skaters, home_skaters)
            if high == 5 and low == 3:
                labels.append(f"{adv_side} 5-on-3")
            elif high - low == 1:
                labels.append(f"{adv_side} power play")
            else:
                labels.append(f"{adv_side} skater advantage")

    return f"{base} ({', '.join(labels)})" if labels else base


def strength_hint(situation_code: Optional[str]) -> str:
    """Backward-compatible alias for normalized strength labels."""
    if situation_code is None:
        return "unknown"
    return format_situation(str(situation_code))


def sort_plays(plays: list[dict]) -> list[dict]:
    """Stable chronological ordering within the official play-by-play feed."""

    def key(p: dict) -> tuple:
        pd = p.get("periodDescriptor") or {}
        period = int(pd.get("number", 0))
        tip = p.get("timeInPeriod") or "0:00"
        sec = parse_time_in_period(tip)
        so = int(p.get("sortOrder", 0))
        eid = int(p.get("eventId", 0))
        return period, sec, so, eid

    return sorted(plays, key=key)
