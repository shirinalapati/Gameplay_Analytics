"""
Playback helpers: load normalized game data from SQLite and advance logical time.

The Streamlit UI drives timing (sleep + rerun); this module supplies data slices
for the current event index and clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import sqlite3

from src.utils import DEFAULT_DB_PATH, GAME_TYPE_REGULAR, get_connection


SHOTS_QUERY = """
SELECT
    s.*,
    g.home_abbr,
    g.away_abbr,
    g.home_team_id,
    g.away_team_id,
    CASE
        WHEN s.team_id = g.home_team_id THEN g.home_abbr
        WHEN s.team_id = g.away_team_id THEN g.away_abbr
        ELSE NULL
    END AS team_abbr
FROM shots s
JOIN games g ON g.game_id = s.game_id
WHERE s.game_id = ?
ORDER BY s.event_idx
"""

EVENTS_QUERY = """
SELECT e.*, g.home_abbr, g.away_abbr
FROM events e
JOIN games g ON g.game_id = e.game_id
WHERE e.game_id = ?
ORDER BY e.event_idx
"""

GAME_META_QUERY = """
SELECT * FROM games WHERE game_id = ?
"""


def load_game_meta(conn: sqlite3.Connection, game_id: int) -> pd.DataFrame:
    return pd.read_sql(GAME_META_QUERY, conn, params=(game_id,))


def load_events(conn: sqlite3.Connection, game_id: int) -> pd.DataFrame:
    return pd.read_sql(EVENTS_QUERY, conn, params=(game_id,))


def load_shots(conn: sqlite3.Connection, game_id: int) -> pd.DataFrame:
    return pd.read_sql(SHOTS_QUERY, conn, params=(game_id,))


def list_games(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT game_id, game_date, home_abbr, away_abbr, home_score, away_score, season
        FROM games
        WHERE season = 20252026 AND game_type = ?
        ORDER BY game_date DESC, game_id DESC
        """,
        conn,
        params=(GAME_TYPE_REGULAR,),
    )


@dataclass
class PlaybackState:
    """Mutable playback cursor (updated by Streamlit session_state)."""

    game_id: int
    event_idx: int = 0
    playing: bool = False
    speed: float = 1.0

    def clamp(self, max_idx: int) -> None:
        self.event_idx = int(max(0, min(self.event_idx, max_idx)))


def slice_at_event(
    events: pd.DataFrame,
    shots: pd.DataFrame,
    event_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
    """
    Return (events_up_to, shots_up_to, current_event_row).
    event_idx is 0-based index into chronologically ordered `events`.
    """
    if events.empty:
        return events, shots, None
    idx = int(max(0, min(event_idx, len(events) - 1)))
    sub_e = events.iloc[: idx + 1].copy()
    last = events.iloc[idx]
    t = float(last["game_seconds"])
    sub_s = shots[shots["game_seconds"] <= t].copy() if not shots.empty else shots
    return sub_e, sub_s, last


def current_time(events: pd.DataFrame, event_idx: int) -> float:
    if events.empty:
        return 0.0
    i = int(max(0, min(event_idx, len(events) - 1)))
    return float(events.iloc[i]["game_seconds"])
