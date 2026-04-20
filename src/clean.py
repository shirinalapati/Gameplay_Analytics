"""
Normalize raw NHL play-by-play JSON into SQLite tables (`games`, `events`, `shots`).

Shot quality and high-danger flags are computed here via `metrics.shot_quality_index`.

Run (from project root) to rebuild from cached raw JSON:

    python -m src.clean
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from src.metrics import is_high_danger, shot_quality_index
from src.utils import (
    DATA_RAW,
    DEFAULT_DB_PATH,
    GAME_TYPE_REGULAR,
    SEASON_NUMERIC,
    game_seconds,
    get_connection,
    init_schema,
    safe_float,
    safe_int,
    sort_plays,
    strength_hint,
)


SHOT_KEYS = {"shot-on-goal", "missed-shot", "goal", "blocked-shot"}


def primary_player_id(details: Optional[dict], type_key: str) -> Optional[int]:
    if not details:
        return None
    if type_key == "goal":
        return safe_int(details.get("scoringPlayerId"))
    if type_key in ("shot-on-goal", "missed-shot", "blocked-shot"):
        return safe_int(details.get("shootingPlayerId"))
    return safe_int(details.get("playerId"))


def upsert_game_from_pbp_dict(
    conn: sqlite3.Connection,
    pbp: dict,
    raw_path: Optional[str] = None,
) -> None:
    """Parse a single play-by-play payload and write to SQLite."""
    gid = int(pbp["id"])
    season = int(pbp.get("season", SEASON_NUMERIC))
    gdate = str(pbp.get("gameDate", ""))
    gtype = int(pbp.get("gameType", GAME_TYPE_REGULAR))
    state = str(pbp.get("gameState", ""))
    start_utc = str(pbp.get("startTimeUTC", ""))

    home = pbp.get("homeTeam") or {}
    away = pbp.get("awayTeam") or {}
    hid = safe_int(home.get("id"))
    aid = safe_int(away.get("id"))
    habbr = str(home.get("abbrev") or "")
    aabbr = str(away.get("abbrev") or "")

    venue = str((pbp.get("venue") or {}).get("default") or "")

    conn.execute("DELETE FROM shots WHERE game_id = ?", (gid,))
    conn.execute("DELETE FROM events WHERE game_id = ?", (gid,))
    conn.execute("DELETE FROM games WHERE game_id = ?", (gid,))

    conn.execute(
        """
        INSERT INTO games (
            game_id, season, game_type, game_date, game_state, start_time_utc,
            home_team_id, away_team_id, home_abbr, away_abbr,
            home_score, away_score, venue, raw_json_path
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            gid,
            season,
            gtype,
            gdate,
            state,
            start_utc,
            hid,
            aid,
            habbr,
            aabbr,
            int(home.get("score") or 0),
            int(away.get("score") or 0),
            venue,
            raw_path,
        ),
    )

    plays = sort_plays(list(pbp.get("plays") or []))

    def team_abbr_for(tid: Optional[int]) -> Optional[str]:
        if tid is None:
            return None
        if hid is not None and tid == hid:
            return habbr
        if aid is not None and tid == aid:
            return aabbr
        return None

    last_home, last_away = 0, 0
    event_rows = []
    shot_rows = []

    for idx, play in enumerate(plays):
        pd = play.get("periodDescriptor") or {}
        period = int(pd.get("number", 1))
        tip = play.get("timeInPeriod") or "0:00"
        tsec = game_seconds(period, tip)
        type_key = str(play.get("typeDescKey") or "")
        details = play.get("details") or {}
        sort_order = safe_int(play.get("sortOrder"))
        situation = str(play.get("situationCode") or "")
        strength = strength_hint(situation)

        tid = safe_int(details.get("eventOwnerTeamId"))
        player_id = primary_player_id(details, type_key)
        x = safe_float(details.get("xCoord"))
        y = safe_float(details.get("yCoord"))
        zone = str(details.get("zoneCode") or "")
        shot_type = str(details.get("shotType") or "") or None

        if type_key == "goal":
            last_home = int(details.get("homeScore") or last_home)
            last_away = int(details.get("awayScore") or last_away)
        sh = last_home
        sa = last_away

        is_goal = 1 if type_key == "goal" else 0
        is_shot_attempt = 1 if type_key in SHOT_KEYS else 0

        event_rows.append(
            (
                gid,
                idx,
                sort_order,
                period,
                tsec,
                type_key,
                type_key,
                tid,
                player_id,
                x,
                y,
                zone,
                situation,
                strength,
                sh,
                sa,
                is_goal,
                is_shot_attempt,
                shot_type,
            )
        )

        if type_key in SHOT_KEYS:
            blocked = 1 if type_key == "blocked-shot" else 0
            base_q = shot_quality_index(x, y)
            q = base_q * (0.22 if blocked else 1.0)
            hd = 1 if (not blocked) and is_high_danger(x, y) else 0
            shot_rows.append(
                (
                    gid,
                    idx,
                    tid,
                    player_id,
                    period,
                    tsec,
                    type_key,
                    shot_type,
                    x,
                    y,
                    zone,
                    situation,
                    strength,
                    sh,
                    sa,
                    is_goal,
                    blocked,
                    q,
                    hd,
                )
            )

    conn.executemany(
        """
        INSERT INTO events (
            game_id, event_idx, sort_order, period, game_seconds, event_type, type_desc_key,
            team_id, player_id, x_coord, y_coord, zone_code, situation_code, strength_state,
            score_home, score_away, is_goal, is_shot_attempt, shot_type
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        event_rows,
    )

    conn.executemany(
        """
        INSERT INTO shots (
            game_id, event_idx, team_id, player_id, period, game_seconds,
            shot_kind, shot_type, x_coord, y_coord, zone_code, situation_code, strength_state,
            score_home, score_away, is_goal, is_blocked, shot_quality, high_danger
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        shot_rows,
    )


def process_all_raw(conn: sqlite3.Connection, raw_dir: Path) -> int:
    """Load every `*_play_by_play.json` under raw_dir."""
    n = 0
    for path in sorted(raw_dir.glob("*_play_by_play.json")):
        with path.open(encoding="utf-8") as fh:
            pbp = json.load(fh)
        upsert_game_from_pbp_dict(conn, pbp, raw_path=str(path))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild SQLite from raw play-by-play JSON.")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--raw-dir", default=str(DATA_RAW))
    args = ap.parse_args()

    conn = get_connection(Path(args.db))
    init_schema(conn)
    count = process_all_raw(conn, Path(args.raw_dir))
    conn.commit()
    conn.close()
    print(f"Processed {count} games from {args.raw_dir}")


if __name__ == "__main__":
    main()
