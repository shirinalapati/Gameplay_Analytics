"""
Download 2025–2026 NHL regular-season schedules and play-by-play JSON from the
public NHL web API (`api-web.nhle.com`) and persist raw payloads under /data/raw.
Finished games use schedule `gameState` **OFF** (official) or **FINAL**, not only FINAL.

Run (from project root):

    python -m src.ingest --start 2025-09-24 --end 2026-04-22

Use `--limit-games N` for a quicker smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import requests

from src.clean import upsert_game_from_pbp_dict
from src.utils import (
    DATA_RAW,
    DEFAULT_DB_PATH,
    EXPECTED_REGULAR_SEASON_GAME_TOTAL,
    GAME_TYPE_REGULAR,
    NHL_PLAY_BY_PLAY_URL,
    NHL_SCHEDULE_URL,
    REGULAR_SEASON_END,
    REGULAR_SEASON_START,
    REQUEST_TIMEOUT_SEC,
    SCHEDULE_COMPLETED_STATES,
    SEASON_NUMERIC,
    get_connection,
    init_schema,
)


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_json(url: str, max_retries: int = 8) -> dict:
    """GET with backoff on rate limits / transient NHL gateway errors."""
    backoff = 1.2
    last: Optional[requests.Response] = None
    for attempt in range(max_retries):
        r = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
        last = r
        if r.status_code in (429, 502, 503, 504) and attempt < max_retries - 1:
            wait = min(60.0, backoff)
            time.sleep(wait)
            backoff = min(backoff * 1.65, 60.0)
            continue
        r.raise_for_status()
        return r.json()
    if last is not None:
        last.raise_for_status()
    raise requests.RequestException("fetch_json: no response")


def iter_schedule_games(schedule_payload: dict) -> list[dict]:
    games: list[dict] = []
    for day in schedule_payload.get("gameWeek", []) or []:
        for g in day.get("games", []) or []:
            games.append(g)
    return games


def ingest(
    start: str,
    end: str,
    db_path: Optional[Path] = None,
    sleep_sec: float = 0.35,
    limit_games: Optional[int] = None,
    skip_existing: bool = True,
) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    init_schema(conn)

    sdt = datetime.strptime(start, "%Y-%m-%d").date()
    edt = datetime.strptime(end, "%Y-%m-%d").date()
    processed = 0

    for d in daterange(sdt, edt):
        ds = d.isoformat()
        url = NHL_SCHEDULE_URL.format(date=ds)
        try:
            sched = fetch_json(url)
        except requests.RequestException as e:
            print(f"[warn] schedule {ds}: {e}")
            time.sleep(sleep_sec)
            continue

        for g in iter_schedule_games(sched):
            gid = g.get("id")
            try:
                gtype = int(g.get("gameType"))
            except (TypeError, ValueError):
                continue
            state = (g.get("gameState") or "").upper()
            try:
                season = int(g.get("season"))
            except (TypeError, ValueError):
                continue
            if season != SEASON_NUMERIC:
                continue
            if gtype != GAME_TYPE_REGULAR:
                continue
            # api-web.nhle.com marks finished RS games as OFF (not FINAL) once the sheet is official.
            if state not in SCHEDULE_COMPLETED_STATES:
                continue

            raw_path = DATA_RAW / f"{gid}_play_by_play.json"
            if skip_existing and raw_path.exists():
                # Still ensure DB has it
                try:
                    with raw_path.open(encoding="utf-8") as fh:
                        pbp = json.load(fh)
                    upsert_game_from_pbp_dict(conn, pbp, raw_path=str(raw_path))
                except Exception as e:
                    print(f"[warn] reprocess {gid}: {e}")
                continue

            pbp_url = NHL_PLAY_BY_PLAY_URL.format(game_id=gid)
            try:
                pbp = fetch_json(pbp_url)
            except requests.RequestException as e:
                print(f"[warn] pbp {gid}: {e}")
                time.sleep(sleep_sec)
                continue

            with raw_path.open("w", encoding="utf-8") as fh:
                json.dump(pbp, fh)

            try:
                upsert_game_from_pbp_dict(conn, pbp, raw_path=str(raw_path))
            except Exception as e:
                print(f"[error] clean {gid}: {e}")
                conn.rollback()
                continue

            processed += 1
            if limit_games is not None and processed >= limit_games:
                conn.commit()
                conn.close()
                print(f"Stopped after {processed} newly downloaded games (limit).")
                _print_regular_season_db_count(db_path)
                return

            time.sleep(sleep_sec)

    conn.commit()
    conn.close()
    print(f"Done. Newly downloaded & cleaned games this run: {processed}")
    _print_regular_season_db_count(db_path)


def _print_regular_season_db_count(db_path: Path) -> None:
    """Report how many 2025–26 regular-season games are in SQLite vs league schedule."""
    try:
        cx = get_connection(db_path)
        row = cx.execute(
            "SELECT COUNT(*) FROM games WHERE season = ? AND game_type = ?",
            (SEASON_NUMERIC, GAME_TYPE_REGULAR),
        ).fetchone()
        cx.close()
        n = int(row[0]) if row else 0
        print(
            f"Database: {n} regular-season games "
            f"(expected {EXPECTED_REGULAR_SEASON_GAME_TOTAL} after a full season; ingest skips non-finished schedule rows)."
        )
    except Exception as e:
        print(f"[warn] could not count games in DB: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest NHL 2025–26 regular-season play-by-play.")
    ap.add_argument("--start", default=REGULAR_SEASON_START)
    ap.add_argument("--end", default=REGULAR_SEASON_END)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--sleep", type=float, default=0.35, help="Polite delay between HTTP calls.")
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    ingest(
        args.start,
        args.end,
        db_path=Path(args.db),
        sleep_sec=args.sleep,
        limit_games=args.limit_games,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
