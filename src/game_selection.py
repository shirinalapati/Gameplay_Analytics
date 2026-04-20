"""
Helpers for multi-step game selection (filters + labels + table rows).

Used by the Streamlit app so users never face a single 1,312-option dropdown.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Optional

import pandas as pd

# Display names for pick labels (Away @ Home). Falls back to abbreviation if missing.
NHL_TEAM_DISPLAY: dict[str, str] = {
    "ANA": "Ducks",
    "ARI": "Coyotes",
    "BOS": "Bruins",
    "BUF": "Sabres",
    "CAR": "Hurricanes",
    "CBJ": "Blue Jackets",
    "CGY": "Flames",
    "CHI": "Blackhawks",
    "COL": "Avalanche",
    "DAL": "Stars",
    "DET": "Red Wings",
    "EDM": "Oilers",
    "FLA": "Panthers",
    "LAK": "Kings",
    "MIN": "Wild",
    "MTL": "Canadiens",
    "NJD": "Devils",
    "NSH": "Predators",
    "NYI": "Islanders",
    "NYR": "Rangers",
    "OTT": "Senators",
    "PHI": "Flyers",
    "PIT": "Penguins",
    "SEA": "Kraken",
    "SJS": "Sharks",
    "STL": "Blues",
    "TBL": "Lightning",
    "TOR": "Maple Leafs",
    "UTA": "Utah",
    "VAN": "Canucks",
    "VGK": "Golden Knights",
    "WPG": "Jets",
    "WSH": "Capitals",
}

ANY = "— Any —"


def _disp(abbr: str) -> str:
    return NHL_TEAM_DISPLAY.get(abbr, abbr)


def enrich_games(games: pd.DataFrame) -> pd.DataFrame:
    """Add parsed dates and human-readable pick labels."""
    df = games.copy()
    df["game_date"] = df["game_date"].astype(str)
    df["dt"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["away_name"] = df["away_abbr"].map(_disp)
    df["home_name"] = df["home_abbr"].map(_disp)
    df["pick_label"] = (
        df["dt"].dt.strftime("%Y-%m-%d")
        + " | "
        + df["away_name"]
        + " @ "
        + df["home_name"]
    )
    df["score_line"] = df["away_score"].astype(str) + "–" + df["home_score"].astype(str)
    return df.sort_values(["dt", "game_id"], ascending=[False, False]).reset_index(drop=True)


def filter_by_date_mode(
    df: pd.DataFrame,
    mode: Literal["all", "month", "range", "recent"],
    month_str: Optional[str],
    d_start: Optional[date],
    d_end: Optional[date],
    recent_days: int,
) -> pd.DataFrame:
    out = df
    ref_end = out["dt"].max()
    if pd.isna(ref_end):
        return out
    ref_end = pd.Timestamp(ref_end).normalize()

    if mode == "recent" and recent_days > 0:
        lo = ref_end - timedelta(days=recent_days)
        out = out[out["dt"].normalize() >= lo]
    elif mode == "month" and month_str and month_str != ANY:
        # month_str like "2026-03"
        y, m = month_str.split("-")
        mask = (out["dt"].dt.year == int(y)) & (out["dt"].dt.month == int(m))
        out = out[mask]
    elif mode == "range" and d_start is not None and d_end is not None:
        lo = pd.Timestamp(d_start)
        hi = pd.Timestamp(d_end) + pd.Timedelta(days=1)
        out = out[(out["dt"] >= lo) & (out["dt"] < hi)]
    return out


def filter_by_teams(
    df: pd.DataFrame,
    focus: str,
    opponent: str,
    matchup_only: bool,
) -> pd.DataFrame:
    out = df
    focus_ok = focus and focus != ANY
    opp_ok = opponent and opponent != ANY

    if focus_ok:
        out = out[(out["away_abbr"] == focus) | (out["home_abbr"] == focus)]

    if opp_ok:
        if focus_ok and matchup_only:
            out = out[
                ((out["away_abbr"] == focus) & (out["home_abbr"] == opponent))
                | ((out["away_abbr"] == opponent) & (out["home_abbr"] == focus))
            ]
        elif focus_ok and not matchup_only:
            # Any game that includes both clubs (not necessarily head-to-head if already narrowed)
            out = out[(out["away_abbr"] == opponent) | (out["home_abbr"] == opponent)]
        else:
            out = out[(out["away_abbr"] == opponent) | (out["home_abbr"] == opponent)]
    return out


def filter_search(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q or not q.strip():
        return df
    s = q.strip().upper()
    m = (
        df["pick_label"].str.upper().str.contains(s, na=False)
        | df["away_abbr"].str.upper().str.contains(s, na=False)
        | df["home_abbr"].str.upper().str.contains(s, na=False)
        | df["away_name"].str.upper().str.contains(s, na=False)
        | df["home_name"].str.upper().str.contains(s, na=False)
        | df["game_id"].astype(str).str.contains(s, na=False)
        | df["game_date"].str.upper().str.contains(s, na=False)
    )
    return df[m].reset_index(drop=True)


def table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Columns for the sidebar preview table (matches row index to `df`)."""
    return pd.DataFrame(
        {
            "Date": df["dt"].dt.strftime("%Y-%m-%d"),
            "Away": df["away_name"] + " (" + df["away_abbr"] + ")",
            "Home": df["home_name"] + " (" + df["home_abbr"] + ")",
            "Final": df["score_line"],
        }
    )


def month_choices(df: pd.DataFrame) -> list[str]:
    if df.empty or df["dt"].isna().all():
        return []
    periods = df["dt"].dt.to_period("M").dropna().unique()
    labels = sorted([str(p) for p in periods], reverse=True)
    return labels


def team_options(df: pd.DataFrame) -> list[str]:
    t = sorted(set(df["home_abbr"]).union(df["away_abbr"]))
    return [ANY] + t


def dataframe_selection_rows(ev) -> list[int]:
    """Normalize Streamlit dataframe `on_select` return value to row indices."""
    if ev is None:
        return []
    if isinstance(ev, dict):
        sel = ev.get("selection") or {}
        if isinstance(sel, dict):
            return list(sel.get("rows") or [])
        return list(getattr(sel, "rows", []) or [])
    sel = getattr(ev, "selection", None)
    if sel is None:
        return []
    return list(getattr(sel, "rows", []) or [])
