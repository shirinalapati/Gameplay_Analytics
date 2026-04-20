"""
Analytics and shot-quality modeling.

**Shot quality choice (explicit):**

We do **not** ship a proprietary calibrated xG model here. Instead we use a
**transparent geometric expected-threat proxy**:

  * NHL play-by-play gives shot coordinates (ft) in a rink-fixed system with nets
    near x = +89 and x = -89 (center ice at 0).
  * For each shot we take distance to the **nearest** goal mouth at (+/-89, 0) and
    an angle proxy at that net, then map through a smooth logistic to a 0–1
    **shot quality index** (unitless threat — interpretable as *relative* danger,
    not calibrated goal probability).

This supports cumulative threat timelines, momentum, and shot maps without
external model artifacts. Swap in your own xG outputs by joining on
(game_id, event_idx) if you have them.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Rink constants (feet) — NHL standard approximations for visualization
NET_X = 89.0
NET_Y = 0.0
GOAL_LINE_X = 89.0

# High-danger region (heuristic slot): close to nearest net and central
HIGH_DANGER_DIST_FT = 22.0
HIGH_DANGER_WIDTH_FT = 24.0


def nearest_goal_mouth(x: float, y: float) -> Tuple[float, float, float]:
    """
    Return (net_x, distance_ft, angle_rad) using whichever net is closer in (x,y).
    Angle is measured at the net between the goal line normal and shooter vector.
    """
    d_left = math.hypot(x + GOAL_LINE_X, y - NET_Y)
    d_right = math.hypot(x - GOAL_LINE_X, y - NET_Y)
    if d_left <= d_right:
        net_x = -GOAL_LINE_X
        dist = d_left
    else:
        net_x = GOAL_LINE_X
        dist = d_right
    vx, vy = x - net_x, y - NET_Y
    # Angle from center line toward shooter (0 = straight ahead from net)
    angle = abs(math.atan2(vy, vx))
    return net_x, dist, angle


def shot_quality_index(x: Optional[float], y: Optional[float]) -> float:
    """
    Map (x,y) to a 0–1 threat index (higher = more dangerous location).
    Uses logistic on distance and a mild angle bonus for central lanes.
    """
    if x is None or y is None:
        return 0.0
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return 0.0

    _, dist, angle = nearest_goal_mouth(xf, yf)
    # Distance decay: close shots dominate
    d_term = 1.0 / (1.0 + math.exp((dist - 28.0) / 7.0))
    # Angle bonus: shots from straight ahead slightly up-weighted vs sharp angle
    angle_term = 0.55 + 0.45 * math.cos(min(angle, math.pi / 2))
    # Lane: slightly favor central y
    lane = math.exp(-((yf / 42.5) ** 2) / 0.45)
    q = float(np.clip(d_term * angle_term * (0.75 + 0.25 * lane), 0.0, 1.0))
    return q


def is_high_danger(x: Optional[float], y: Optional[float]) -> bool:
    if x is None or y is None:
        return False
    _, dist, _ = nearest_goal_mouth(float(x), float(y))
    return (dist < HIGH_DANGER_DIST_FT) and (abs(float(y)) < HIGH_DANGER_WIDTH_FT)


def rolling_momentum(
    df: pd.DataFrame,
    window_events: int,
    team_col: str = "team_abbr",
    quality_col: str = "shot_quality",
) -> pd.DataFrame:
    """
    Rolling sum of shot quality over the last `window_events` *rows* of shot attempts.
    Returns a dataframe indexed like df with columns per team for rolling threat.
    """
    if df.empty:
        return df
    work = df.copy()
    # Only shot rows contribute; others zero-filled
    if quality_col not in work.columns:
        work[quality_col] = 0.0
    teams = sorted(work[team_col].dropna().unique())
    out = work[[c for c in ["game_seconds", team_col, quality_col] if c in work.columns]].copy()
    for t in teams:
        mask = work[team_col] == t
        s = work[quality_col].where(mask, 0.0)
        out[f"roll_{t}"] = s.rolling(window_events, min_periods=1).sum()
    return out


def cumulative_threat_by_team(
    shots: pd.DataFrame,
    time_col: str = "game_seconds",
    team_col: str = "team_abbr",
    quality_col: str = "shot_quality",
) -> pd.DataFrame:
    """Build cumulative shot quality series for each team over game time."""
    if shots.empty:
        return shots
    rows = []
    for t in sorted(shots[team_col].dropna().unique()):
        sub = shots[shots[team_col] == t].sort_values(time_col)
        sub = sub.assign(cum_threat=sub[quality_col].cumsum())
        rows.append(sub.assign(team=t))
    return pd.concat(rows, ignore_index=True)


def detect_momentum_swings(
    shot_times: np.ndarray,
    home_threat_rate: np.ndarray,
    away_threat_rate: np.ndarray,
    threshold_ratio: float = 1.6,
) -> list[dict]:
    """
    Simple swing detector: flag times where rolling threat ratio crosses threshold.
    Returns annotation dicts for plotting.
    """
    if len(shot_times) < 3:
        return []
    ann = []
    eps = 1e-6
    ratio = (home_threat_rate + eps) / (away_threat_rate + eps)
    for i in range(1, len(ratio)):
        if ratio[i] > threshold_ratio and ratio[i - 1] <= threshold_ratio:
            ann.append(
                {"t": float(shot_times[i]), "label": "Home surge", "side": "home"}
            )
        elif (1.0 / ratio[i]) > threshold_ratio and (1.0 / ratio[i - 1]) <= threshold_ratio:
            ann.append(
                {"t": float(shot_times[i]), "label": "Away surge", "side": "away"}
            )
    return ann


def _safe_team_view(df: pd.DataFrame, team_col: str = "team_abbr") -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["game_seconds", team_col])
    out = df.copy()
    if "game_seconds" not in out.columns:
        out["game_seconds"] = np.nan
    out["game_seconds"] = pd.to_numeric(out["game_seconds"], errors="coerce")
    out = out[out["game_seconds"].notna()]
    if team_col not in out.columns:
        out[team_col] = None
    out[team_col] = out[team_col].astype(str)
    return out


def momentum_profile(
    events: pd.DataFrame,
    shots: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    *,
    team_col: str = "team_abbr",
    quality_col: str = "shot_quality",
    half_life_sec: float = 60.0,
    oz_event_weight: float = 0.075,
    entry_weight: float = 0.26,
    entry_success_bonus: float = 0.52,
    entry_success_window_sec: float = 12.0,
) -> pd.DataFrame:
    """
    Explicit momentum model with exponentially decayed impulses.

    Momentum_t(team) = exp(-dt / half_life) * Momentum_(t-1)
                       + ThreatImpulse_t
                       + oz_event_weight * OffensiveZoneEvent_t
                       + entry_weight * EntryProxy_t
                       + entry_success_bonus * EntryToShotSuccess_t
    """
    if events is None or events.empty:
        return pd.DataFrame(
            columns=[
                "game_seconds",
                "home_momentum",
                "away_momentum",
                "net_momentum",
                "home_impulse",
                "away_impulse",
                "home_entry_success",
                "away_entry_success",
            ]
        )

    ev = _safe_team_view(events, team_col=team_col)
    sh = _safe_team_view(shots, team_col=team_col)
    if quality_col not in sh.columns:
        sh[quality_col] = 0.0
    teams = [home_abbr, away_abbr]

    # Shot-quality impulses (threat).
    threat_impulse = (
        sh.groupby(["game_seconds", team_col], as_index=False)[quality_col]
        .sum()
        .rename(columns={quality_col: "threat_impulse"})
    )
    # Offensive-zone event pressure proxy.
    oz = ev[ev["zone_code"].astype(str).str.upper() == "O"]
    oz_impulse = (
        oz.groupby(["game_seconds", team_col], as_index=False)
        .size()
        .rename(columns={"size": "oz_events"})
    )

    # Controlled-entry proxy: same team transitions into O-zone from N/D within 8s.
    e = ev.sort_values("game_seconds").reset_index(drop=True)
    e["prev_team"] = e[team_col].shift(1)
    e["prev_zone"] = e["zone_code"].astype(str).str.upper().shift(1)
    e["prev_t"] = e["game_seconds"].shift(1)
    e["zone_now"] = e["zone_code"].astype(str).str.upper()
    e["dt_prev"] = e["game_seconds"] - e["prev_t"]
    entry_proxy = e[
        (e[team_col].isin(teams))
        & (e["zone_now"] == "O")
        & (e["prev_team"] == e[team_col])
        & (e["prev_zone"].isin(["N", "D"]))
        & (e["dt_prev"].between(0, 8.0))
    ][["game_seconds", team_col]].copy()

    if entry_proxy.empty:
        entry_proxy["entry_success"] = []
    else:
        # Success proxy: entry followed by same-team shot in X seconds.
        sh_small = sh[[team_col, "game_seconds"]].sort_values("game_seconds")
        succ = []
        for r in entry_proxy.itertuples(index=False):
            hit = sh_small[
                (sh_small[team_col] == getattr(r, team_col))
                & (sh_small["game_seconds"] > r.game_seconds)
                & (sh_small["game_seconds"] <= r.game_seconds + entry_success_window_sec)
            ]
            succ.append(1 if not hit.empty else 0)
        entry_proxy["entry_success"] = succ

    entry_impulse = (
        entry_proxy.groupby(["game_seconds", team_col], as_index=False)
        .agg(entries=("entry_success", "size"), entry_success=("entry_success", "sum"))
    )

    # Build impulse table on event timestamps.
    ts = sorted(set(ev["game_seconds"].tolist()) | set(sh["game_seconds"].tolist()))
    if not ts:
        return pd.DataFrame()
    base = pd.DataFrame({"game_seconds": ts})
    pieces = {}
    for team in teams:
        t = base.copy()
        t["team_abbr"] = team
        t = t.merge(threat_impulse, how="left", on=["game_seconds", "team_abbr"])
        t = t.merge(oz_impulse, how="left", on=["game_seconds", "team_abbr"])
        t = t.merge(entry_impulse, how="left", on=["game_seconds", "team_abbr"])
        t = t.fillna({"threat_impulse": 0.0, "oz_events": 0, "entries": 0, "entry_success": 0})
        t["impulse"] = (
            t["threat_impulse"]
            + oz_event_weight * t["oz_events"].astype(float)
            + entry_weight * t["entries"].astype(float)
            + entry_success_bonus * t["entry_success"].astype(float)
        )
        pieces[team] = t

    out = pd.DataFrame({"game_seconds": ts})
    decay_denom = max(1e-6, float(half_life_sec))
    for team, prefix in ((home_abbr, "home"), (away_abbr, "away")):
        m = 0.0
        vals = []
        prev_t = float(ts[0])
        team_imp = pieces[team].set_index("game_seconds")
        for t in ts:
            tt = float(t)
            dt = max(0.0, tt - prev_t)
            m *= math.exp(-dt / decay_denom)
            m += float(team_imp.at[t, "impulse"])
            vals.append(m)
            prev_t = tt
        out[f"{prefix}_momentum"] = vals
        out[f"{prefix}_impulse"] = pieces[team]["impulse"].astype(float).values
        out[f"{prefix}_entry_success"] = pieces[team]["entry_success"].astype(float).values
    out["net_momentum"] = out["home_momentum"] - out["away_momentum"]
    return out


def pressure_summary_last_seconds(
    shots: pd.DataFrame,
    current_time: float,
    window_sec: float,
    team_col: str = "team_abbr",
    quality_col: str = "shot_quality",
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate pressure ledger stats in (current_time - window, current_time]."""
    if shots.empty:
        return pd.DataFrame(
            columns=[
                team_col,
                "attempts",
                "threat",
                "high_danger",
                "rebound_chains",
                "oz_events",
                "oz_share",
                "possession_proxy_sec",
                "longest_oz_sequence_sec",
            ]
        )
    lo = current_time - window_sec
    seg = shots[(shots["game_seconds"] > lo) & (shots["game_seconds"] <= current_time)].copy()
    agg = (
        seg.groupby(team_col, as_index=False)
        .agg(
            attempts=(quality_col, "count"),
            threat=(quality_col, "sum"),
            high_danger=("high_danger", "sum"),
        )
        .sort_values("threat", ascending=False)
    )

    # Rebound chains: consecutive same-team shots <= 3 seconds apart.
    seg = seg.sort_values(["team_abbr", "game_seconds"])
    seg["prev_t"] = seg.groupby(team_col)["game_seconds"].shift(1)
    seg["rebound"] = ((seg["game_seconds"] - seg["prev_t"]) <= 3.0).astype(int)
    rb = seg.groupby(team_col, as_index=False)["rebound"].sum().rename(columns={"rebound": "rebound_chains"})
    agg = agg.merge(rb, on=team_col, how="left")

    # Event-derived context for possession/zone control.
    if events is not None and not events.empty and team_col in events.columns:
        ev = events[(events["game_seconds"] > lo) & (events["game_seconds"] <= current_time)].copy()
        ev = ev.sort_values("game_seconds")
        oz = ev[ev["zone_code"].astype(str).str.upper() == "O"]
        oz_counts = oz.groupby(team_col, as_index=False).size().rename(columns={"size": "oz_events"})
        total_oz = float(max(1, len(oz)))
        oz_counts["oz_share"] = oz_counts["oz_events"].astype(float) / total_oz

        ev["next_t"] = ev["game_seconds"].shift(-1)
        ev["dt"] = (ev["next_t"] - ev["game_seconds"]).clip(lower=0.0, upper=8.0).fillna(0.0)
        poss = ev.groupby(team_col, as_index=False)["dt"].sum().rename(columns={"dt": "possession_proxy_sec"})

        oz = oz.copy()
        oz["prev_team"] = oz[team_col].shift(1)
        oz["prev_t"] = oz["game_seconds"].shift(1)
        oz["new_chain"] = ((oz[team_col] != oz["prev_team"]) | ((oz["game_seconds"] - oz["prev_t"]) > 12.0)).astype(int)
        oz["chain_id"] = oz["new_chain"].cumsum()
        chain_len = (
            oz.groupby([team_col, "chain_id"], as_index=False)["game_seconds"]
            .agg(lambda s: float(s.max() - s.min()) if len(s) > 1 else 0.0)
            .groupby(team_col, as_index=False)["game_seconds"]
            .max()
            .rename(columns={"game_seconds": "longest_oz_sequence_sec"})
        )

        agg = agg.merge(oz_counts, on=team_col, how="left")
        agg = agg.merge(poss, on=team_col, how="left")
        agg = agg.merge(chain_len, on=team_col, how="left")
    agg = agg.fillna(
        {
            "rebound_chains": 0,
            "oz_events": 0,
            "oz_share": 0.0,
            "possession_proxy_sec": 0.0,
            "longest_oz_sequence_sec": 0.0,
        }
    )
    return agg.sort_values("threat", ascending=False)


def detect_key_moments(
    momentum_df: pd.DataFrame,
    shots: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    *,
    swing_window_sec: int = 50,
    burst_window_sec: float = 12.0,
) -> dict:
    """Detect biggest momentum swing, strongest burst, and a turning-point proxy."""
    out = {"swing": None, "burst": None, "turning_point": None}
    if momentum_df is None or momentum_df.empty:
        return out

    m = momentum_df.sort_values("game_seconds").copy()
    m = m.set_index("game_seconds")
    sec_idx = np.arange(int(m.index.min()), int(m.index.max()) + 1)
    step = m.reindex(sec_idx).ffill().bfill()
    step.index.name = "game_seconds"
    step = step.reset_index()
    step["swing_delta"] = step["net_momentum"] - step["net_momentum"].shift(swing_window_sec)
    if step["swing_delta"].notna().any():
        i = int(step["swing_delta"].abs().idxmax())
        row = step.iloc[i]
        d = float(row["swing_delta"])
        out["swing"] = {
            "team": home_abbr if d >= 0 else away_abbr,
            "delta": abs(d),
            "start_sec": max(0.0, float(row["game_seconds"] - swing_window_sec)),
            "end_sec": float(row["game_seconds"]),
        }

    # Highest threat burst sequence in burst_window_sec.
    if shots is not None and not shots.empty:
        s = shots.sort_values("game_seconds").reset_index(drop=True)
        best = None
        for team in [home_abbr, away_abbr]:
            st = s[s["team_abbr"] == team].reset_index(drop=True)
            j = 0
            acc = 0.0
            hd_acc = 0.0
            for i in range(len(st)):
                while j < len(st) and (float(st.loc[j, "game_seconds"]) - float(st.loc[i, "game_seconds"]) <= burst_window_sec):
                    acc += float(st.loc[j, "shot_quality"])
                    hd_acc += float(st.loc[j, "high_danger"])
                    j += 1
                attempts = j - i
                if attempts >= 2:
                    cand = {
                        "team": team,
                        "threat": acc,
                        "attempts": attempts,
                        "high_danger": int(hd_acc),
                        "start_sec": float(st.loc[i, "game_seconds"]),
                        "end_sec": float(st.loc[j - 1, "game_seconds"]),
                    }
                    if best is None or cand["threat"] > best["threat"]:
                        best = cand
                acc -= float(st.loc[i, "shot_quality"])
                hd_acc -= float(st.loc[i, "high_danger"])
        out["burst"] = best

    # Turning point proxy: strongest sustained net momentum edge after midpoint.
    tail = step[step["game_seconds"] >= (0.5 * float(step["game_seconds"].max()))]
    if not tail.empty:
        ix = int(tail["net_momentum"].abs().idxmax())
        tr = tail.loc[ix]
        out["turning_point"] = {
            "team": home_abbr if float(tr["net_momentum"]) >= 0 else away_abbr,
            "net_momentum": abs(float(tr["net_momentum"])),
            "sec": float(tr["game_seconds"]),
        }
    return out


def game_takeaway_summary(
    shots: pd.DataFrame,
    momentum_df: pd.DataFrame,
    key_moments: dict,
    home_abbr: str,
    away_abbr: str,
    final_home_score: int,
    final_away_score: int,
) -> dict:
    """Generate end-of-game 'so what?' summary fields for analyst-facing readout."""
    def _threat(team: str) -> float:
        if shots is None or shots.empty:
            return 0.0
        return float(shots[shots["team_abbr"] == team]["shot_quality"].sum())

    home_t = _threat(home_abbr)
    away_t = _threat(away_abbr)
    threat_diff = home_t - away_t
    expected_winner = home_abbr if threat_diff >= 0 else away_abbr
    scoreboard_winner = home_abbr if final_home_score >= final_away_score else away_abbr
    mismatch = expected_winner != scoreboard_winner

    decisive = "Third period / late game"
    tp = key_moments.get("turning_point") if key_moments else None
    if tp:
        sec = float(tp["sec"])
        if sec < 1200:
            decisive = "First period"
        elif sec < 2400:
            decisive = "Second period"
    return {
        "expected_winner": expected_winner,
        "scoreboard_winner": scoreboard_winner,
        "threat_diff": abs(threat_diff),
        "score_diff": abs(int(final_home_score) - int(final_away_score)),
        "score_process_mismatch": mismatch,
        "decisive_phase": decisive,
    }
