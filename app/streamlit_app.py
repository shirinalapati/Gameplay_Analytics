"""
Game Playback Analytics — Streamlit console (2025–2026 NHL regular season).

Run from project root:

    export PYTHONPATH=.
    streamlit run app/streamlit_app.py

Deploy: see README (Docker / cloud). Requires SQLite at data/processed/nhl_playback.db.
"""

from __future__ import annotations

import sys
import time
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import (
    detect_key_moments,
    game_takeaway_summary,
    momentum_profile,
    pressure_summary_last_seconds,
)
from src.playback import (
    DEFAULT_DB_PATH,
    current_time,
    list_games,
    load_events,
    load_game_meta,
    load_shots,
    slice_at_event,
)
from src.game_selection import (
    ANY,
    dataframe_selection_rows,
    enrich_games,
    filter_by_date_mode,
    filter_by_teams,
    filter_search,
    month_choices,
    table_for_display,
    team_options,
)
from src.utils import EXPECTED_REGULAR_SEASON_GAME_TOTAL, format_situation, get_connection
from src.visuals import (
    fig_cumulative_threat,
    fig_danger_counts,
    fig_dual_bar_compare,
    fig_momentum_model,
    fig_rolling_momentum,
    fig_shot_map,
)

SPEED_TO_INTERVAL_MS = {
    0.5: 900,
    1.0: 700,
    2.0: 400,
    5.0: 180,
}

st.set_page_config(
    page_title="Game Playback Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#38bdf8"
TEXT_MAIN = "#f9fafb"
TEXT_MUTED = "#cbd5e1"
BG = "#111827"
CARD = "#1f2937"
ERR_BG = "#1a1030"
ERR_FG = "#f0abfc"


def _css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; color: {TEXT_MAIN}; }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {TEXT_MAIN}; }}
        section[data-testid="stSidebar"] > div {{
            background-color: {CARD};
        }}
        div[data-testid="stMetricValue"] {{ color: {ACCENT}; }}
        div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
        a {{ color: #7dd3fc !important; }}
        a:hover {{ color: #bae6fd !important; }}
        [data-testid="stCaption"] {{ color: {TEXT_MUTED} !important; }}
        div[data-testid="stDecoration"] {{ background: {CARD}; }}
        /* Streamlit exception / traceback readability */
        .stException, .stException * {{
            background-color: {ERR_BG} !important;
            color: {ERR_FG} !important;
        }}
        pre, code {{ font-size: 0.85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_css()


@st.cache_data(ttl=120, show_spinner=False)
def _cached_games(db_path: str) -> pd.DataFrame:
    with closing(get_connection(Path(db_path))) as cx:
        return list_games(cx)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_events(db_path: str, game_id: int) -> pd.DataFrame:
    with closing(get_connection(Path(db_path))) as cx:
        return load_events(cx, game_id)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_shots(db_path: str, game_id: int) -> pd.DataFrame:
    with closing(get_connection(Path(db_path))) as cx:
        return load_shots(cx, game_id)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_meta(db_path: str, game_id: int) -> pd.DataFrame:
    with closing(get_connection(Path(db_path))) as cx:
        return load_game_meta(cx, game_id)


def _ensure_state(game_id: int, max_idx: int) -> None:
    if "game_id" not in st.session_state:
        st.session_state.game_id = game_id
    if "event_idx" not in st.session_state:
        st.session_state.event_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "speed" not in st.session_state:
        st.session_state.speed = 1.0
    if "_last_step_ts" not in st.session_state:
        st.session_state._last_step_ts = time.monotonic()
    if st.session_state.game_id != game_id:
        st.session_state.game_id = game_id
        st.session_state.event_idx = 0
        st.session_state.playing = False
        st.session_state._auto_pb_last = -1
        st.session_state._last_step_ts = time.monotonic()
    st.session_state.event_idx = int(max(0, min(int(st.session_state.event_idx), max_idx)))


def _pick_game_interactive(games_raw: pd.DataFrame) -> int:
    """
    Multi-step filters + searchable list + preview table (no single giant dropdown of all games).
    Returns selected game_id.
    """
    base = enrich_games(games_raw)

    st.sidebar.markdown("### Pick a game")
    month_sel = ANY
    with st.sidebar.expander("Filters", expanded=True):
        date_scope = st.selectbox(
            "Date scope",
            ["all", "recent", "month", "range"],
            format_func=lambda k: {
                "all": "All dates",
                "recent": "Recent games",
                "month": "One month",
                "range": "Date range",
            }[k],
            key="sf_date_scope",
        )
        recent_days = 14
        if date_scope == "recent":
            recent_days = int(st.selectbox("Recent window (days)", [7, 14, 30], index=1, key="sf_recent"))

        if date_scope == "month":
            mopts = month_choices(base)
            if not mopts:
                st.caption("No months found in DB.")
            month_sel = st.selectbox(
                "Month",
                [ANY] + mopts,
                index=0,
                key="sf_month",
            )

        d_start = d_end = None
        if date_scope == "range":
            lo = pd.Timestamp(base["dt"].min()).date()
            hi = pd.Timestamp(base["dt"].max()).date()
            c1, c2 = st.columns(2)
            d_start = c1.date_input("From", value=lo, min_value=lo, max_value=hi, key="sf_d0")
            d_end = c2.date_input("To", value=hi, min_value=lo, max_value=hi, key="sf_d1")
            if d_start and d_end and d_start > d_end:
                d_start, d_end = d_end, d_start

        teams = team_options(base)
        t_focus = st.selectbox("Team", teams, key="sf_team_focus")
        t_opp = st.selectbox("Opponent (optional)", teams, key="sf_team_opp")
        matchup_only = st.checkbox(
            "Head-to-head only (both teams)",
            value=True,
            key="sf_h2h",
            help="If both teams are set: ON = only games between those two clubs; OFF = any game that includes both.",
        )

    search_q = st.sidebar.text_input(
        "Search",
        placeholder="e.g. Oilers, 2026-03-12, EDM",
        key="sf_search",
    )

    month_param = month_sel if (date_scope == "month" and month_sel != ANY) else None
    f = filter_by_date_mode(
        base,
        date_scope,
        month_param,
        d_start,
        d_end,
        recent_days,
    )
    f = filter_by_teams(f, t_focus, t_opp, matchup_only)
    f = filter_search(f, search_q).reset_index(drop=True)

    if f.empty:
        st.sidebar.warning("No games match these filters — widen date or team filters.")
        return int(base.iloc[0]["game_id"])

    st.sidebar.caption(f"**{len(f):,}** games match")

    is_playing = bool(st.session_state.get("playing", False))
    if is_playing:
        st.sidebar.caption("Preview table paused during playback for smoother replay.")
    else:
        disp = table_for_display(f)
        st.sidebar.markdown("**Preview** (sort / search in the table toolbar)")
        ev = st.dataframe(
            disp,
            width="stretch",
            height=260,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="games_browser_tbl",
        )
        rows = dataframe_selection_rows(ev)
        if rows:
            st.session_state.game_pick_idx = int(rows[0])

    opts = list(range(len(f)))
    mx = len(f) - 1
    raw_cur = st.session_state.get("game_pick_idx", 0)
    try:
        cur = int(raw_cur)
    except (TypeError, ValueError):
        # Handle legacy / invalid session values (e.g., label text persisted as key value).
        if isinstance(raw_cur, str) and raw_cur in set(f["pick_label"].astype(str)):
            cur = int(f.index[f["pick_label"].astype(str) == raw_cur][0])
        else:
            cur = 0
    if cur > mx or cur < 0:
        st.session_state.game_pick_idx = 0
        cur = 0

    ix = st.sidebar.selectbox(
        "Choose game",
        opts,
        index=cur,
        format_func=lambda i: f.iloc[int(i)]["pick_label"],
        key="game_pick_idx",
    )
    gid = int(f.iloc[int(ix)]["game_id"])
    st.session_state.picked_gid = gid
    return gid


def _playback_autotick(max_idx: int) -> None:
    """
    Advance one event per autorefresh tick while playing.
    Uses a monotonic counter from streamlit-autorefresh (full reruns), not arbitrary widget reruns.
    """
    if not st.session_state.playing or st.session_state.event_idx >= max_idx:
        if st.session_state.event_idx >= max_idx:
            st.session_state.playing = False
        return

    interval_ms = int(SPEED_TO_INTERVAL_MS.get(float(st.session_state.speed), 700))
    # debounce=True (default): timer-driven refreshes only, not every widget rerun — avoids double-steps.
    tick = st_autorefresh(interval=interval_ms, key="playback_autorefresh", debounce=True)
    prev = int(st.session_state.get("_auto_pb_last", -1))
    now = time.monotonic()
    last_step_ts = float(st.session_state.get("_last_step_ts", now))
    # Guard against extra reruns: only advance when a new timer tick arrives AND enough wall time passed.
    if tick > prev and (now - last_step_ts) >= (interval_ms / 1000.0) * 0.9:
        st.session_state._auto_pb_last = tick
        st.session_state._last_step_ts = now
        st.session_state.event_idx = min(int(st.session_state.event_idx) + 1, max_idx)
    if st.session_state.event_idx >= max_idx:
        st.session_state.playing = False


def _fmt_clock(sec: float) -> str:
    sec_i = int(max(0, sec))
    mm, ss = divmod(sec_i, 60)
    return f"{mm:02d}:{ss:02d}"


def _fmt_period_clock(period: int, game_seconds: float) -> str:
    """Format elapsed/remaining time within the current period."""
    p = int(max(1, period))
    t = float(max(0.0, game_seconds))
    if p <= 3:
        period_len = 1200.0
        start = float((p - 1) * 1200)
    else:
        period_len = 300.0
        start = float(3600 + (p - 4) * 300)
    elapsed = max(0.0, min(period_len, t - start))
    remaining = max(0.0, period_len - elapsed)
    return f"{_fmt_clock(elapsed)} elapsed ({_fmt_clock(remaining)} remaining)"


def _events_with_team_abbr(events: pd.DataFrame, meta: pd.Series) -> pd.DataFrame:
    out = events.copy()
    try:
        home_id = int(float(meta.get("home_team_id")))
    except (TypeError, ValueError):
        home_id = -1
    try:
        away_id = int(float(meta.get("away_team_id")))
    except (TypeError, ValueError):
        away_id = -1
    home_abbr = str(meta.get("home_abbr") or "")
    away_abbr = str(meta.get("away_abbr") or "")
    out["team_abbr"] = None
    if "team_id" in out.columns:
        out.loc[out["team_id"] == home_id, "team_abbr"] = home_abbr
        out.loc[out["team_id"] == away_id, "team_abbr"] = away_abbr
    return out


def _on_scrub_change(scrub_key: str, max_idx: int) -> None:
    """User-driven scrub should become the single source of truth for event_idx."""
    try:
        idx = int(st.session_state.get(scrub_key, 0))
    except (TypeError, ValueError):
        idx = 0
    st.session_state.event_idx = int(max(0, min(idx, max_idx)))
    st.session_state.playing = False
    st.session_state._auto_pb_last = -1
    st.session_state._last_step_ts = time.monotonic()


def main() -> None:
    db_s = str(DEFAULT_DB_PATH)
    try:
        games = _cached_games(db_s)
    except Exception as e:
        st.error(f"Could not read database at `{db_s}`. Run ingest first.\n\n`{e}`")
        return

    if games.empty:
        st.error(
            "No games in the database. Ingest 2025–26 data first:\n\n"
            "`export PYTHONPATH=. && python -m src.ingest --limit-games 5`"
        )
        return

    st.sidebar.title("Controls")
    n_games = len(games)
    if n_games < EXPECTED_REGULAR_SEASON_GAME_TOTAL:
        st.sidebar.info(
            f"A complete **2025–26** regular season is **{EXPECTED_REGULAR_SEASON_GAME_TOTAL:,}** games, "
            "and by season’s end they are all finished (**OFF** / **FINAL** in the schedule API). "
            "If your count is lower, the pipeline has not loaded every game yet — run a **full ingest** "
            "(and use **Clear chart cache** after it finishes).\n\n"
            "`export PYTHONPATH=. && python -m src.ingest`"
        )

    gid = _pick_game_interactive(games)

    try:
        meta = _cached_meta(db_s, gid).iloc[0]
        events = _cached_events(db_s, gid)
        shots = _cached_shots(db_s, gid)
    except Exception as e:
        st.error(f"Failed to load game {gid}: {e}")
        return

    if events.empty:
        st.error("This game has no normalized events. Re-run `python -m src.clean`.")
        return

    max_idx = max(0, len(events) - 1)
    _ensure_state(gid, max_idx)
    scrub_key = f"event_scrub_{gid}"

    win_sec = st.sidebar.slider("Rolling window (seconds)", min_value=20, max_value=300, value=120, step=10)
    pressure_win = st.sidebar.slider("Pressure summary window (minutes)", 1, 10, 5)
    momentum_half_life = st.sidebar.slider("Momentum decay half-life (seconds)", 20, 180, 60, 5)
    prev_speed = float(st.session_state.speed)
    speed = st.sidebar.select_slider(
        "Playback speed",
        options=[0.5, 1.0, 2.0, 5.0],
        value=prev_speed if prev_speed in SPEED_TO_INTERVAL_MS else 1.0,
    )
    st.session_state.speed = float(speed)
    if float(speed) != prev_speed:
        # Make speed changes take effect immediately during playback.
        st.session_state._auto_pb_last = -1
        st.session_state._last_step_ts = time.monotonic()
    map_period = st.sidebar.selectbox("Shot map period", ["All", "1", "2", "3", "OT+"], index=0)

    st.sidebar.markdown("---")
    row1_c1, row1_c2 = st.sidebar.columns(2)
    if row1_c1.button("Play", use_container_width=True, type="primary"):
        st.session_state.playing = True
        st.session_state._auto_pb_last = -1
        st.session_state._last_step_ts = time.monotonic()
    if row1_c2.button("Pause", use_container_width=True):
        st.session_state.playing = False
    row2_c1, row2_c2 = st.sidebar.columns(2)
    if row2_c1.button("Reset", use_container_width=True):
        st.session_state.playing = False
        st.session_state.event_idx = 0
        st.session_state._auto_pb_last = -1
        st.session_state._last_step_ts = time.monotonic()
        if scrub_key in st.session_state:
            del st.session_state[scrub_key]
    if row2_c2.button("Step", use_container_width=True, help="Advance one event"):
        st.session_state.playing = False
        st.session_state.event_idx = min(int(st.session_state.event_idx) + 1, max_idx)
        if scrub_key in st.session_state:
            del st.session_state[scrub_key]

    # Apply autoplay step BEFORE rendering slider so playback and scrubber stay in lockstep.
    if st.session_state.playing:
        _playback_autotick(max_idx)

    # Slider widget session value must match logical index or it overwrites advances on rerun.
    st.session_state[scrub_key] = int(st.session_state.event_idx)

    st.sidebar.markdown(f"**Events:** {len(events):,}")

    st.sidebar.slider(
        "Event scrubber",
        min_value=0,
        max_value=max_idx,
        key=scrub_key,
        disabled=bool(st.session_state.playing),
        on_change=_on_scrub_change,
        args=(scrub_key, max_idx),
        help="Drag to jump. Play uses auto-refresh ticks so the scrubber stays in sync.",
    )

    habbr, aabbr = str(meta["home_abbr"]), str(meta["away_abbr"])
    sub_e, sub_s, cur = slice_at_event(events, shots, st.session_state.event_idx)
    tnow = current_time(events, st.session_state.event_idx)
    tab_labels = [
        "About This App",
        "Game Playback",
        "Momentum & Pressure",
        "Shot Map",
        "Insights",
        "Team Comparison",
    ]
    if "active_main_tab" not in st.session_state or st.session_state.active_main_tab not in tab_labels:
        st.session_state.active_main_tab = "About This App"
    active_tab = st.radio(
        "View",
        tab_labels,
        key="active_main_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    # Heavy model computations only when needed. This keeps high-speed playback responsive.
    needs_model = active_tab in {"Momentum & Pressure", "Insights"}
    events_team = _events_with_team_abbr(sub_e, meta) if needs_model else pd.DataFrame()
    model_momentum = pd.DataFrame()
    key_moments: dict = {"swing": None, "burst": None, "turning_point": None}
    takeaway = {
        "expected_winner": habbr,
        "scoreboard_winner": habbr if (cur is not None and int(cur["score_home"]) >= int(cur["score_away"])) else aabbr,
        "threat_diff": 0.0,
        "score_diff": abs(int(cur["score_home"]) - int(cur["score_away"])) if cur is not None else 0,
        "score_process_mismatch": False,
        "decisive_phase": "Third period / late game",
    }
    if needs_model:
        events_team = _events_with_team_abbr(sub_e, meta)
        model_momentum = momentum_profile(
            events_team,
            sub_s,
            habbr,
            aabbr,
            half_life_sec=float(momentum_half_life),
            entry_success_window_sec=12.0,
        )
        key_moments = detect_key_moments(model_momentum, sub_s, habbr, aabbr)
        takeaway = game_takeaway_summary(
            shots=sub_s,
            momentum_df=model_momentum,
            key_moments=key_moments,
            home_abbr=habbr,
            away_abbr=aabbr,
            final_home_score=int(cur["score_home"]) if cur is not None else 0,
            final_away_score=int(cur["score_away"]) if cur is not None else 0,
        )

    if active_tab == "About This App":
        st.markdown(
            """
            ## About This App

            ### Game Playback Analytics: Real-Time Hockey Insights from NHL Play-by-Play

            This is a research-oriented analytics console that replays completed 2025–2026 NHL regular-season games from public play-by-play data and layers custom threat, momentum, and pressure metrics on top — not a public scoreboard product.

            ### Why simulated live playback?

            Hockey operations and analysts care less about final box score totals and more about when dangerous sequences cluster, how pressure evolves, and how shot quality accumulates over time.

            Replaying games event-by-event preserves timing, context, and game state, allowing patterns to emerge that are lost in static summaries.

            ### What this app shows
            - Cumulative shot threat over time (relative scoring danger)
            - Rolling momentum and pressure using decayed time windows
            - High-danger chance accumulation
            - Rink shot maps (size and color encoded by threat)
            - Automated insights on momentum swings, threat bursts, and turning points
            - Team-vs-team metric comparisons across threat and shot outcomes
            - Event-level playback controls for exploring sequences

            ### Controls include:
            - Play (auto-refresh replay)
            - Step (single event advancement)
            - Pause / Reset
            - Scrubber for jumping to any point in the game

            Games can be filtered by date, team, and opponent, with text search and a preview table — avoiding a single large dropdown across all 1,312 games.

            ## Interactive Controls

            The sidebar allows users to tune how pressure, momentum, and playback are interpreted:

            **Rolling window (seconds)**
            Defines how much recent game time is used when calculating rolling threat and pressure.
            Smaller values highlight short bursts of activity, while larger values emphasize sustained pressure.

            **Pressure summary window (minutes)**
            Controls the time span used for aggregated pressure metrics (e.g., attempts, threat, high-danger chances).
            Larger windows provide more stable summaries of which team is controlling play.

            **Momentum decay half-life (seconds)**
            Determines how quickly past events lose influence in the momentum model.
            A shorter half-life emphasizes recent sequences, while a longer half-life allows momentum to persist over time.

            **Playback speed**
            Adjusts how quickly events are replayed.
            This affects only the viewing experience, not the underlying analytics.
            All the options are:
            - 0.5x → 900 ms/event
            - 1.0x → 700 ms/event
            - 2.0x → 400 ms/event
            - 5.0x → 180 ms/event

            ## Methodology

            ### Data scope
            - Season: 2025–2026 NHL regular season only
            - Source: Public NHL Web API play-by-play JSON
            - Replay model: Completed games are stored in SQLite and replayed chronologically as a simulated live feed for analysis, not a live data stream

            ### Shot quality (Threat Index)

            Shot quality is modeled using a transparent geometric proxy based on shot location relative to the nearest goal (x = ±89 ft).

            - Interpreted as relative scoring danger, not calibrated goal probability
            - Comparable within this app, not across external xG models
            - Prioritizes interpretability and reproducibility over black-box modeling

            ### High-danger definition

            High-danger is a transparent location-based flag (not a trained model label):
            - Only unblocked shot attempts can be marked high-danger
            - Shot distance to the nearest net must be **< 22 ft**
            - Lateral offset must be within the slot band: **|y| < 24 ft**

           
            ### Rolling momentum

            Momentum is modeled using an explicit time-decay process that prioritizes recent events while retaining short-term context.

            $$
            M_t = e^{-\\Delta t / \\text{half-life}} \\cdot M_{t-1}
            + \\text{threat}_t
            + \\text{OZ}_t
            + \\text{entry}_t
            + \\text{entry→shot}_t
            $$

            Where:
            - $M_t$ is momentum at time $t$
            - $\\Delta t$ is time since the previous event
            - half-life controls how quickly past momentum decays (user-adjustable)
            - $\\text{threat}_t$ is the sum of shot_quality values for the team at time $t$
              (from shot attempts in the shots table at that timestamp).
            - $\\text{OZ}_t$ is the count of team events with zone_code == "O" at time $t$
              (offensive-zone pressure proxy from the events table).
            - $\\text{entry}_t$ is the count of entry proxies at time $t$:
              same team, transition into offensive zone (`O`) from neutral/defensive zone (`N`/`D`),
              and transition occurring within 8 seconds.
            - $\\text{entry→shot}_t$ is the count of successful entry proxies at time $t$:
              an entry_t event followed by a same-team shot within 12 seconds.

            **Momentum update rule**

            The model combines event-driven impulses with exponential decay:
            the expanded equation and the compact `impulse_t` form below are mathematically equivalent ways of writing the same update.

            $$
            \\text{impulse}_t = \\text{threat}_t
            + 0.075 \\cdot \\text{OZ}_t
            + 0.26 \\cdot \\text{entry}_t
            + 0.52 \\cdot (\\text{entry→shot}_t)
            $$

            $$
            M_t = e^{-\\frac{\\Delta t}{\\text{half-life}}} \\cdot M_{t-1} + \\text{impulse}_t
            $$

            This formulation captures bursts of sustained pressure while allowing momentum to naturally dissipate when play slows or possession changes.

            ### Entry & possession proxies

            Because public play-by-play does not include full tracking data, key transitions are estimated:

            - Entry proxy: Same-team transition into the offensive zone from neutral/defensive zone in short succession
            - Entry success: Entry followed by a same-team shot within 12 seconds
            - Possession proxy: Time between consecutive team-owned events (capped to avoid inflation)

            These proxies approximate transition play and sustained offensive pressure.

            ### Playback engine
            - Play mode: Uses streamlit-autorefresh to simulate continuous event flow
            - Step mode: Advances one event at a time without auto-play
            - All charts and metrics remain synchronized to the current event index

            ### Pipeline

            Data flows through a reproducible pipeline:

            **Ingest → Normalize → Store → Compute → Visualize**

            - Data ingestion and normalization
            - SQLite -> structured storage
            - Threat, momentum, and pressure calculations
            - Plotly visualizations
            - This app -> interactive analysis layer

            ### Limitations
            - No player tracking or pre-shot pass context
            - Shot-based threat does not capture screens, rush types, or goalie positioning
            - Entry and possession are proxy-based, not ground truth

            These tradeoffs prioritize open-data reproducibility while still capturing meaningful structure in game flow.

            ### Big picture

            This app is designed to answer questions like:

            - When did momentum actually shift?
            - Which sequences created meaningful scoring pressure?
            - Did the final score reflect the run of play?

            By focusing on process over outcome, it provides a clearer view of how hockey games are actually won and lost.
            """
        )

    if active_tab == "Game Playback":
        mcols = st.columns(4)
        mcols[0].metric("Clock (seconds)", f"{tnow:,.0f}")
        mcols[1].metric("Home", f"{habbr}  {int(cur['score_home']) if cur is not None else 0}")
        mcols[2].metric("Away", f"{aabbr}  {int(cur['score_away']) if cur is not None else 0}")
        mcols[3].metric("Playback", f"{st.session_state.event_idx + 1}/{len(events)}")
        st.caption("Playback advances by **event index** (full play-by-play), not broadcast clock.")

        if cur is not None:
            cur_period = int(cur["period"])
            cur_secs = float(cur["game_seconds"])
            situation_label = format_situation(str(cur["situation_code"] or ""))
            st.markdown(
                f"**Current event:** `{cur['type_desc_key']}` · P{cur_period} · "
                f"{_fmt_period_clock(cur_period, cur_secs)} · `{situation_label}`"
            )
        st.plotly_chart(
            fig_cumulative_threat(sub_s, habbr, aabbr, current_time=tnow),
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )
        if sub_s.empty and not shots.empty:
            first = shots.sort_values("game_seconds").iloc[0]
            first_sec = float(first["game_seconds"])
            first_event_idx = int(first["event_idx"])
            st.info(
                "Playback is advancing, but no shot attempts have occurred yet in this game state. "
                f"First shot arrives at `{_fmt_clock(first_sec)}` (event `{first_event_idx}`)."
            )
            if st.button("Jump to first shot", key=f"jump_first_shot_{gid}"):
                st.session_state.playing = False
                st.session_state.event_idx = int(max(0, min(first_event_idx, max_idx)))
                st.session_state._last_step_ts = time.monotonic()
                if scrub_key in st.session_state:
                    del st.session_state[scrub_key]
                st.rerun()

    if active_tab == "Momentum & Pressure":
        st.plotly_chart(
            fig_momentum_model(model_momentum, habbr, aabbr, key_moments=key_moments),
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )
        if not model_momentum.empty:
            latest = model_momentum.iloc[-1]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{habbr} model momentum", f"{float(latest['home_momentum']):.2f}")
            mc2.metric(f"{aabbr} model momentum", f"{float(latest['away_momentum']):.2f}")
            leader = habbr if float(latest["net_momentum"]) >= 0 else aabbr
            mc3.metric("Current edge", f"{leader} ({abs(float(latest['net_momentum'])):.2f})")

        st.plotly_chart(
            fig_rolling_momentum(sub_s, habbr, aabbr, window_sec=float(win_sec)),
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )
        st.plotly_chart(
            fig_danger_counts(sub_s, habbr, aabbr),
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )

        pres = pressure_summary_last_seconds(
            sub_s,
            current_time=tnow,
            window_sec=float(pressure_win) * 60.0,
            events=events_team,
        )
        st.subheader(f"Pressure ledger (last {pressure_win} min, threat-weighted)")
        if pres.empty:
            st.info("No shot attempts in this window yet.")
        else:
            show = pres.copy()
            if "oz_share" in show.columns:
                show["oz_share"] = (100.0 * show["oz_share"]).round(1).astype(str) + "%"
            show = show.rename(
                columns={
                    "threat": "total_threat",
                    "high_danger": "high_danger_attempts",
                    "rebound_chains": "rebound_chain_events",
                    "oz_events": "offensive_zone_events",
                    "oz_share": "offensive_zone_share",
                    "possession_proxy_sec": "possession_proxy_seconds",
                    "longest_oz_sequence_sec": "longest_offensive_zone_sequence_seconds",
                }
            )
            st.dataframe(show, width="stretch", hide_index=True)
            st.markdown(
                """
                - `team_abbr`: Team abbreviation for each row.
                - `attempts`: Number of shot attempts in the selected pressure window.
                - `total_threat`: Sum of shot-quality threat values in the window.
                - `high_danger_attempts`: Count of attempts marked as high-danger.
                - `rebound_chain_events`: Number of rebound-linked pressure events.
                - `offensive_zone_events`: Count of offensive-zone events in the window.
                - `offensive_zone_share`: Percentage of pressure-window events occurring in the offensive zone.
                - `possession_proxy_seconds`: Approximate same-team possession time from event timing.
                - `longest_offensive_zone_sequence_seconds`: Longest continuous offensive-zone sequence duration.
                """
            )

    if active_tab == "Shot Map":
        shots_for_map = sub_s
        if shots_for_map.empty and not shots.empty:
            shots_for_map = shots
            st.caption("Showing full-game shots because playback has not reached the first shot yet.")
        if map_period == "OT+":
            sub_for_map = shots_for_map[shots_for_map["period"] >= 4]
            st.plotly_chart(
                fig_shot_map(sub_for_map, habbr, aabbr, period_filter=None),
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False, "responsive": True},
            )
        else:
            mp = int(map_period) if map_period in {"1", "2", "3"} else None
            st.plotly_chart(
                fig_shot_map(shots_for_map, habbr, aabbr, period_filter=mp, max_seconds=None),
                width="stretch",
                config={"displayModeBar": True, "displaylogo": False, "responsive": True},
            )

    if active_tab == "Insights":
        st.subheader("Automated key-moment detection")
        swing = key_moments.get("swing")
        burst = key_moments.get("burst")
        turn = key_moments.get("turning_point")
        if swing:
            st.markdown(
                f"- **Biggest momentum swing:** `{swing['team']}` from `{_fmt_clock(swing['start_sec'])}` to `{_fmt_clock(swing['end_sec'])}` "
                f"(swing score `{swing['delta']:.2f}`)."
            )
        if burst:
            st.markdown(
                f"- **Highest threat burst:** `{burst['team']}` produced `{burst['attempts']}` attempts "
                f"({burst['high_danger']} high-danger) from `{_fmt_clock(burst['start_sec'])}` to `{_fmt_clock(burst['end_sec'])}` "
                f"for `{burst['threat']:.2f}` threat."
            )
        if turn:
            st.markdown(
                f"- **Turning point proxy:** `{turn['team']}` took sustained edge around `{_fmt_clock(turn['sec'])}` "
                f"(net momentum `{turn['net_momentum']:.2f}`)."
            )
        if not any([swing, burst, turn]):
            st.info("Not enough events yet at this playback point to detect robust key moments.")

        st.subheader("So what? end-of-window summary")
        s1, s2, s3 = st.columns(3)
        s1.metric("Process winner (threat)", takeaway["expected_winner"])
        s2.metric("Scoreboard leader", takeaway["scoreboard_winner"])
        s3.metric("Threat edge", f"{takeaway['threat_diff']:.2f}")
        if takeaway["score_process_mismatch"]:
            st.warning(
                "Scoreboard lead does not match underlying chance quality (process) in this playback slice."
            )
        st.caption(f"Decisive phase proxy: **{takeaway['decisive_phase']}**.")

    if active_tab == "Team Comparison":
        def _totals(side: pd.DataFrame) -> tuple[float, float, float, float]:
            if side is None or side.empty:
                return 0.0, 0.0, 0.0, 0.0
            unblocked = side[side["is_blocked"] == 0]
            blocked_ct = float((side["is_blocked"] == 1).sum())
            return (
                float(side["shot_quality"].sum()),
                float(len(unblocked)),
                blocked_ct,
                float(side["high_danger"].sum()),
            )

        h_vals = _totals(sub_s[sub_s["team_abbr"] == habbr] if not sub_s.empty else sub_s)
        a_vals = _totals(sub_s[sub_s["team_abbr"] == aabbr] if not sub_s.empty else sub_s)

        metrics_lbl = ["Cumulative threat", "Unblocked attempts", "Blocked attempts", "High-danger"]

        st.plotly_chart(
            fig_dual_bar_compare(
                metrics_lbl,
                list(h_vals),
                list(a_vals),
                habbr,
                aabbr,
                "Game totals (current playback)",
            ),
            width="stretch",
            config={"displayModeBar": True, "displaylogo": False, "responsive": True},
        )

        st.caption(
            "Totals update with playback position — they are **not** final box-score summaries "
            "unless you scrub to the last event."
        )

    st.divider()


main()
