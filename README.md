# Game Playback Analytics

Research-oriented Streamlit app for replaying completed 2025-2026 NHL regular-season games and analyzing game flow with custom threat, momentum, and pressure metrics.

## What this project does

- Replays events in order (play/pause/reset/step/scrub) instead of showing static box scores.
- Tracks cumulative threat, rolling momentum, high-danger accumulation, and shot maps.
- Detects key moments (momentum swings, threat bursts, turning-point proxies).
- Compares teams in the same playback window with pressure and chance-quality metrics.

## Why this exists

Final scores hide process. This app is designed to answer questions like:

- When did pressure actually shift?
- Which sequences created the most danger?
- Did scoreline and underlying process agree?

The goal is decision support and game review, not a public live-score product.

## Data scope

- Season: 2025-2026 NHL regular season only.
- Source: public NHL Web API (`api-web.nhle.com`).
- Coverage target: 1,312 games (32 teams * 82 games / 2).
- Replay mode: completed games replayed locally as simulated live flow.

## Methodology at a glance

- **Shot quality index (`shot_quality`)**: transparent geometric threat proxy based on shot distance/angle to nearest net (`x = +/-89`), bounded to `[0,1]`.
- **High-danger flag (`high_danger`)**: `1` only for unblocked attempts where distance to nearest net `< 22 ft` and `abs(y) < 24 ft`.
- **Blocked shots**: retained with down-weighted threat contribution.
- **Momentum model**: decayed accumulation of threat plus pressure proxies (offensive-zone events, entries, entry-to-shot success).
- **Pressure ledger**: windowed team summary including attempts, threat, high-danger, rebound chains, offensive-zone share, possession proxy, and longest O-zone sequence.

See `WHITE_PAPER.md` for detailed formulas, assumptions, and design rationale.

## Quick start

```bash
cd NHL_App
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build the database

Ingest full season window:

```bash
export PYTHONPATH=.
python -m src.ingest
```

Optional:

```bash
# Custom date window
python -m src.ingest --start 2025-09-24 --end 2026-04-22

# Quick smoke run
python -m src.ingest --limit-games 3
```

Rebuild SQLite from already-downloaded raw JSON:

```bash
export PYTHONPATH=.
python -m src.clean
```

## Run the app

```bash
export PYTHONPATH=.
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Docker run

```bash
docker build -t nhl-playback .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/data/processed/nhl_playback.db:/app/data/processed/nhl_playback.db:ro" \
  nhl-playback
```

## Project layout

```text
app/
  streamlit_app.py
src/
  ingest.py
  clean.py
  playback.py
  metrics.py
  visuals.py
  utils.py
sql/
  schema.sql
data/
  raw/        # raw API payloads (gitignored)
  processed/  # sqlite db (gitignored)
README.md
WHITE_PAPER.md
```

## Limitations

- No player-tracking or pre-shot pass context in public feed.
- Threat is a relative, interpretable proxy (not calibrated xG probability).
- Some contextual labels are heuristic and feed-dependent.
- Playback timing follows event order, not broadcast synchronization.

## Roadmap

- Plug in calibrated external xG model on `(game_id, event_idx)`.
- Add richer sequence and swing diagnostics.
- Expand persistence/deployment options beyond SQLite.
- Add automated tests around metric invariants.

## Attribution

Uses public NHL API data and is not affiliated with or endorsed by the NHL.
