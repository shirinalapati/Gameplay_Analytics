# White Paper: Replay-Driven NHL Game Flow Analytics

## Abstract

This project implements a transparent analytics framework for evaluating NHL game flow from public play-by-play data from the 2025-2026 regular season. Instead of emphasizing final outcomes, the system replays completed games event-by-event and computes interpretable proxies for threat, momentum, and pressure. The output is an interactive decision-support interface that helps analysts identify when control shifted, where dangerous sequences clustered, and whether the scoreline reflected underlying process.

## 1) Problem and Motivation

Traditional hockey summaries (score, shots, and basic counting stats) compress time and context. Coaching and operations workflows often require sequence-level understanding:

- Which team controlled dangerous play during key stretches?
- Did momentum shifts precede score changes?
- Were apparent leads supported by shot-quality process?

The design objective is to preserve chronology and context while remaining fully reproducible from open data.

## 2) Scope and Data

- Season scope: 2025-2026 NHL regular season.
- Source: public NHL Web API (`api-web.nhle.com`) schedule and play-by-play payloads.
- Storage: normalized SQLite schema (`games`, `events`, `shots`) with explicit event ordering.
- Replay mode: completed games replayed as simulated live flow (not a live NHL feed).

## 3) System Architecture

Pipeline stages:

1. Ingest schedule and game payloads.
2. Normalize event-level records and shot-attempt rows.
3. Persist into SQLite with stable indexing and timestamps.
4. Compute metrics for threat, pressure, and momentum.
5. Visualize in Streamlit and Plotly with playback controls.

Core modules:

- `src/ingest.py`: API collection.
- `src/clean.py`: transformation and DB loads.
- `src/playback.py`: event-index slicing for replay.
- `src/metrics.py`: analytics logic.
- `src/visuals.py`: chart construction.
- `app/streamlit_app.py`: interactive application.

## 4) Metric Definitions

### 4.1 Shot Quality (Threat Index)

Each shot attempt receives a unitless threat score in `[0,1]` from a geometric mapping of shot location to the nearest net (`x = +/-89 ft`, `y = 0` centerline). The model intentionally prioritizes interpretability:

- distance term (close attempts weighted higher),
- angle term (more central lanes weighted higher),
- lane term (mild center-slot emphasis).

Important: this is a relative threat proxy, not a calibrated goal probability.

### 4.2 High-Danger Flag

A shot is marked high-danger if all conditions hold:

- attempt is not blocked,
- distance to nearest net is `< 22 ft`,
- lateral offset is in the slot band: `abs(y) < 24 ft`.

This yields a binary, reproducible label suitable for cumulative tracking.

### 4.3 Blocked Shot Treatment

Blocked attempts are retained as pressure events but receive a down-weighted threat contribution to avoid overstating quality while preserving territorial pressure information.

### 4.4 Rolling and Cumulative Measures

- Cumulative threat: running total of shot-quality index by team.
- Rolling momentum windows: local pressure summaries over configurable game-time windows.
- Pressure ledger: windowed table including attempts, threat, high-danger, rebound chains, offensive-zone event share, possession-time proxy, and longest offensive-zone sequence.

### 4.5 Decayed Momentum Model

Momentum uses explicit temporal decay:

`M_t = exp(-delta_t / half_life) * M_(t-1) + impulse_t`

where `impulse_t` combines shot threat and pressure proxies (offensive-zone activity, entries, and entry-to-shot success) with fixed transparent weights.

## 5) Product Surface

The app exposes five analysis tabs:

- `Game Playback`: synchronized timeline replay and scoreboard context.
- `Momentum & Pressure`: decayed momentum and pressure ledger views.
- `Shot Map`: spatial threat visualization of attempts.
- `Insights`: automated key-moment extraction and process-vs-score summary.
- `Team Comparison`: side-by-side aggregate comparison for current playback position.

This structure links sequence diagnostics to interpretable team-level summaries.

## 6) Validation Philosophy

This project uses engineering validation rather than proprietary model benchmarking:

- deterministic transformations from raw payload to analytics outputs,
- transparent formulas and thresholds,
- reproducible replay state from event index and game time,
- explicit assumptions documented in code and app text.

Future extensions can overlay calibrated xG models while preserving current interpretability.

## 7) Limitations

- Public play-by-play lacks player tracking, puck trajectory, screens, and many pre-shot context features.
- Coordinate-based shot quality cannot capture full tactical nuance.
- Several contextual signals (entries, possession proxies) are inferred heuristics.
- Simulated playback mirrors logical event progression, not broadcast timing.

## 8) Roadmap

Planned improvements:

- optional calibrated xG integration on `(game_id, event_idx)`,
- richer sequence segmentation and swing diagnostics,
- stronger automated tests for metric invariants,
- deployment hardening and scalable data backends.

## 9) Conclusion

Replay-driven analytics provides a practical middle ground between static box-score reporting and full tracking-based modeling. By combining transparent metric construction with synchronized event playback, this system helps users evaluate game process in a way that is explainable, reproducible, and directly actionable for post-game analysis.
