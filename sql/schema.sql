-- NHL Game Playback Analytics — SQLite schema
-- Season scope: 2025–2026 regular season (game_type = 2 on NHL Web API; 1 = preseason)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    game_type INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    game_state TEXT,
    start_time_utc TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_abbr TEXT,
    away_abbr TEXT,
    home_score INTEGER,
    away_score INTEGER,
    venue TEXT,
    raw_json_path TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_games_season_date ON games(season, game_date);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(game_id),
    event_idx INTEGER NOT NULL,
    sort_order INTEGER,
    period INTEGER NOT NULL,
    game_seconds REAL NOT NULL,
    event_type TEXT NOT NULL,
    type_desc_key TEXT,
    team_id INTEGER,
    player_id INTEGER,
    x_coord REAL,
    y_coord REAL,
    zone_code TEXT,
    situation_code TEXT,
    strength_state TEXT,
    score_home INTEGER,
    score_away INTEGER,
    is_goal INTEGER NOT NULL DEFAULT 0,
    is_shot_attempt INTEGER NOT NULL DEFAULT 0,
    shot_type TEXT,
    UNIQUE (game_id, event_idx)
);

CREATE INDEX IF NOT EXISTS idx_events_game_idx ON events(game_id, event_idx);
CREATE INDEX IF NOT EXISTS idx_events_game_seconds ON events(game_id, game_seconds);

CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(game_id),
    event_idx INTEGER NOT NULL,
    team_id INTEGER,
    player_id INTEGER,
    period INTEGER NOT NULL,
    game_seconds REAL NOT NULL,
    shot_kind TEXT NOT NULL,
    shot_type TEXT,
    x_coord REAL,
    y_coord REAL,
    zone_code TEXT,
    situation_code TEXT,
    strength_state TEXT,
    score_home INTEGER,
    score_away INTEGER,
    is_goal INTEGER NOT NULL DEFAULT 0,
    is_blocked INTEGER NOT NULL DEFAULT 0,
    shot_quality REAL NOT NULL,
    high_danger INTEGER NOT NULL DEFAULT 0,
    UNIQUE (game_id, event_idx),
    FOREIGN KEY (game_id, event_idx) REFERENCES events(game_id, event_idx)
);

CREATE INDEX IF NOT EXISTS idx_shots_game ON shots(game_id, game_seconds);
