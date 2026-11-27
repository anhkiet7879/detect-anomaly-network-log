
CREATE TABLE IF NOT EXISTS log_windows (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    window_start_utc TIMESTAMP NOT NULL UNIQUE,
    epoch BIGINT NOT NULL,

    n_logs INTEGER NOT NULL DEFAULT 0,
    n_info INTEGER NOT NULL DEFAULT 0,
    n_warning INTEGER NOT NULL DEFAULT 0,
    n_severe INTEGER NOT NULL DEFAULT 0,
    n_error INTEGER NOT NULL DEFAULT 0,
    n_fatal INTEGER NOT NULL DEFAULT 0,
    n_failure INTEGER NOT NULL DEFAULT 0,
    n_other_level INTEGER NOT NULL DEFAULT 0,

    n_nodes INTEGER NOT NULL DEFAULT 0,
    n_components INTEGER NOT NULL DEFAULT 0,
    n_alert_lines INTEGER NOT NULL DEFAULT 0,

    is_alert_window BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_log_windows_time
ON log_windows (window_start_utc);

CREATE INDEX IF NOT EXISTS idx_log_windows_alert
ON log_windows (is_alert_window);
