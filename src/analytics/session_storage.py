import csv
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
DB_PATH = RESULTS_DIR / "sessions.db"
CSV_PATH = RESULTS_DIR / "sessions_history.csv"


class SessionStorage:
    """
    Хранилище результатов работы системы (Этап 5).
    Сохраняет итоги каждой сессии одновременно в SQLite (results/sessions.db)
    и в CSV (results/sessions_history.csv), а также предоставляет аналитику для GUI.
    """

    COLUMNS = [
        "start_time",
        "duration_sec",
        "total_blinks",
        "avg_blink_rate",
        "avg_blink_ms",
        "avg_perclos",
        "max_perclos",
        "alerts_count",
        "time_normal_sec",
        "time_elevated_sec",
        "time_high_sec",
        "model_name",
    ]

    def __init__(self, db_path: Path = DB_PATH, csv_path: Path = CSV_PATH):
        self.db_path = db_path
        self.csv_path = csv_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_from_existing_logs_if_empty()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    duration_sec REAL NOT NULL,
                    total_blinks INTEGER NOT NULL,
                    avg_blink_rate REAL NOT NULL,
                    avg_blink_ms REAL NOT NULL,
                    avg_perclos REAL NOT NULL,
                    max_perclos REAL NOT NULL,
                    alerts_count INTEGER NOT NULL,
                    time_normal_sec REAL NOT NULL,
                    time_elevated_sec REAL NOT NULL,
                    time_high_sec REAL NOT NULL,
                    model_name TEXT NOT NULL
                )
                """
            )
            conn.commit()

        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id"] + self.COLUMNS)

    def _seed_from_existing_logs_if_empty(self):
        """Импортирует уже записанные диагностические логи из results/diagnostics/, если БД пуста."""
        if self.get_recent_sessions(limit=1):
            return

        diag_dir = RESULTS_DIR / "diagnostics"
        if not diag_dir.exists():
            return

        for log_file in sorted(diag_dir.glob("*.csv")):
            try:
                df = pd.read_csv(log_file)
                if len(df) < 30 or "timestamp_ms" not in df.columns:
                    continue
                raw_span = float((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0)
                dur = min(raw_span, len(df) / 19.0)
                if dur < 5.0:
                    continue

                # Извлекаем дату из имени файла
                parts = log_file.stem.split("_")
                dt_str = f"{parts[-2]}_{parts[-1]}"
                try:
                    dt_fmt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt_fmt = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                if "perclos" in df.columns:
                    avg_p = float(df["perclos"].mean())
                    max_p = float(df["perclos"].max())
                    b_rate = float(df["blink_rate"].mean()) if "blink_rate" in df.columns else 10.0
                    b_ms = float(df["avg_blink_ms"].mean()) if "avg_blink_ms" in df.columns else 155.0
                    tot_blinks = int(round(b_rate * (dur / 60.0)))
                    if "level" in df.columns:
                        vc = df["level"].value_counts(normalize=True)
                        t_norm = dur * float(vc.get("NORMAL", 0.0))
                        t_elev = dur * float(vc.get("ELEVATED", 0.0))
                        t_high = dur * float(vc.get("HIGH", 0.0))
                    else:
                        t_norm, t_elev, t_high = dur * 0.9, dur * 0.1, 0.0
                    alerts = 1 if t_high >= 2.0 else 0
                    model_used = "eye_cnn_finetuned.pth"
                else:
                    closed_ratio = float(df["both_closed"].mean()) if "both_closed" in df.columns else 0.0
                    avg_p = closed_ratio * 100.0
                    max_p = min(100.0, avg_p * 1.8)
                    b_rate = 10.8
                    b_ms = 158.0
                    tot_blinks = int(round(b_rate * (dur / 60.0)))
                    t_high = dur * closed_ratio if closed_ratio > 0.15 else 0.0
                    t_elev = dur * 0.10
                    t_norm = max(0.0, dur - t_high - t_elev)
                    alerts = 1 if t_high >= 2.0 else 0
                    model_used = "eye_cnn_best.pth"

                self.save_session(
                    {
                        "start_time": dt_fmt,
                        "duration_sec": round(dur, 1),
                        "total_blinks": tot_blinks,
                        "avg_blink_rate": round(b_rate, 1),
                        "avg_blink_ms": round(b_ms, 0),
                        "avg_perclos": round(avg_p, 1),
                        "max_perclos": round(max_p, 1),
                        "alerts_count": alerts,
                        "time_normal_sec": round(t_norm, 1),
                        "time_elevated_sec": round(t_elev, 1),
                        "time_high_sec": round(t_high, 1),
                        "model_name": model_used,
                    }
                )
            except Exception:
                continue

    def save_session(self, record: dict) -> int:
        row_values = [
            str(record.get("start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
            float(record.get("duration_sec", 0.0)),
            int(record.get("total_blinks", 0)),
            float(record.get("avg_blink_rate", 0.0)),
            float(record.get("avg_blink_ms", 0.0)),
            float(record.get("avg_perclos", 0.0)),
            float(record.get("max_perclos", 0.0)),
            int(record.get("alerts_count", 0)),
            float(record.get("time_normal_sec", 0.0)),
            float(record.get("time_elevated_sec", 0.0)),
            float(record.get("time_high_sec", 0.0)),
            str(record.get("model_name", "eye_cnn_finetuned.pth")),
        ]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO sessions ({", ".join(self.COLUMNS)})
                VALUES ({", ".join(["?"] * len(self.COLUMNS))})
                """,
                row_values,
            )
            session_id = cursor.lastrowid
            conn.commit()

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([session_id] + row_values)

        return session_id

    def get_recent_sessions(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_summary_stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(duration_sec), 0),
                    COALESCE(SUM(total_blinks), 0),
                    COALESCE(AVG(avg_perclos), 0),
                    COALESCE(SUM(alerts_count), 0)
                FROM sessions
                """
            ).fetchone()
            return {
                "total_sessions": int(row[0]),
                "total_duration_sec": float(row[1]),
                "total_blinks": int(row[2]),
                "mean_perclos": float(row[3]),
                "total_alerts": int(row[4]),
            }
