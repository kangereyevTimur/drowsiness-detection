from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "results" / "diagnostics"
REPORTS_DIR = ROOT / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_session_csv() -> Path | None:
    if not DIAG_DIR.exists():
        return None
    candidates = sorted(
        [p for p in DIAG_DIR.glob("*.csv") if p.stat().st_size > 300],
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def generate_session_report_png(csv_path: Path | None = None) -> Path | None:
    """
    Строит сводный аналитический 4-панельный график сессии (PNG)
    и сохраняет его в results/reports/.
    """
    target_csv = csv_path if (csv_path and csv_path.exists()) else find_latest_session_csv()
    if target_csv is None or not target_csv.exists():
        return None

    df = pd.read_csv(target_csv)
    if len(df) < 5 or "timestamp_ms" not in df.columns:
        fallback = find_latest_session_csv()
        if fallback is None or fallback == target_csv or not fallback.exists():
            return None
        target_csv = fallback
        df = pd.read_csv(target_csv)
        if len(df) < 5 or "timestamp_ms" not in df.columns:
            return None

    t_sec = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
    # Защита от разрывов времени при спящем режиме
    dt = np.diff(t_sec, prepend=0.0)
    dt = np.where(dt > 1.0, 0.04, dt)
    t_sec = np.cumsum(dt)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Аналитический отчёт сессии мониторинга: {target_csv.name}", fontsize=14, fontweight="bold")

    # Панель 1: Drowsiness Score или Вероятности CNN
    if "score" in df.columns:
        axes[0].plot(t_sec, df["score"], color="#D93025", linewidth=1.8, label="Drowsiness Score (0..100)")
        axes[0].axhline(65.0, color="red", linestyle="--", alpha=0.7, label="Порог HIGH (65)")
        axes[0].axhline(35.0, color="orange", linestyle="--", alpha=0.7, label="Порог ELEVATED (35)")
        axes[0].fill_between(t_sec, df["score"], 0, color="#D93025", alpha=0.12)
        axes[0].set_ylim(-2, 105)
        axes[0].set_ylabel("Score")
        axes[0].set_title("1. Интегральный индекс сонливости (Drowsiness Score)")
    else:
        axes[0].plot(t_sec, df["left_cnn_prob"], label="Left CNN Prob", color="#1A73E8", alpha=0.85)
        axes[0].plot(t_sec, df["right_cnn_prob"], label="Right CNN Prob", color="#F9AB00", alpha=0.85)
        axes[0].set_ylabel("CNN Prob")
        axes[0].set_title("1. Вероятности закрытия глаз (EyeCNN)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Панель 2: PERCLOS (%) и Моргания
    if "perclos" in df.columns:
        axes[1].plot(t_sec, df["perclos"], color="#9334E6", linewidth=1.8, label="PERCLOS (%, окно 60с)")
        axes[1].axhline(28.0, color="red", linestyle=":", alpha=0.7, label="Критический PERCLOS (28%)")
        axes[1].axhline(15.0, color="orange", linestyle=":", alpha=0.7, label="Повышенный PERCLOS (15%)")
        axes[1].set_ylabel("PERCLOS (%)")
        axes[1].set_title("2. Динамика показателя PERCLOS во времени")
    else:
        axes[1].step(t_sec, df["both_closed"], color="#D93025", where="post", label="Оба глаза закрыты")
        axes[1].set_ylabel("Closed (0/1)")
        axes[1].set_title("2. Фиксация закрытия обоих глаз")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Панель 3: Степень раскрытия глаз EAR и зевание MAR
    axes[2].plot(t_sec, df["left_ear"], color="#00ACC1", alpha=0.85, label="Left Eye EAR")
    axes[2].plot(t_sec, df["right_ear"], color="#1E8E3E", alpha=0.85, label="Right Eye EAR")
    if "mar" in df.columns:
        axes[2].plot(t_sec, df["mar"], color="#E8710A", linewidth=1.5, label="Mouth MAR (Зевание)")
        axes[2].axhline(0.55, color="#E8710A", linestyle="--", alpha=0.5, label="Порог зевка MAR (0.55)")
    axes[2].set_ylabel("EAR / MAR")
    axes[2].set_title("3. Геометрические показатели: раскрытие глаз (EAR) и рта (MAR)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Панель 4: Положение головы (Head Pose) или Состояния
    if "pitch_deg" in df.columns:
        axes[3].plot(t_sec, df["pitch_deg"], color="#1A73E8", label="Pitch (Наклон вперёд/вниз, °)")
        axes[3].plot(t_sec, df["roll_deg"], color="#5F6368", alpha=0.75, label="Roll (Наклон к плечу, °)")
        axes[3].axhline(16.0, color="red", linestyle="--", alpha=0.6, label="Порог клевания носом (+16°)")
        axes[3].set_ylabel("Угол (°)")
        axes[3].set_title("4. Положение и наклон головы (Head Pose)")
    else:
        axes[3].step(t_sec, df["left_closed"], label="Left Closed", color="#1A73E8", alpha=0.7, where="post")
        axes[3].step(t_sec, df["right_closed"], label="Right Closed", color="#F9AB00", alpha=0.7, where="post")
        axes[3].set_ylabel("State (0/1)")
        axes[3].set_title("4. Состояния левого и правого глаза")
    axes[3].set_xlabel("Время наблюдения (секунды)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_path = REPORTS_DIR / f"report_{target_csv.stem}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

