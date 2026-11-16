import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_log(csv_path: Path):
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден.")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("Файл лога пуст.")
        return

    # Перевод времени в секунды от старта
    df["time_sec"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0

    print(f"\n=== Анализ диагностического лога: {csv_path.name} ===")
    print(f"Всего кадров: {len(df)}")
    print(f"Длительность записи: {df['time_sec'].iloc[-1]:.1f} сек")

    # Статистика вероятностей CNN
    print("\n--- Статистика вероятности закрытия CNN (Closed Prob) ---")
    print(f"Левый глаз  (L): среднее = {df['left_cnn_prob'].mean():.3f}, медиана = {df['left_cnn_prob'].median():.3f}, std = {df['left_cnn_prob'].std():.3f}")
    print(f"Правый глаз (R): среднее = {df['right_cnn_prob'].mean():.3f}, медиана = {df['right_cnn_prob'].median():.3f}, std = {df['right_cnn_prob'].std():.3f}")
    diff_cnn = np.abs(df["left_cnn_prob"] - df["right_cnn_prob"])
    print(f"Среднее расхождение CNN между глазами (|L - R|): {diff_cnn.mean():.3f}")

    # Статистика EAR
    print("\n--- Статистика Eye Aspect Ratio (EAR) ---")
    print(f"Левый глаз  (L EAR): среднее = {df['left_ear'].mean():.3f}, std = {df['left_ear'].std():.3f}")
    print(f"Правый глаз (R EAR): среднее = {df['right_ear'].mean():.3f}, std = {df['right_ear'].std():.3f}")
    diff_ear = np.abs(df["left_ear"] - df["right_ear"])
    print(f"Среднее расхождение EAR между глазами: {diff_ear.mean():.3f}")

    # Расхождение состояний
    mismatch_states = (df["left_closed"] != df["right_closed"]).mean() * 100.0
    print(f"\nКадры с несовпадающим состоянием глаз (L != R): {mismatch_states:.2f}%")

    # Построение графиков
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. CNN Probabilities
    axes[0].plot(df["time_sec"], df["left_cnn_prob"], label="Left Eye CNN (Closed Prob)", color="blue", alpha=0.8)
    axes[0].plot(df["time_sec"], df["right_cnn_prob"], label="Right Eye CNN (Closed Prob)", color="orange", alpha=0.8)
    axes[0].axhline(0.65, color="red", linestyle="--", label="Close Threshold (0.65)")
    axes[0].axhline(0.38, color="green", linestyle="--", label="Open Threshold (0.38)")
    axes[0].set_ylabel("CNN Probability")
    axes[0].set_title("Сравнение вероятностей нейросети для левого и правого глаза")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. EAR
    axes[1].plot(df["time_sec"], df["left_ear"], label="Left Eye EAR", color="cyan", alpha=0.8)
    axes[1].plot(df["time_sec"], df["right_ear"], label="Right Eye EAR", color="magenta", alpha=0.8)
    axes[1].set_ylabel("EAR value")
    axes[1].set_title("Eye Aspect Ratio (EAR) по кадрам")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 3. Результирующие состояния
    axes[2].step(df["time_sec"], df["left_closed"], label="Left Closed", color="blue", alpha=0.7, where="post")
    axes[2].step(df["time_sec"], df["right_closed"], label="Right Closed", color="orange", alpha=0.7, where="post")
    axes[2].step(df["time_sec"], df["both_closed"], label="Both Closed", color="red", linewidth=2, where="post")
    axes[2].set_ylabel("State (0=Open, 1=Closed)")
    axes[2].set_xlabel("Время (секунды)")
    axes[2].set_title("Состояние глаз во времени")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot = csv_path.with_suffix(".png")
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"\nГрафик сохранен в: {output_plot}\n")


def main():
    parser = argparse.ArgumentParser(description="Анализ диагностических логов детектора глаз")
    parser.add_argument("log_file", nargs="?", help="Путь к CSV файлу лога (если не указан, берется последний)")
    args = parser.parse_args()

    logs_dir = Path("results/diagnostics")
    if args.log_file:
        target = Path(args.log_file)
    else:
        files = sorted(logs_dir.glob("diagnostic_*.csv"))
        if not files:
            print(f"В папке {logs_dir} еще нет диагностических логов. Запустите сначала src/detect_drowsiness_v4.py")
            return
        target = files[-1]

    analyze_log(target)


if __name__ == "__main__":
    main()

