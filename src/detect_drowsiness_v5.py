import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.drowsiness_detector import DrowsinessEvaluator
from src.detection.eye_detector import (
    BASE_CNN_PATH,
    DEVICE,
    FINETUNED_CNN_PATH,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EyeCNN,
    StabilizedEyeTracker,
    calculate_ear,
    crop_square_eye,
)

CNN_PATH = FINETUNED_CNN_PATH if FINETUNED_CNN_PATH.exists() else BASE_CNN_PATH
LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"
LOGS_DIR = ROOT / "results" / "diagnostics"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_SECONDS = 3.5


def draw_hud(frame, status: dict, base_l: float, base_r: float, is_calibrating: bool, rem_cal: float):
    h, w = frame.shape[:2]

    # Полупрозрачная левая панель телеметрии
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (395, 320), (22, 24, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    level = status["level"]
    if is_calibrating:
        lvl_color = (0, 220, 255)
        lvl_text = f"CALIBRATING ({rem_cal:.1f}s)"
    elif level == DrowsinessEvaluator.LEVEL_HIGH:
        lvl_color = (40, 40, 255)
        lvl_text = "HIGH DROWSINESS!"
    elif level == DrowsinessEvaluator.LEVEL_ELEVATED:
        lvl_color = (0, 180, 255)
        lvl_text = "ELEVATED DROWSINESS"
    else:
        lvl_color = (60, 220, 80)
        lvl_text = "NORMAL STATE"

    # Заголовок состояния и индекс сонливости
    cv2.putText(frame, lvl_text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, lvl_color, 2, cv2.LINE_AA)
    score = status["score"]
    cv2.putText(frame, f"Drowsiness Score: {score:4.1f} / 100", (24, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Полоса прогресса Score
    bar_x, bar_y, bar_w, bar_h = 24, 76, 340, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    fill_w = int(bar_w * min(1.0, max(0.0, score / 100.0)))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), lvl_color, -1)

    # Семь ключевых показателей
    pose = status["pose"]
    yawn = status["yawn"]
    metrics_lines = [
        f"1. EAR (Mean)      : {status['mean_ear']:.2f} (L0:{base_l:.2f} R0:{base_r:.2f})",
        f"2. PERCLOS (60s)   : {status['perclos']:5.1f}%",
        f"3. Blink Rate      : {status['blink_rate_per_min']:4.1f} / min (Total: {status['total_blinks']})",
        f"4. Blink Duration  : {status['avg_blink_duration_ms']:4.0f} ms (Last: {status.get('last_blink_duration_ms', 0):.0f} ms)",
        f"5. Closure Time    : {status['closure_duration_sec']:4.1f} s",
        f"6. Head Pose       : P:{pose['pitch_deg']:+4.0f} Y:{pose['yaw_deg']:+4.0f} R:{pose['roll_deg']:+4.0f} [{pose['pose_status']}]",
        f"7. Yawning (MAR)   : {yawn['mar']:.2f} (Yawns: {yawn['total_yawns']}) [{yawn['status']}]",
    ]

    y_pos = 110
    for line in metrics_lines:
        cv2.putText(frame, line, (24, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        y_pos += 25

    # Причины повышения уровня сонливости
    reasons = status.get("reasons", [])
    if reasons:
        reason_str = " | ".join(reasons[:2])
        cv2.putText(frame, f"Triggers: {reason_str}", (24, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Alerts: {status['alerts_count']}  [Q: Exit | R: Recalibrate]", (24, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Баннер тревоги при высокой сонливости
    if level == DrowsinessEvaluator.LEVEL_HIGH and not is_calibrating:
        cv2.rectangle(frame, (w // 2 - 220, 15), (w // 2 + 220, 72), (0, 0, 190), -1)
        cv2.rectangle(frame, (w // 2 - 220, 15), (w // 2 + 220, 72), (255, 255, 255), 2)
        cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!", (w // 2 - 195, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    if not CNN_PATH.exists() or not LANDMARKER_PATH.exists():
        raise FileNotFoundError("Модели не найдены в папке models/")

    print(f"Устройство: {DEVICE}")
    print(f"Используемая модель CNN: {CNN_PATH.name}")

    model = EyeCNN().to(DEVICE)
    model.load_state_dict(torch.load(CNN_PATH, map_location="cpu", weights_only=True))
    model.eval()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру. Проверьте разрешения macOS.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    trackers = [StabilizedEyeTracker("Left"), StabilizedEyeTracker("Right")]
    evaluator = DrowsinessEvaluator(window_seconds=60.0)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"drowsiness_v5_{session_id}.csv"
    csv_file = open(log_file, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_ms", "level", "score", "perclos", "blink_rate", "avg_blink_ms",
        "closure_sec", "left_cnn_prob", "right_cnn_prob", "left_ear", "right_ear",
        "left_closed", "right_closed", "both_closed", "pitch_deg", "yaw_deg", "roll_deg"
    ])

    app_start = time.monotonic()
    calibration_start = app_start
    last_timestamp = -1

    print("\n=== Детектор сонливости v5 (6 признаков + уровни сонливости) ===")
    print("Первые 3.5 секунды смотрите прямо в камеру с открытыми глазами.")
    print("Клавиши: [Q] - выход, [R] - повторная калибровка\n")

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.monotonic()
                elapsed_sec = now - calibration_start
                is_calibrating = elapsed_sec < CALIBRATION_SECONDS
                rem_cal = max(0.0, CALIBRATION_SECONDS - elapsed_sec)

                ts = max(int((now - app_start) * 1000), last_timestamp + 1)
                last_timestamp = ts

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    left_crop = crop_square_eye(frame, lm, LEFT_EYE_LANDMARKS)
                    right_crop = crop_square_eye(frame, lm, RIGHT_EYE_LANDMARKS)

                    if left_crop is not None and right_crop is not None:
                        batch = torch.stack([left_crop[1], right_crop[1]]).to(DEVICE)
                        with torch.inference_mode():
                            probs = torch.softmax(model(batch), dim=1)[:, 0].cpu().tolist()

                        l_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, w, h)
                        r_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, w, h)

                        l_closed, l_sq, _, l_ratio = trackers[0].update(probs[0], l_ear, is_calibrating)
                        r_closed, r_sq, _, r_ratio = trackers[1].update(probs[1], r_ear, is_calibrating)

                        both_closed = l_closed and r_closed
                        mean_ear = float(np.nanmean([l_ear, r_ear]))
                        mean_ratio = float(np.nanmean([l_ratio, r_ratio]))

                        status = evaluator.update(
                            now_sec=now,
                            elapsed_session_sec=max(0.1, elapsed_sec - CALIBRATION_SECONDS),
                            is_calibrating=is_calibrating,
                            both_closed=both_closed,
                            mean_ear=mean_ear,
                            mean_ear_ratio=mean_ratio,
                            landmarks=lm,
                            frame_w=w,
                            frame_h=h,
                        )

                        # Отрисовка рамок вокруг глаз
                        for crop_data, name, is_c, is_s, prob, ear_v in [
                            (left_crop, "L", l_closed, l_sq, probs[0], l_ear),
                            (right_crop, "R", r_closed, r_sq, probs[1], r_ear),
                        ]:
                            box = crop_data[0]
                            if is_c:
                                col, st = (0, 70, 255), "CLOSED"
                            elif is_s:
                                col, st = (0, 215, 255), "SQUINT"
                            else:
                                col, st = (50, 210, 80), "OPEN"
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), col, 2)
                            cv2.putText(frame, f"{name}:{st} P:{prob:.2f} EAR:{ear_v:.2f}",
                                        (box[0], max(20, box[1] - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

                        # Превью кропов
                        frame[15:85, w - 160:w - 90] = cv2.resize(left_crop[2], (70, 70))
                        frame[15:85, w - 80:w - 10] = cv2.resize(right_crop[2], (70, 70))
                        cv2.rectangle(frame, (w - 160, 15), (w - 90, 85), (200, 200, 200), 1)
                        cv2.rectangle(frame, (w - 80, 15), (w - 10, 85), (200, 200, 200), 1)

                        draw_hud(
                            frame, status,
                            trackers[0].baseline_ear or 0.0,
                            trackers[1].baseline_ear or 0.0,
                            is_calibrating, rem_cal
                        )

                        pose = status["pose"]
                        csv_writer.writerow([
                            ts, status["level"], f"{status['score']:.2f}", f"{status['perclos']:.2f}",
                            f"{status['blink_rate_per_min']:.2f}", f"{status['avg_blink_duration_ms']:.1f}",
                            f"{status['closure_duration_sec']:.2f}", f"{probs[0]:.4f}", f"{probs[1]:.4f}",
                            f"{l_ear:.4f}", f"{r_ear:.4f}", int(l_closed), int(r_closed), int(both_closed),
                            f"{pose['pitch_deg']:.1f}", f"{pose['yaw_deg']:.1f}", f"{pose['roll_deg']:.1f}"
                        ])
                else:
                    cv2.putText(frame, "Face not detected", (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2)

                cv2.imshow("Drowsiness Detection v5 - Multi-Feature System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    calibration_start = time.monotonic()
                    for tr in trackers:
                        tr.reset()
                    evaluator.reset()
                    print("Запущена повторная калибровка...")

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nЛог сессии сохранён в: {log_file}")


if __name__ == "__main__":
    main()

