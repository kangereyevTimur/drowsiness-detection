import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Пути
ROOT = Path(__file__).resolve().parent.parent
LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"
DATASET_DIR = ROOT / "data" / "webcam_dataset"

LEFT_EYE_LANDMARKS = {"outer": 33, "inner": 133}
RIGHT_EYE_LANDMARKS = {"outer": 263, "inner": 362}


def crop_square_eye_bgr(frame, landmarks, eye_def, scale_factor=1.35):
    h, w = frame.shape[:2]
    p_out = np.array([landmarks[eye_def["outer"]].x * w, landmarks[eye_def["outer"]].y * h])
    p_in = np.array([landmarks[eye_def["inner"]].x * w, landmarks[eye_def["inner"]].y * h])

    eye_width = np.linalg.norm(p_out - p_in)
    if eye_width < 10:
        return None

    center_x = int((p_out[0] + p_in[0]) / 2.0)
    center_y = int((p_out[1] + p_in[1]) / 2.0)

    box_size = int(eye_width * scale_factor)
    half_size = box_size // 2

    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(w, center_x + half_size)
    y2 = min(h, center_y + half_size)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    return cv2.resize(roi, (64, 64))


def main():
    parser = argparse.ArgumentParser(description="Сбор собственного датасета глаз с веб-камеры")
    parser.add_argument("--subject", type=str, default="user01", help="Идентификатор участника (например, user01)")
    args = parser.parse_args()

    for category in ["open", "closed"]:
        (DATASET_DIR / category).mkdir(parents=True, exist_ok=True)

    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Файл {LANDMARKER_PATH} не найден.")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n=== Сбор собственного датасета глаз ===")
    print("Управление режимом сохранения:")
    print("  [O] - зажать / переключить запись: ОТКРЫТЫЕ глаза (включая прищур)")
    print("  [C] - зажать / переключить запись: ЗАКРЫТЫЕ глаза")
    print("  [Пробел] - пауза записи")
    print("  [Q] - выход\n")

    current_mode = "idle"  # idle, open, closed
    counters = {"open": 0, "closed": 0}
    start_time = time.monotonic()
    last_timestamp = -1

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.monotonic()
                ts = max(int((now - start_time) * 1000), last_timestamp + 1)
                last_timestamp = ts

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    left_crop = crop_square_eye_bgr(frame, lm, LEFT_EYE_LANDMARKS)
                    right_crop = crop_square_eye_bgr(frame, lm, RIGHT_EYE_LANDMARKS)

                    if left_crop is not None and right_crop is not None:
                        # Если включен режим записи, сохраняем кадры
                        if current_mode in ["open", "closed"]:
                            uid = int(time.time() * 1000)
                            f_l = DATASET_DIR / current_mode / f"{args.subject}_L_{uid}.png"
                            f_r = DATASET_DIR / current_mode / f"{args.subject}_R_{uid}.png"
                            cv2.imwrite(str(f_l), left_crop)
                            cv2.imwrite(str(f_r), right_crop)
                            counters[current_mode] += 2

                        # Превью
                        frame[15:79, w - 150:w - 86] = left_crop
                        frame[15:79, w - 75:w - 11] = right_crop

                # UI статус
                mode_color = (0, 255, 0) if current_mode == "open" else ((0, 0, 255) if current_mode == "closed" else (180, 180, 180))
                cv2.putText(frame, f"MODE: {current_mode.upper()}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
                cv2.putText(frame, f"Saved -> Open: {counters['open']} | Closed: {counters['closed']}",
                            (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
                cv2.putText(frame, "[O] Rec Open/Squint  [C] Rec Closed  [Space] Pause  [Q] Exit",
                            (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow("Webcam Eye Dataset Collector", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("o"):
                    current_mode = "open"
                elif key == ord("c"):
                    current_mode = "closed"
                elif key == ord(" "):
                    current_mode = "idle"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nСбор завершён! Сохранено: Open = {counters['open']}, Closed = {counters['closed']}")
        print(f"Кадры находятся в: {DATASET_DIR}")


if __name__ == "__main__":
    main()

