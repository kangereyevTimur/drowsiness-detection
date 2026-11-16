import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --- Пути ---
ROOT = Path(__file__).resolve().parent.parent
FINETUNED_CNN_PATH = ROOT / "models" / "eye_cnn_finetuned.pth"
BASE_CNN_PATH = ROOT / "models" / "eye_cnn_best.pth"
CNN_PATH = FINETUNED_CNN_PATH if FINETUNED_CNN_PATH.exists() else BASE_CNN_PATH
LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"
LOGS_DIR = ROOT / "results" / "diagnostics"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Параметры алгоритма ---
DROWSINESS_THRESHOLD = 2.0  # сек непрерывного закрытия
EMA_ALPHA = 0.30            # коэффициент сглаживания вероятности закрытия
CLOSE_THRESHOLD = 0.65      # порог закрытия
OPEN_THRESHOLD = 0.38       # порог открытия (гистерезис)
MIN_CONFIRM_FRAMES = 3      # число кадров для подтверждения смены состояния
EAR_CALIBRATION_SECONDS = 3.5

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- MediaPipe Landmark Indices ---
# Анатомически правый глаз (на зеркальном экране — слева)
# Внешний угол: 33, внутренний угол: 133
# Пары вертикальных точек: (160, 144), (159, 145), (158, 153)
LEFT_EYE_LANDMARKS = {
    "outer": 33,
    "inner": 133,
    "top_bottom_pairs": [(160, 144), (159, 145), (158, 153)]
}

# Анатомически левый глаз (на зеркальном экране — справа)
# Внешний угол: 263, внутренний угол: 362
# Пары вертикальных точек: (387, 373), (386, 374), (385, 380)
RIGHT_EYE_LANDMARKS = {
    "outer": 263,
    "inner": 362,
    "top_bottom_pairs": [(387, 373), (386, 374), (385, 380)]
}


class EyeCNN(nn.Module):
    """Архитектура обученной нейросети классификации глаз."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cnn_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# CLAHE адаптивный выравниватель контраста (сглаживает боковые тени)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def calculate_ear(landmarks, eye_def, frame_w, frame_h):
    """
    Вычисляет Eye Aspect Ratio (EAR) по 3 вертикальным парам точек.
    EAR = ( ||p_top1 - p_bot1|| + ||p_top2 - p_bot2|| + ||p_top3 - p_bot3|| ) / (3 * ||outer - inner||)
    """
    p_out = np.array([landmarks[eye_def["outer"]].x * frame_w, landmarks[eye_def["outer"]].y * frame_h])
    p_in = np.array([landmarks[eye_def["inner"]].x * frame_w, landmarks[eye_def["inner"]].y * frame_h])

    horizontal = np.linalg.norm(p_out - p_in)
    if horizontal < 1.0:
        return float("nan")

    vertical_sum = 0.0
    for top_idx, bot_idx in eye_def["top_bottom_pairs"]:
        p_top = np.array([landmarks[top_idx].x * frame_w, landmarks[top_idx].y * frame_h])
        p_bot = np.array([landmarks[bot_idx].x * frame_w, landmarks[bot_idx].y * frame_h])
        vertical_sum += np.linalg.norm(p_top - p_bot)

    return float(vertical_sum / (3.0 * horizontal))


def crop_square_eye(frame, landmarks, eye_def, scale_factor=1.35):
    """
    Вырезает строго КВАДРАТНУЮ область глаза вокруг анатомического центра.
    Размер квадрата зависит от расстояния между уголками глаза (outer - inner),
    а НЕ от степени раскрытия век.
    Это предотвращает сплющивание и искажение пропорций при закрытии глаза.
    """
    h, w = frame.shape[:2]
    p_out = np.array([landmarks[eye_def["outer"]].x * w, landmarks[eye_def["outer"]].y * h])
    p_in = np.array([landmarks[eye_def["inner"]].x * w, landmarks[eye_def["inner"]].y * h])

    eye_width = np.linalg.norm(p_out - p_in)
    if eye_width < 10:
        return None

    # Центр глаза — середина между уголками
    center_x = int((p_out[0] + p_in[0]) / 2.0)
    center_y = int((p_out[1] + p_in[1]) / 2.0)

    # Квадратный размер рамки
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

    # Преобразуем в PIL (RGB -> Grayscale через cnn_transform, как при обучении)
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_roi)
    tensor = cnn_transform(pil_img)

    return (x1, y1, x2, y2), tensor, roi


class StabilizedEyeTracker:
    """Управляет состоянием одного глаза: калибровка, разделение прищура и закрытия, гистерезис."""
    def __init__(self, name):
        self.name = name
        self.ema_prob = None
        self.closed = False
        self.is_squinting = False
        self.pending_count = 0
        self.baseline_ear = None
        self.ear_calibration_samples = []

    def calibrate(self, ear):
        # Собираем только устойчивые значения нормального открытого глаза
        if np.isfinite(ear) and 0.18 <= ear <= 0.40:
            self.ear_calibration_samples.append(ear)

    def finalize_calibration(self):
        if len(self.ear_calibration_samples) >= 10:
            med = float(np.median(self.ear_calibration_samples))
            # Физиологическое ограничение: нормальный baseline взрослого перед монитором 0.25 - 0.33
            self.baseline_ear = float(np.clip(med, 0.24, 0.33))
        elif len(self.ear_calibration_samples) > 0:
            self.baseline_ear = float(np.clip(np.mean(self.ear_calibration_samples), 0.24, 0.33))
        else:
            self.baseline_ear = 0.28  # эталонный baseline по умолчанию

    def update(self, cnn_closed_prob, ear, is_calibrating):
        if is_calibrating:
            self.calibrate(ear)
            return False, False, cnn_closed_prob, float("nan")

        if self.baseline_ear is None:
            self.finalize_calibration()

        safe_ear = ear if np.isfinite(ear) else 0.25
        ratio = safe_ear / self.baseline_ear

        # --- Распознавание прищура (Squinting) vs Закрытия (Closed) ---
        # 1. При полном закрытии веки физически смыкаются: EAR падает < 0.09.
        # 2. При прищуре веки сближены (EAR 0.12 - 0.22), но CNN видит зрачок (cnn_closed_prob < 0.35).
        # 3. Глаз считается закрытым только при согласованном подтверждении!
        
        self.is_squinting = False

        if cnn_closed_prob < 0.55:
            # Дообученная CNN видит открытый или прищуренный глаз
            if safe_ear < 0.19 or ratio < 0.68:
                self.is_squinting = True
            raw_closed = False
            fused_prob = cnn_closed_prob * 0.7

        elif cnn_closed_prob >= 0.65 and (safe_ear < 0.18 or ratio < 0.62):
            # И дообученная CNN, и геометрия подтверждают закрытие
            raw_closed = True
            fused_prob = 0.8 * cnn_closed_prob + 0.2 * (1.0 - min(1.0, ratio))

        elif cnn_closed_prob >= 0.85:
            # Высокая уверенность дообученной CNN в закрытии
            raw_closed = True
            fused_prob = cnn_closed_prob

        else:
            # Пограничное состояние (0.55 <= cnn_closed_prob < 0.65) -> трактуем как прищур
            self.is_squinting = True
            raw_closed = False
            fused_prob = cnn_closed_prob

        # Экспоненциальное сглаживание
        if self.ema_prob is None:
            self.ema_prob = fused_prob
        else:
            self.ema_prob = EMA_ALPHA * fused_prob + (1.0 - EMA_ALPHA) * self.ema_prob

        # Временной гистерезис: подтверждение 3 кадрами
        target_state = raw_closed
        if target_state != self.closed:
            self.pending_count += 1
            if self.pending_count >= MIN_CONFIRM_FRAMES:
                self.closed = target_state
                self.pending_count = 0
        else:
            self.pending_count = 0

        return self.closed, self.is_squinting, self.ema_prob, ratio


def main():
    if not CNN_PATH.exists() or not LANDMARKER_PATH.exists():
        raise FileNotFoundError("Модели не найдены в models/")

    print(f"Запуск детектора v4 на устройстве: {DEVICE}")
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
        raise RuntimeError("Не удалось открыть веб-камеру. Проверьте разрешения в macOS.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    trackers = [
        StabilizedEyeTracker("Left_Eye"),
        StabilizedEyeTracker("Right_Eye"),
    ]

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"diagnostic_{session_id}.csv"
    csv_file = open(log_file, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_ms", "left_cnn_prob", "left_ear", "left_ratio", "left_closed",
        "right_cnn_prob", "right_ear", "right_ratio", "right_closed", "both_closed"
    ])

    closed_start_time = None
    calibration_start = time.monotonic()
    app_start = time.monotonic()
    last_timestamp = -1

    print("\n=== Детектор сонливости v4 ===")
    print("1. В первые 3.5 секунды калибровки держите глаза открытыми и смотрите в камеру.")
    print(f"2. Логирование диагностики ведется в: {log_file}")
    print("Клавиши: [Q] - выход, [R] - повторная калибровка, [S] - сохранить скриншот\n")

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.monotonic()
                ts = max(int((now - app_start) * 1000), last_timestamp + 1)
                last_timestamp = ts

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect_for_video(mp_image, ts)

                is_calibrating = (now - calibration_start) < EAR_CALIBRATION_SECONDS
                detected = False
                both_closed = False

                left_info = {}
                right_info = {}

                if detection_result.face_landmarks:
                    lm = detection_result.face_landmarks[0]
                    detected = True

                    left_crop = crop_square_eye(frame, lm, LEFT_EYE_LANDMARKS)
                    right_crop = crop_square_eye(frame, lm, RIGHT_EYE_LANDMARKS)

                    if left_crop is not None and right_crop is not None:
                        # Пакетный инференс CNN для обоих глаз
                        batch_tensor = torch.stack([left_crop[1], right_crop[1]]).to(DEVICE)
                        with torch.inference_mode():
                            outputs = model(batch_tensor)
                            # Индекс 0 — CLOSED в train_dataset.class_to_idx
                            closed_probs = torch.softmax(outputs, dim=1)[:, 0].cpu().tolist()

                        left_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, w, h)
                        right_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, w, h)

                        l_closed, l_squint, l_ema, l_ratio = trackers[0].update(closed_probs[0], left_ear, is_calibrating)
                        r_closed, r_squint, r_ema, r_ratio = trackers[1].update(closed_probs[1], right_ear, is_calibrating)

                        left_info = {"prob": closed_probs[0], "ear": left_ear, "ratio": l_ratio, "closed": l_closed}
                        right_info = {"prob": closed_probs[1], "ear": right_ear, "ratio": r_ratio, "closed": r_closed}

                        # Отрисовка рамок вокруг глаз
                        for crop_data, eye_name, is_c, is_sq, prob, ear_val in [
                            (left_crop, "L", l_closed, l_squint, closed_probs[0], left_ear),
                            (right_crop, "R", r_closed, r_squint, closed_probs[1], right_ear)
                        ]:
                            box = crop_data[0]
                            if is_c:
                                color = (0, 70, 255)      # Красный: закрыт
                                state_str = "CLOSED"
                            elif is_sq:
                                color = (0, 215, 255)     # Жёлтый: прищур (бодрствует!)
                                state_str = "SQUINT"
                            else:
                                color = (50, 210, 80)     # Зелёный: открыт
                                state_str = "OPEN"

                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(
                                frame,
                                f"{eye_name}: {state_str} CNN:{prob:.2f} EAR:{ear_val:.2f}",
                                (box[0], max(20, box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA
                            )

                        # Отображение превью кропов в верхнем правом углу
                        preview_l = cv2.resize(left_crop[2], (70, 70))
                        preview_r = cv2.resize(right_crop[2], (70, 70))
                        frame[15:85, w - 160:w - 90] = preview_l
                        frame[15:85, w - 80:w - 10] = preview_r
                        cv2.rectangle(frame, (w - 160, 15), (w - 90, 85), (200, 200, 200), 1)
                        cv2.rectangle(frame, (w - 80, 15), (w - 10, 85), (200, 200, 200), 1)
                        cv2.putText(frame, "L Crop", (w - 155, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                        cv2.putText(frame, "R Crop", (w - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                        both_closed = l_closed and r_closed

                        # Запись в CSV
                        csv_writer.writerow([
                            ts,
                            f"{closed_probs[0]:.4f}", f"{left_ear:.4f}", f"{l_ratio:.4f}", int(l_closed),
                            f"{closed_probs[1]:.4f}", f"{right_ear:.4f}", f"{r_ratio:.4f}", int(r_closed),
                            int(both_closed)
                        ])

                # Логика уведомления о сонливости
                if not detected:
                    closed_start_time = None
                    cv2.putText(frame, "Face not detected", (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 160, 255), 2)
                elif is_calibrating:
                    closed_start_time = None
                    rem = max(0.0, EAR_CALIBRATION_SECONDS - (now - calibration_start))
                    cv2.putText(frame, f"CALIBRATING EAR... Look straight ({rem:.1f}s)", (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
                elif both_closed:
                    if closed_start_time is None:
                        closed_start_time = now
                    dur = now - closed_start_time
                    cv2.putText(frame, f"Eyes CLOSED: {dur:.1f}s", (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 70, 255), 2)
                    if dur >= DROWSINESS_THRESHOLD:
                        cv2.rectangle(frame, (15, 65), (420, 125), (0, 0, 180), -1)
                        cv2.putText(frame, "DROWSINESS ALERT!", (25, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                else:
                    closed_start_time = None
                    cv2.putText(frame, "Active (Eyes OPEN)", (25, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 210, 80), 2)

                # Информация о порогах внизу экрана
                base_l = trackers[0].baseline_ear or 0.0
                base_r = trackers[1].baseline_ear or 0.0
                cv2.putText(frame, f"Base EAR -> L: {base_l:.2f} | R: {base_r:.2f}  [Q: Exit | R: Recalibrate]",
                            (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

                cv2.imshow("Drowsiness Detection v4 - Stabilized Square Crop & Symmetric EAR", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    calibration_start = time.monotonic()
                    trackers[0].ear_calibration_samples.clear()
                    trackers[1].ear_calibration_samples.clear()
                    trackers[0].baseline_ear = None
                    trackers[1].baseline_ear = None
                    closed_start_time = None
                    print("Запущена повторная калибровка EAR...")
                elif key == ord("s"):
                    shot_path = ROOT / "results" / f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(str(shot_path), frame)
                    print(f"Снимок сохранен: {shot_path}")

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nДиагностические данные успешно сохранены в: {log_file}")


if __name__ == "__main__":
    main()

