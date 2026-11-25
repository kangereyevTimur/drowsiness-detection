
import cv2
import time
import torch
import torch.nn as nn
import mediapipe as mp

from pathlib import Path
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

CNN_PATH = Path("models/eye_cnn_best.pth")
LANDMARKER_PATH = Path("models/face_landmarker.task")

DROWSINESS_THRESHOLD = 2.0

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Устройство:", DEVICE)


# ==========================================
# 2. АРХИТЕКТУРА НАШЕЙ CNN
# ==========================================

class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# 3. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ
# ==========================================

model = EyeCNN()

model.load_state_dict(
    torch.load(
        CNN_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model = model.to(DEVICE)
model.eval()

print("CNN успешно загружена")


# ==========================================
# 4. ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ==========================================
# 5. MEDIAPIPE FACE LANDMARKER
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)


# ==========================================
# 6. ТОЧКИ ВОКРУГ ГЛАЗ
# ==========================================

LEFT_EYE = [33, 133, 159, 145, 160, 144]
RIGHT_EYE = [362, 263, 386, 374, 387, 373]


# ==========================================
# 7. ФУНКЦИЯ РАСПОЗНАВАНИЯ ГЛАЗА
# ==========================================

def classify_eye(frame, landmarks, indices):
    height, width = frame.shape[:2]

    points = [
        (
            int(landmarks[i].x * width),
            int(landmarks[i].y * height)
        )
        for i in indices
    ]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    eye_width = max(xs) - min(xs)
    eye_height = max(ys) - min(ys)

    margin_x = int(eye_width * 0.25)
    margin_y = max(
        int(eye_height * 0.8),
        int(eye_width * 0.15)
    )

    x1 = max(0, min(xs) - margin_x)
    y1 = max(0, min(ys) - margin_y)
    x2 = min(width, max(xs) + margin_x)
    y2 = min(height, max(ys) + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    eye_image = frame[y1:y2, x1:x2]

    if eye_image.size == 0:
        return None

    eye_rgb = cv2.cvtColor(
        eye_image,
        cv2.COLOR_BGR2RGB
    )

    eye_pil = Image.fromarray(eye_rgb)

    tensor = transform(eye_pil)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

        prediction = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[prediction].item()
        )

    label = "CLOSED" if prediction == 0 else "OPEN"

    return x1, y1, x2, y2, label, confidence


# ==========================================
# 8. ЗАПУСК ВЕБ-КАМЕРЫ
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Не удалось открыть камеру")

print("Камера запущена")
print("Закрой оба глаза на 2 секунды")
print("Нажми Q для выхода")

start_time = time.monotonic()
last_timestamp = -1

eyes_closed_since = None


# ==========================================
# 9. ОСНОВНОЙ ЦИКЛ
# ==========================================

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        timestamp_ms = max(
            timestamp_ms,
            last_timestamp + 1
        )

        last_timestamp = timestamp_ms

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------
        # Если лицо обнаружено
        # ----------------------------------

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            eye_states = []

            for indices in [LEFT_EYE, RIGHT_EYE]:
                eye_result = classify_eye(
                    frame,
                    landmarks,
                    indices
                )

                if eye_result is None:
                    continue

                x1, y1, x2, y2, label, confidence = eye_result

                eye_states.append(label)

                color = (
                    (0, 255, 0)
                    if label == "OPEN"
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # ----------------------------------
            # Определяем закрытие обоих глаз
            # ----------------------------------

            both_closed = (
                len(eye_states) == 2
                and all(
                    state == "CLOSED"
                    for state in eye_states
                )
            )

            current_time = time.monotonic()

            if both_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = current_time

                closed_duration = (
                    current_time - eyes_closed_since
                )

                cv2.putText(
                    frame,
                    f"Eyes closed: {closed_duration:.1f} sec",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                # Предупреждение о сонливости
                if closed_duration >= DROWSINESS_THRESHOLD:
                    cv2.rectangle(
                        frame,
                        (15, 70),
                        (min(frame.shape[1] - 10, 450), 130),
                        (0, 0, 180),
                        -1
                    )

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (25, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (255, 255, 255),
                        2
                    )

            else:
                eyes_closed_since = None

        # ----------------------------------
        # Если лицо не найдено
        # ----------------------------------

        else:
            eyes_closed_since = None

            cv2.putText(
                frame,
                "Face not detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

        # ----------------------------------
        # Показываем изображение
        # ----------------------------------

        cv2.imshow(
            "Drowsiness Detection - CNN",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


