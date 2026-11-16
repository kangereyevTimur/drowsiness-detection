import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
CNN_PATH = ROOT / 'models' / 'eye_cnn_best.pth'
LANDMARKER_PATH = ROOT / 'models' / 'face_landmarker.task'

DROWSINESS_THRESHOLD = 2.0
EMA_ALPHA = 0.35
CLOSE_THRESHOLD = 0.68
OPEN_THRESHOLD = 0.40
MIN_CONFIRM_FRAMES = 3
EAR_WEIGHT = 0.30  # Base contribution of eye geometry
EAR_STRONG_CLOSED_RATIO = 0.53  # Relative to calibrated open eye
EAR_STRONG_OPEN_RATIO = 0.88
EAR_CALIBRATION_SECONDS = 4.0
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

LEFT_EYE = [33, 133, 159, 145, 160, 144]
RIGHT_EYE = [362, 263, 386, 374, 387, 373]

class EyeCNN(nn.Module):
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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

class EyeState:
    def __init__(self):
        self.ema = None
        self.closed = False
        self.pending = 0
        self.ear_baseline = None
        self.ear_samples = []

    def reset_temporal(self):
        self.ema = None
        self.closed = False
        self.pending = 0

    def update(self, cnn_closed, ear, calibrating):
        if calibrating and np.isfinite(ear) and 0.10 < ear < 0.65 and cnn_closed < 0.55:
            self.ear_samples.append(float(ear))
        if not calibrating and self.ear_baseline is None and len(self.ear_samples) >= 8:
            self.ear_baseline = float(np.median(self.ear_samples))

        # Use a per-eye baseline; very small EAR can overrule an erroneous CNN OPEN.
        fused = float(cnn_closed)
        ratio = float('nan')
        if self.ear_baseline is not None and np.isfinite(ear) and self.ear_baseline > 0.10:
            ratio = float(ear / self.ear_baseline)
            ear_closed = float(np.clip((EAR_STRONG_OPEN_RATIO - ratio) / 0.35, 0.0, 1.0))
            weight = 0.75 if ratio <= EAR_STRONG_CLOSED_RATIO else EAR_WEIGHT
            fused = (1.0 - weight) * fused + weight * ear_closed

        self.ema = fused if self.ema is None else EMA_ALPHA * fused + (1 - EMA_ALPHA) * self.ema
        proposed = self.closed
        if not self.closed and self.ema >= CLOSE_THRESHOLD:
            proposed = True
        elif self.closed and self.ema <= OPEN_THRESHOLD:
            proposed = False

        if proposed != self.closed:
            self.pending += 1
            if self.pending >= MIN_CONFIRM_FRAMES:
                self.closed = proposed
                self.pending = 0
        else:
            self.pending = 0
        return self.closed, self.ema, ratio


def eye_ear(landmarks, ids, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in ids]
    horizontal = np.linalg.norm(p[0] - p[1])
    if horizontal < 1:
        return float('nan')
    return float((np.linalg.norm(p[2] - p[3]) + np.linalg.norm(p[4] - p[5])) / (2 * horizontal))


def crop_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    xs, ys = zip(*points)
    ew, eh = max(xs) - min(xs), max(ys) - min(ys)
    if ew < 8 or eh < 1:
        return None
    mx = int(ew * 0.25)
    my = max(int(eh * 0.8), int(ew * 0.15))
    x1, y1 = max(0, min(xs) - mx), max(0, min(ys) - my)
    x2, y2 = min(w, max(xs) + mx), min(h, max(ys) + my)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    tensor = transform(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
    return (x1, y1, x2, y2), tensor


def main():
    if not CNN_PATH.exists() or not LANDMARKER_PATH.exists():
        raise FileNotFoundError('Проверь файлы models/eye_cnn_best.pth и models/face_landmarker.task')
    print('Устройство:', DEVICE)
    model = EyeCNN().to(DEVICE)
    model.load_state_dict(torch.load(CNN_PATH, map_location='cpu', weights_only=True))
    model.eval()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Камера не открылась. Проверь разрешения macOS.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    states = [EyeState(), EyeState()]
    closed_since = None
    last_face_seen = time.monotonic()
    start = time.monotonic()
    last_timestamp = -1
    calibration_start = time.monotonic()
    print('Первые 4 секунды смотри прямо в камеру с открытыми глазами.')
    print('v3: EAR stronger when eyelids are clearly closed.')
    print('R - повторная калибровка, Q - выход')

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.monotonic()
                ts = max(int((now - start) * 1000), last_timestamp + 1)
                last_timestamp = ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                calibrating = now - calibration_start < EAR_CALIBRATION_SECONDS
                detected = False
                both_closed = False

                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    crops = [crop_eye(frame, lm, ids) for ids in (LEFT_EYE, RIGHT_EYE)]
                    if all(c is not None for c in crops):
                        detected = True
                        last_face_seen = now
                        batch = torch.stack([c[1] for c in crops]).to(DEVICE)
                        with torch.inference_mode():
                            closed_probs = torch.softmax(model(batch), dim=1)[:, 0].cpu().tolist()
                        flags = []
                        for i, (ids, item, prob) in enumerate(zip((LEFT_EYE, RIGHT_EYE), crops, closed_probs)):
                            box, _ = item
                            ear = eye_ear(lm, ids, w, h)
                            is_closed, smooth, ratio = states[i].update(prob, ear, calibrating)
                            flags.append(is_closed)
                            x1, y1, x2, y2 = box
                            color = (0, 70, 255) if is_closed else (50, 210, 80)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = 'CLOSED' if is_closed else 'OPEN'
                            ear_info = f'EAR:{ratio:.2f}' if np.isfinite(ratio) else 'EAR:cal'
                            cv2.putText(frame, f'{label} CNN:{prob:.2f} {ear_info}',
                                        (max(5, x1 - 25), max(20, y1 - 9)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
                        both_closed = all(flags)
                        # Bilateral confirmation: rescue one misclassified eye only when
                        # both geometric eye measurements strongly indicate closure.
                        ratios = [
                            eye_ear(lm, ids, w, h) / state.ear_baseline
                            if state.ear_baseline is not None and state.ear_baseline > 0.10
                            else float('nan')
                            for ids, state in zip((LEFT_EYE, RIGHT_EYE), states)
                        ]
                        if (not calibrating and not both_closed and any(flags)
                                and all(np.isfinite(r) and r <= 0.58 for r in ratios)):
                            both_closed = True
                            cv2.putText(frame, 'Both eyelids closed (EAR confirmed)',
                                        (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                                        (0, 170, 255), 2)

                if not detected:
                    closed_since = None
                    cv2.putText(frame, 'Face/eyes not detected', (25, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
                    if now - last_face_seen > 0.7:
                        for s in states:
                            s.reset_temporal()
                elif calibrating:
                    closed_since = None
                    remaining = max(0.0, EAR_CALIBRATION_SECONDS - (now - calibration_start))
                    cv2.putText(frame, f'Keep eyes OPEN: {remaining:.1f}s', (25, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
                elif both_closed:
                    if closed_since is None:
                        closed_since = now
                    duration = now - closed_since
                    cv2.putText(frame, f'Eyes closed: {duration:.1f}s', (25, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 70, 255), 2)
                    if duration >= DROWSINESS_THRESHOLD:
                        cv2.rectangle(frame, (15, 62), (min(w - 10, 440), 120), (0, 0, 170), -1)
                        cv2.putText(frame, 'DROWSINESS ALERT!', (25, 101),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    closed_since = None
                    cv2.putText(frame, 'Monitoring eyes', (25, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 210, 80), 2)

                if not calibrating and any(s.ear_baseline is None for s in states):
                    cv2.putText(frame, 'EAR not calibrated - CNN smoothing active',
                                (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
                cv2.putText(frame, 'Q: quit  R: recalibrate', (25, h - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
                cv2.imshow('Drowsiness Detection v3 - CNN + EAR', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('r'):
                    calibration_start = time.monotonic()
                    closed_since = None
                    states = [EyeState(), EyeState()]
                    print('Повторная калибровка: смотри прямо, глаза открыты.')
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
