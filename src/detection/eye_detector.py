from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
FINETUNED_CNN_PATH = ROOT / "models" / "eye_cnn_finetuned.pth"
BASE_CNN_PATH = ROOT / "models" / "eye_cnn_best.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LEFT_EYE_LANDMARKS = {
    "outer": 33,
    "inner": 133,
    "top_bottom_pairs": [(160, 144), (159, 145), (158, 153)],
}

RIGHT_EYE_LANDMARKS = {
    "outer": 263,
    "inner": 362,
    "top_bottom_pairs": [(387, 373), (386, 374), (385, 380)],
}


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


cnn_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def calculate_ear(landmarks, eye_def, frame_w: int, frame_h: int) -> float:
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


def crop_square_eye(frame, landmarks, eye_def, scale_factor: float = 1.35):
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

    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_roi)
    tensor = cnn_transform(pil_img)

    return (x1, y1, x2, y2), tensor, roi


class StabilizedEyeTracker:
    """
    Отслеживает состояние одного глаза (OPEN / SQUINT / CLOSED).
    Подтверждение за 2 кадра (~65 мс) позволяет улавливать все естественные моргания (100-400 мс),
    отсекая при этом одиночные случайные выбросы.
    """

    def __init__(self, name: str, confirm_frames: int = 2, ema_alpha: float = 0.45):
        self.name = name
        self.confirm_frames = confirm_frames
        self.ema_alpha = ema_alpha
        self.ema_prob = None
        self.closed = False
        self.is_squinting = False
        self.pending_count = 0
        self.baseline_ear = None
        self.ear_calibration_samples = []

    def reset(self):
        self.ema_prob = None
        self.closed = False
        self.is_squinting = False
        self.pending_count = 0
        self.baseline_ear = None
        self.ear_calibration_samples.clear()

    def calibrate(self, ear: float):
        if np.isfinite(ear) and 0.16 <= ear <= 0.40:
            self.ear_calibration_samples.append(ear)

    def finalize_calibration(self):
        if len(self.ear_calibration_samples) >= 8:
            med = float(np.median(self.ear_calibration_samples))
            self.baseline_ear = float(np.clip(med, 0.22, 0.33))
        elif len(self.ear_calibration_samples) > 0:
            self.baseline_ear = float(np.clip(np.mean(self.ear_calibration_samples), 0.22, 0.33))
        else:
            self.baseline_ear = 0.26

    def update(self, cnn_closed_prob: float, ear: float, is_calibrating: bool):
        if is_calibrating:
            self.calibrate(ear)
            return False, False, cnn_closed_prob, 1.0

        if self.baseline_ear is None:
            self.finalize_calibration()

        safe_ear = ear if np.isfinite(ear) else 0.24
        ratio = safe_ear / self.baseline_ear

        self.is_squinting = False

        if safe_ear < 0.095:
            # Полное физиологическое смыкание век
            raw_closed = True
            fused_prob = max(0.80, cnn_closed_prob)
        elif safe_ear < 0.135 and cnn_closed_prob >= 0.22:
            # Сильное смыкание век, подтверждённое CNN
            raw_closed = True
            fused_prob = max(0.70, cnn_closed_prob)
        elif cnn_closed_prob >= 0.55 and (safe_ear < 0.19 or ratio < 0.68):
            # И CNN, и геометрия подтверждают закрытие
            raw_closed = True
            fused_prob = 0.8 * cnn_closed_prob + 0.2 * (1.0 - min(1.0, ratio))
        elif cnn_closed_prob >= 0.82:
            raw_closed = True
            fused_prob = cnn_closed_prob
        else:
            # Открытый глаз или прищур (quando зрачок виден, а EAR > 0.10)
            if safe_ear < 0.19 or ratio < 0.68:
                self.is_squinting = True
            raw_closed = False
            fused_prob = cnn_closed_prob * 0.6

        if self.ema_prob is None:
            self.ema_prob = fused_prob
        else:
            self.ema_prob = self.ema_alpha * fused_prob + (1.0 - self.ema_alpha) * self.ema_prob

        if raw_closed != self.closed:
            self.pending_count += 1
            if self.pending_count >= self.confirm_frames:
                self.closed = raw_closed
                self.pending_count = 0
        else:
            self.pending_count = 0

        return self.closed, self.is_squinting, float(self.ema_prob), float(ratio)

