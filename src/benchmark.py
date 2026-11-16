import sys
import time
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
    DEVICE,
    FINETUNED_CNN_PATH,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EyeCNN,
    StabilizedEyeTracker,
    calculate_ear,
    crop_square_eye,
)

LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"


def benchmark_cnn(device_name: str, iterations: int = 300) -> tuple[float, float]:
    device = torch.device(device_name)
    model = EyeCNN().to(device)
    if FINETUNED_CNN_PATH.exists():
        model.load_state_dict(torch.load(FINETUNED_CNN_PATH, map_location="cpu", weights_only=True))
    model.eval()

    dummy_batch = torch.randn(2, 1, 64, 64, device=device)

    with torch.inference_mode():
        for _ in range(30):
            _ = model(dummy_batch)
        if device_name == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy_batch)
        if device_name == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / iterations) * 1000.0
    fps = iterations / elapsed
    return avg_ms, fps


def benchmark_end_to_end(frame_w: int = 1280, frame_h: int = 720, iterations: int = 100) -> tuple[float, float]:
    """
    Замеряет полный конвейер обработки кадра без учёта задержки веб-камеры:
    BGR->RGB + MediaPipe FaceLandmarker + кроп глаз + EyeCNN + StabilizedEyeTracker + DrowsinessEvaluator.
    """
    if not LANDMARKER_PATH.exists():
        return 0.0, 0.0

    model = EyeCNN().to(DEVICE)
    if FINETUNED_CNN_PATH.exists():
        model.load_state_dict(torch.load(FINETUNED_CNN_PATH, map_location="cpu", weights_only=True))
    model.eval()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    trackers = [StabilizedEyeTracker("Left"), StabilizedEyeTracker("Right")]
    evaluator = DrowsinessEvaluator(window_seconds=60.0)

    # Формируем тестовый кадр
    frame = np.full((frame_h, frame_w, 3), 128, dtype=np.uint8)
    ts_ms = 0

    for _ in range(10):
        ts_ms += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

    t0 = time.perf_counter()
    for i in range(iterations):
        ts_ms += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            l_crop = crop_square_eye(frame, lm, LEFT_EYE_LANDMARKS)
            r_crop = crop_square_eye(frame, lm, RIGHT_EYE_LANDMARKS)
            if l_crop is not None and r_crop is not None:
                batch = torch.stack([l_crop[1], r_crop[1]]).to(DEVICE)
                with torch.inference_mode():
                    probs = torch.softmax(model(batch), dim=1)[:, 0].cpu().tolist()
                l_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, frame_w, frame_h)
                r_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, frame_w, frame_h)
                l_c, _, _, l_r = trackers[0].update(probs[0], l_ear, False)
                r_c, _, _, r_r = trackers[1].update(probs[1], r_ear, False)
                evaluator.update(
                    now_sec=i * 0.033,
                    elapsed_session_sec=max(0.1, i * 0.033),
                    is_calibrating=False,
                    both_closed=(l_c and r_c),
                    mean_ear=float(np.nanmean([l_ear, r_ear])),
                    mean_ear_ratio=float(np.nanmean([l_r, r_r])),
                    landmarks=lm,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
    if str(DEVICE) == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    landmarker.close()

    avg_ms = (elapsed / iterations) * 1000.0
    fps = iterations / elapsed
    return avg_ms, fps


def main():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    print("1. Pure EyeCNN inference (batch = 2 eyes, 1x64x64):")
    print("-" * 56)
    for dev in devices:
        avg_ms, fps = benchmark_cnn(dev)
        print(f"Device: {dev.upper():<4} | Latency: {avg_ms:6.2f} ms/frame | Throughput: {fps:7.1f} FPS")

    print("\n2. End-to-End Pipeline (MediaPipe + Crop + EyeCNN + 7-feature Evaluator):")
    print("-" * 56)
    for w, h in [(640, 480), (1280, 720)]:
        avg_ms, fps = benchmark_end_to_end(w, h)
        print(f"Resolution: {w}x{h} | Latency: {avg_ms:6.2f} ms/frame | Pipeline FPS: {fps:6.1f} FPS")


if __name__ == "__main__":
    main()
