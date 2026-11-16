import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.eye_detector import FINETUNED_CNN_PATH, EyeCNN


def benchmark_device(device_name: str, iterations: int = 300) -> tuple[float, float]:
    device = torch.device(device_name)
    model = EyeCNN().to(device)
    if FINETUNED_CNN_PATH.exists():
        model.load_state_dict(torch.load(FINETUNED_CNN_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # Батч из двух глаз (левый + правый) на один кадр видео
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


def main():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    print("Benchmark EyeCNN (batch size = 2 eyes, 1x64x64):")
    print("-" * 52)
    for dev in devices:
        avg_ms, fps = benchmark_device(dev)
        print(f"Device: {dev.upper():<4} | Latency: {avg_ms:6.2f} ms/frame | Max Throughput: {fps:7.1f} FPS")


if __name__ == "__main__":
    main()
