import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Система определения сонливости и усталости человека (Deep Learning + MediaPipe)"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Запустить в режиме окна OpenCV (detect_drowsiness_v5) вместо графического интерфейса CustomTkinter",
    )
    args = parser.parse_args()

    if args.cli:
        print("[1/2] Загрузка библиотек компьютерного зрения (PyTorch, MediaPipe, OpenCV)...", flush=True)
        from src.detect_drowsiness_v5 import main as run_cli

        print("[2/2] Запуск окна видеомониторинга OpenCV...", flush=True)
        run_cli()
    else:
        print("[1/3] Загрузка библиотек (PyTorch, MediaPipe, CustomTkinter)...", flush=True)
        from src.interface.app import main as run_gui

        print("[2/3] Инициализация нейросети EyeCNN и графической панели...", flush=True)
        run_gui()


if __name__ == "__main__":
    main()
