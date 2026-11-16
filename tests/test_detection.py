from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.detection.drowsiness_detector import DrowsinessEvaluator
from src.detection.eye_detector import (
    LEFT_EYE_LANDMARKS,
    EyeCNN,
    StabilizedEyeTracker,
    calculate_ear,
    crop_square_eye,
)
from src.detection.yawn_detector import YawnDetector


def _make_landmarks(overrides: dict[int, tuple[float, float, float]] | None = None):
    """Создаёт 468 синтетических точек MediaPipe в нормализованных координатах [0, 1]."""
    pts = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    # Базовые точки лица (нос, подбородок, лоб, глаза, рот) для HeadPoseEstimator и YawnDetector
    defaults = {
        1: (0.50, 0.52, -0.05),    # кончик носа
        152: (0.50, 0.78, 0.0),    # подбородок
        33: (0.35, 0.40, 0.0),     # левый глаз внешний
        133: (0.43, 0.40, 0.0),    # левый глаз внутренний
        160: (0.37, 0.38, 0.0),
        144: (0.37, 0.42, 0.0),
        159: (0.39, 0.38, 0.0),
        145: (0.39, 0.42, 0.0),
        158: (0.41, 0.38, 0.0),
        153: (0.41, 0.42, 0.0),
        263: (0.65, 0.40, 0.0),    # правый глаз внешний
        362: (0.57, 0.40, 0.0),    # правый глаз внутренний
        61: (0.42, 0.65, 0.0),     # левый уголок рта
        291: (0.58, 0.65, 0.0),    # правый уголок рта
        81: (0.46, 0.64, 0.0),
        178: (0.46, 0.66, 0.0),
        13: (0.50, 0.64, 0.0),
        14: (0.50, 0.66, 0.0),
        311: (0.54, 0.64, 0.0),
        402: (0.54, 0.66, 0.0),
        0: (0.50, 0.62, 0.0),
        17: (0.50, 0.68, 0.0),
    }
    for idx, (x, y, z) in defaults.items():
        pts[idx] = SimpleNamespace(x=x, y=y, z=z)
    if overrides:
        for idx, (x, y, z) in overrides.items():
            pts[idx] = SimpleNamespace(x=x, y=y, z=z)
    return pts


class TestEyeDetector(unittest.TestCase):
    def test_eye_cnn_forward_shape(self):
        model = EyeCNN()
        model.eval()
        batch = torch.randn(2, 1, 64, 64)
        with torch.inference_mode():
            logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_calculate_ear_positive(self):
        landmarks = _make_landmarks()
        ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS, 640, 480)
        self.assertFalse(np.isnan(ear))
        self.assertGreater(ear, 0.2)
        self.assertLess(ear, 0.6)

    def test_crop_square_eye_aspect_and_tensor(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = _make_landmarks()
        result = crop_square_eye(frame, landmarks, LEFT_EYE_LANDMARKS)
        self.assertIsNotNone(result)
        box, tensor, roi = result
        x1, y1, x2, y2 = box
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertAlmostEqual((x2 - x1) / (y2 - y1), 1.0, delta=0.1)
        self.assertEqual(tuple(tensor.shape), (1, 64, 64))
        self.assertEqual(roi.ndim, 3)

    def test_stabilized_eye_tracker_squint_vs_closed(self):
        tracker = StabilizedEyeTracker("Left", confirm_frames=2)
        # Калибровка открытых глаз
        for _ in range(10):
            tracker.update(cnn_closed_prob=0.05, ear=0.30, is_calibrating=True)

        # Прищур: EAR ~ 0.16, но зрачок виден (низкая вероятность закрытия CNN) -> SQUINT без CLOSED
        closed, squinting, _, _ = tracker.update(cnn_closed_prob=0.25, ear=0.16, is_calibrating=False)
        closed, squinting, _, _ = tracker.update(cnn_closed_prob=0.25, ear=0.16, is_calibrating=False)
        self.assertFalse(closed)
        self.assertTrue(squinting)

        # Физическое закрытие век: EAR < 0.09 -> подтверждается через 2 кадра
        tracker.update(cnn_closed_prob=0.95, ear=0.07, is_calibrating=False)
        closed, _, _, _ = tracker.update(cnn_closed_prob=0.95, ear=0.07, is_calibrating=False)
        self.assertTrue(closed)


class TestYawnAndDrowsinessEvaluator(unittest.TestCase):
    def test_yawn_detector_speech_vs_yawn(self):
        detector = YawnDetector(mar_threshold=0.55, min_yawn_sec=1.6, ema_alpha=1.0)
        # Открытый рот (MAR > 0.7)
        open_mouth = _make_landmarks({
            81: (0.46, 0.58, 0.0), 178: (0.46, 0.72, 0.0),
            13: (0.50, 0.58, 0.0), 14: (0.50, 0.72, 0.0),
            311: (0.54, 0.58, 0.0), 402: (0.54, 0.72, 0.0),
        })
        # Короткое открытие (< 1.6 с) не считается зевком
        res_short = detector.update(open_mouth, 640, 480, now_sec=1.0, is_calibrating=False)
        res_short = detector.update(open_mouth, 640, 480, now_sec=2.0, is_calibrating=False)
        self.assertFalse(res_short["is_yawning"])
        self.assertEqual(res_short["total_yawns"], 0)

        # Продолжительное открытие (>= 1.6 с) засчитывается как зевок
        res_yawn = detector.update(open_mouth, 640, 480, now_sec=2.8, is_calibrating=False)
        self.assertTrue(res_yawn["is_yawning"])
        self.assertEqual(res_yawn["total_yawns"], 1)

    def test_drowsiness_evaluator_levels(self):
        evaluator = DrowsinessEvaluator(window_seconds=60.0)
        landmarks = _make_landmarks()

        # Калибровка
        evaluator.update(0.0, 0.1, True, False, 0.30, 1.0, landmarks, 640, 480)
        # Нормальное бодрствование
        st_norm = evaluator.update(1.0, 1.0, False, False, 0.30, 1.0, landmarks, 640, 480)
        self.assertEqual(st_norm["level"], DrowsinessEvaluator.LEVEL_NORMAL)

        # Симуляция непрерывного закрытия глаз на 2.5 секунды -> HIGH
        evaluator.update(2.0, 2.0, False, True, 0.06, 0.2, landmarks, 640, 480)
        st_high = evaluator.update(4.5, 4.5, False, True, 0.06, 0.2, landmarks, 640, 480)
        self.assertEqual(st_high["level"], DrowsinessEvaluator.LEVEL_HIGH)
        self.assertGreaterEqual(st_high["score"], 65.0)
        self.assertEqual(st_high["alerts_count"], 1)


if __name__ == "__main__":
    unittest.main()
