from collections import deque
import numpy as np


class YawnDetector:
    """
    Детектор зевания (Yawning Detection) по точкам губ MediaPipe Face Landmarker.
    Вычисляет MAR (Mouth Aspect Ratio) по внутреннему контуру губ и отличает
    продолжительный зевок (>= 1.6 с при MAR >= 0.55) от обычной речи или мимики.
    """

    # Индексы MediaPipe Face Mesh для губ:
    # Горизонтальные уголки рта: 61 (левый), 291 (правый)
    # Внутренние вертикальные пары губ: (81, 178), (13, 14), (311, 402)
    # Внешние точки верхней и нижней губы для рамки: 0 (верх), 17 (низ)
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    INNER_PAIRS = [(81, 178), (13, 14), (311, 402)]
    OUTER_CONTOUR = [61, 291, 0, 17, 37, 267, 84, 314]

    def __init__(
        self,
        mar_threshold: float = 0.55,
        min_yawn_sec: float = 1.6,
        window_seconds: float = 300.0,
        ema_alpha: float = 0.35,
    ):
        self.mar_threshold = mar_threshold
        self.min_yawn_sec = min_yawn_sec
        self.window_seconds = window_seconds
        self.ema_alpha = ema_alpha

        self.smooth_mar = 0.0
        self.mouth_open_start = None
        self.current_open_duration = 0.0
        self.is_yawning = False
        self._yawn_counted_for_current_event = False

        self.total_yawns = 0
        self.recent_yawns = deque()  # хранит timestamps зевков за последние 5 минут

    def reset(self):
        self.smooth_mar = 0.0
        self.mouth_open_start = None
        self.current_open_duration = 0.0
        self.is_yawning = False
        self._yawn_counted_for_current_event = False
        self.total_yawns = 0
        self.recent_yawns.clear()

    def _calculate_mar_and_box(self, landmarks, w: int, h: int):
        p_left = np.array([landmarks[self.MOUTH_LEFT].x * w, landmarks[self.MOUTH_LEFT].y * h])
        p_right = np.array([landmarks[self.MOUTH_RIGHT].x * w, landmarks[self.MOUTH_RIGHT].y * h])

        horizontal = np.linalg.norm(p_right - p_left)
        if horizontal < 1.0:
            return 0.0, None

        vertical_sum = 0.0
        for top_idx, bot_idx in self.INNER_PAIRS:
            p_top = np.array([landmarks[top_idx].x * w, landmarks[top_idx].y * h])
            p_bot = np.array([landmarks[bot_idx].x * w, landmarks[bot_idx].y * h])
            vertical_sum += np.linalg.norm(p_top - p_bot)

        mar = float(vertical_sum / (3.0 * horizontal))

        # Вычисляем прямоугольную рамку вокруг рта с небольшим отступом
        xs = [int(landmarks[i].x * w) for i in self.OUTER_CONTOUR]
        ys = [int(landmarks[i].y * h) for i in self.OUTER_CONTOUR]
        pad_x = max(8, int(horizontal * 0.18))
        pad_y = max(8, int(horizontal * 0.18))
        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(w, max(xs) + pad_x)
        y2 = min(h, max(ys) + pad_y)

        return mar, (x1, y1, x2, y2)

    def update(self, landmarks, w: int, h: int, now_sec: float, is_calibrating: bool) -> dict:
        raw_mar, mouth_box = self._calculate_mar_and_box(landmarks, w, h)

        if is_calibrating:
            self.smooth_mar = raw_mar
            return {
                "mar": float(raw_mar),
                "is_mouth_open": False,
                "is_yawning": False,
                "yawn_duration_sec": 0.0,
                "total_yawns": 0,
                "recent_yawns_5m": 0,
                "mouth_box": mouth_box,
                "status": "CLOSED",
            }

        self.smooth_mar = self.ema_alpha * raw_mar + (1.0 - self.ema_alpha) * self.smooth_mar
        is_mouth_open = self.smooth_mar >= self.mar_threshold

        if is_mouth_open:
            if self.mouth_open_start is None:
                self.mouth_open_start = now_sec
                self._yawn_counted_for_current_event = False
            self.current_open_duration = max(0.0, now_sec - self.mouth_open_start)

            if self.current_open_duration >= self.min_yawn_sec:
                self.is_yawning = True
                if not self._yawn_counted_for_current_event:
                    self.total_yawns += 1
                    self.recent_yawns.append(now_sec)
                    self._yawn_counted_for_current_event = True
            else:
                self.is_yawning = False
        else:
            self.mouth_open_start = None
            self.current_open_duration = 0.0
            self.is_yawning = False
            self._yawn_counted_for_current_event = False

        # Очищаем зевки старше 5 минут (window_seconds)
        cutoff = now_sec - self.window_seconds
        while self.recent_yawns and self.recent_yawns[0] < cutoff:
            self.recent_yawns.popleft()

        if self.is_yawning:
            status_str = "YAWNING!"
        elif is_mouth_open:
            status_str = "OPEN"
        else:
            status_str = "CLOSED"

        return {
            "mar": float(self.smooth_mar),
            "is_mouth_open": bool(is_mouth_open),
            "is_yawning": bool(self.is_yawning),
            "yawn_duration_sec": float(self.current_open_duration),
            "total_yawns": int(self.total_yawns),
            "recent_yawns_5m": int(len(self.recent_yawns)),
            "mouth_box": mouth_box,
            "status": status_str,
        }

