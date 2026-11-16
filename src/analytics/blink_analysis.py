from collections import deque
import numpy as np


class BlinkAnalyzer:
    """
    Отслеживает:
    - Eye Closure Duration (время непрерывного закрытия глаз, с)
    - Blink Rate (частота морганий в минуту в скользящем окне 60 с)
    - Blink Duration (средняя и последняя продолжительность морганий, мс)
    - Общее число морганий за сессию
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        min_blink_sec: float = 0.055,
        max_blink_sec: float = 0.70,
    ):
        self.window_seconds = window_seconds
        self.min_blink_sec = min_blink_sec
        self.max_blink_sec = max_blink_sec

        self.closed_start_time = None
        self.current_closure_duration = 0.0
        self.total_blinks = 0
        self.last_blink_duration_ms = 0.0
        self.last_blink_end_time = -10.0

        # Хранит кортежи (end_timestamp_sec, duration_sec)
        self.recent_blinks = deque()

    def reset(self):
        self.closed_start_time = None
        self.current_closure_duration = 0.0
        self.total_blinks = 0
        self.last_blink_duration_ms = 0.0
        self.last_blink_end_time = -10.0
        self.recent_blinks.clear()

    def update(self, now_sec: float, is_closed: bool, elapsed_session_sec: float) -> dict:
        if is_closed:
            if self.closed_start_time is None:
                self.closed_start_time = now_sec
            self.current_closure_duration = max(0.0, now_sec - self.closed_start_time)
        else:
            if self.closed_start_time is not None:
                duration = max(0.040, now_sec - self.closed_start_time)
                # Если длительность попадает в физиологический диапазон моргания и прошёл рефрактерный период 0.25 с
                if (
                    self.min_blink_sec <= duration <= self.max_blink_sec
                    and (now_sec - self.last_blink_end_time) >= 0.25
                ):
                    self.total_blinks += 1
                    self.last_blink_duration_ms = duration * 1000.0
                    self.last_blink_end_time = now_sec
                    self.recent_blinks.append((now_sec, duration))
                self.closed_start_time = None
            self.current_closure_duration = 0.0

        # Очищаем моргания старше window_seconds
        cutoff = now_sec - self.window_seconds
        while self.recent_blinks and self.recent_blinks[0][0] < cutoff:
            self.recent_blinks.popleft()

        # Частота морганий в минуту (нормализуем на прошедшее время в начале сессии, минимум 15 с)
        effective_window_min = max(15.0, min(self.window_seconds, elapsed_session_sec)) / 60.0
        blink_rate_per_min = len(self.recent_blinks) / effective_window_min

        if self.recent_blinks:
            avg_blink_ms = float(np.mean([b[1] for b in self.recent_blinks]) * 1000.0)
        else:
            avg_blink_ms = 0.0

        return {
            "closure_duration_sec": float(self.current_closure_duration),
            "total_blinks": int(self.total_blinks),
            "blink_rate_per_min": float(blink_rate_per_min),
            "avg_blink_duration_ms": float(avg_blink_ms),
            "last_blink_duration_ms": float(self.last_blink_duration_ms),
        }
