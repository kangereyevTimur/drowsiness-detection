from collections import deque


class PerclosCalculator:
    """
    Вычисляет показатель PERCLOS (Percentage of Eyelid Closure over the Pupil over Time)
    в скользящем временном окне.
    """

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        # Храним кортежи (timestamp_sec, dt_sec, is_closed)
        self.history = deque()
        self.last_time = None

    def reset(self):
        self.history.clear()
        self.last_time = None

    def update(self, now_sec: float, is_closed: bool) -> float:
        """
        Добавляет наблюдение текущего кадра и возвращает PERCLOS в процентах (0..100).
        """
        if self.last_time is None:
            dt = 0.033
        else:
            dt = max(0.001, min(0.5, now_sec - self.last_time))
        self.last_time = now_sec

        self.history.append((now_sec, dt, 1.0 if is_closed else 0.0))

        # Удаляем записи старше window_seconds
        cutoff = now_sec - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        total_time = sum(item[1] for item in self.history)
        if total_time <= 0.001:
            return 0.0

        # Защита от скачка PERCLOS от первого моргания в первые секунды после калибровки
        effective_denom = max(10.0, total_time)
        closed_time = sum(item[1] * item[2] for item in self.history)
        return float((closed_time / effective_denom) * 100.0)
