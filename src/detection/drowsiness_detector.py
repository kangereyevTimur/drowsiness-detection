from src.analytics.blink_analysis import BlinkAnalyzer
from src.analytics.perclos import PerclosCalculator
from src.detection.head_pose import HeadPoseEstimator
from src.detection.yawn_detector import YawnDetector


class DrowsinessEvaluator:
    """
    Многофакторный алгоритм оценки уровня сонливости на основе 7 признаков:
    1. EAR (степень раскрытия глаз)
    2. PERCLOS (доля времени с закрытыми глазами, %)
    3. Blink Rate (частота моргания, раз/мин)
    4. Blink Duration (продолжительность моргания, мс)
    5. Eye Closure Duration (время непрерывного закрытия глаз, с)
    6. Head Pose (положение и наклон головы: Pitch, Yaw, Roll)
    7. Yawning / MAR (степень раскрытия рта и частота зеваний)
    """

    LEVEL_NORMAL = "NORMAL"
    LEVEL_ELEVATED = "ELEVATED"
    LEVEL_HIGH = "HIGH"

    LEVEL_RU = {
        "NORMAL": "Нормальное состояние",
        "ELEVATED": "Повышенная сонливость",
        "HIGH": "Высокая сонливость",
    }

    def __init__(self, window_seconds: float = 60.0):
        self.perclos_calc = PerclosCalculator(window_seconds=window_seconds)
        self.blink_analyzer = BlinkAnalyzer(window_seconds=window_seconds)
        self.head_pose = HeadPoseEstimator()
        self.yawn_detector = YawnDetector()

        self.last_time = None
        self.current_level = self.LEVEL_NORMAL
        self.alerts_count = 0
        self._in_high_alert = False
        self.sensitivity_multiplier = 1.0

        # Накопленное время в каждом состоянии (сек)
        self.state_durations = {
            self.LEVEL_NORMAL: 0.0,
            self.LEVEL_ELEVATED: 0.0,
            self.LEVEL_HIGH: 0.0,
        }

    def reset(self):
        self.perclos_calc.reset()
        self.blink_analyzer.reset()
        self.head_pose.reset()
        self.yawn_detector.reset()
        self.last_time = None
        self.current_level = self.LEVEL_NORMAL
        self.alerts_count = 0
        self._in_high_alert = False
        self.state_durations = {
            self.LEVEL_NORMAL: 0.0,
            self.LEVEL_ELEVATED: 0.0,
            self.LEVEL_HIGH: 0.0,
        }

    @staticmethod
    def _build_recommendation(
        level: str,
        perclos_pct: float,
        recent_yawns: int,
        closure_dur: float,
        nod_dur: float,
    ) -> str:
        if level == DrowsinessEvaluator.LEVEL_HIGH:
            if closure_dur >= 2.2:
                return "Зафиксировано длительное закрытие глаз — срочно сделайте паузу и разомнитесь."
            if perclos_pct >= 28.0:
                return "Высокий показатель PERCLOS — рекомендуется перерыв минимум на 10–15 минут."
            return "Обнаружены сильные признаки усталости — остановите работу и отдохните."
        if level == DrowsinessEvaluator.LEVEL_ELEVATED:
            if recent_yawns >= 2:
                return f"Зафиксировано {recent_yawns} зевка за 5 мин — проветрите помещение или сделайте паузу."
            if nod_dur >= 1.5:
                return "Обнаружен наклон головы — выпрямите осанку и смените фокус взгляда."
            if perclos_pct >= 18.0:
                return "Растёт усталость век (PERCLOS > 18%) — сделайте короткий перерыв на 5 минут."
            return "Появляются первые признаки утомления — следите за концентрацией."
        return "Состояние в норме — продолжайте работу в штатном режиме."

    def update(
        self,
        now_sec: float,
        elapsed_session_sec: float,
        is_calibrating: bool,
        both_closed: bool,
        mean_ear: float,
        mean_ear_ratio: float,
        landmarks,
        frame_w: int,
        frame_h: int,
    ) -> dict:
        if self.last_time is None:
            dt = 0.033
        else:
            dt = max(0.001, min(0.5, now_sec - self.last_time))
        self.last_time = now_sec

        # 1. Head Pose & Yawning (MAR)
        pose_info = self.head_pose.update(landmarks, frame_w, frame_h, now_sec, is_calibrating)
        yawn_info = self.yawn_detector.update(landmarks, frame_w, frame_h, now_sec, is_calibrating)

        if is_calibrating:
            return {
                "level": self.LEVEL_NORMAL,
                "level_ru": "Калибровка...",
                "score": 0.0,
                "perclos": 0.0,
                "closure_duration_sec": 0.0,
                "total_blinks": 0,
                "blink_rate_per_min": 0.0,
                "avg_blink_duration_ms": 0.0,
                "mean_ear": mean_ear,
                "mean_ear_ratio": mean_ear_ratio,
                "pose": pose_info,
                "yawn": yawn_info,
                "reasons": [],
                "recommendation": "Идёт калибровка базового положения глаз и головы...",
                "alerts_count": self.alerts_count,
                "state_durations": dict(self.state_durations),
            }

        # 2. PERCLOS & Blinks
        perclos_pct = self.perclos_calc.update(now_sec, both_closed)
        blink_info = self.blink_analyzer.update(now_sec, both_closed, elapsed_session_sec)

        closure_dur = blink_info["closure_duration_sec"]
        blink_rate = blink_info["blink_rate_per_min"]
        avg_blink_ms = blink_info["avg_blink_duration_ms"]
        nod_dur = pose_info["nod_duration_sec"]
        recent_yawns = yawn_info["recent_yawns_5m"]

        # Признак накопленной усталости: одиночное закрытие на ~2 с без фона усталости не бьёт сразу HIGH
        has_fatigue_context = perclos_pct >= 18.0 or recent_yawns >= 2 or nod_dur >= 1.5
        sens = max(0.7, min(1.3, self.sensitivity_multiplier))
        high_closure_thr = (2.3 if has_fatigue_context else 3.2) / sens
        elevated_closure_thr = (1.5 if has_fatigue_context else 2.2) / sens

        # 3. Расчёт многофакторного индекса сонливости (0..100)
        score = 0.0
        reasons = []

        # Фактор A: Непрерывное закрытие глаз (Eye Closure Duration)
        if closure_dur >= high_closure_thr:
            score += 72.0 + min(28.0, (closure_dur - high_closure_thr) * 18.0)
            reasons.append(f"Глаза закрыты {closure_dur:.1f}с")
        elif closure_dur >= elevated_closure_thr:
            score += 36.0 + min(24.0, (closure_dur - elevated_closure_thr) * 20.0)
            reasons.append(f"Затяжное закрытие ({closure_dur:.1f}с)")
        elif closure_dur >= 1.2:
            score += (closure_dur - 1.2) * 18.0

        # Фактор B: Доля закрытых глаз PERCLOS
        if perclos_pct >= 28.0:
            score += 65.0
            reasons.append(f"Высокий PERCLOS ({perclos_pct:.1f}%)")
        elif perclos_pct >= 16.0:
            score += 24.0 + (perclos_pct - 16.0) * 2.0
            reasons.append(f"Повышенный PERCLOS ({perclos_pct:.1f}%)")
        else:
            score += perclos_pct * 0.7

        # Фактор C: Длительность морганий (Blink Duration)
        if avg_blink_ms >= 380.0:
            score += 22.0
            reasons.append(f"Медленные моргания ({avg_blink_ms:.0f}мс)")
        elif avg_blink_ms >= 280.0:
            score += 12.0
            reasons.append(f"Вялые моргания ({avg_blink_ms:.0f}мс)")

        # Фактор D: Частота морганий (Blink Rate)
        if elapsed_session_sec >= 20.0 and blink_rate >= 26.0:
            score += 15.0
            reasons.append(f"Частое моргание ({blink_rate:.0f}/мин)")

        # Фактор E: Положение и наклон головы (Head Pose)
        if nod_dur >= 1.8:
            score += 25.0 + min(25.0, (nod_dur - 1.8) * 10.0)
            reasons.append(f"Наклон головы ({pose_info['pose_status']} {nod_dur:.1f}с)")
            if closure_dur >= 1.4:
                score += 25.0

        # Фактор F: Зевание (Yawning / MAR)
        if yawn_info["is_yawning"]:
            score += 30.0
            reasons.append(f"Зевание ({yawn_info['yawn_duration_sec']:.1f}с)")
        if recent_yawns >= 3:
            score += 28.0
            reasons.append(f"Частые зевки ({recent_yawns} за 5м)")
        elif recent_yawns == 2:
            score += 16.0
            reasons.append("Повторное зевание (2 за 5м)")

        score = float(min(100.0, max(0.0, score * self.sensitivity_multiplier)))

        # 4. Определение уровня сонливости
        if score >= 68.0 or closure_dur >= high_closure_thr or perclos_pct >= 28.0:
            level = self.LEVEL_HIGH
        elif (
            score >= 38.0
            or closure_dur >= elevated_closure_thr
            or perclos_pct >= 16.0
            or nod_dur >= 1.8
            or yawn_info["is_yawning"]
        ):
            level = self.LEVEL_ELEVATED
        else:
            level = self.LEVEL_NORMAL

        # Подсчёт срабатываний высокой тревоги и времени в состояниях
        if level == self.LEVEL_HIGH and not self._in_high_alert:
            self.alerts_count += 1
            self._in_high_alert = True
        elif level != self.LEVEL_HIGH:
            self._in_high_alert = False

        self.current_level = level
        self.state_durations[level] += dt
        recommendation = self._build_recommendation(level, perclos_pct, recent_yawns, closure_dur, nod_dur)

        return {
            "level": level,
            "level_ru": self.LEVEL_RU[level],
            "score": score,
            "perclos": perclos_pct,
            "closure_duration_sec": closure_dur,
            "total_blinks": blink_info["total_blinks"],
            "blink_rate_per_min": blink_rate,
            "avg_blink_duration_ms": avg_blink_ms,
            "last_blink_duration_ms": blink_info["last_blink_duration_ms"],
            "mean_ear": mean_ear,
            "mean_ear_ratio": mean_ear_ratio,
            "pose": pose_info,
            "yawn": yawn_info,
            "reasons": reasons,
            "recommendation": recommendation,
            "alerts_count": self.alerts_count,
            "state_durations": dict(self.state_durations),
        }
