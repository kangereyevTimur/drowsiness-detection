import math
import numpy as np


class HeadPoseEstimator:
    """
    Оценивает положение и наклон головы (Pitch, Yaw, Roll) по ключевым точкам MediaPipe Face Landmarker.
    Включает калибровку нейтрального положения головы пользователя перед экраном MacBook.
    """

    # Ключевые точки лица:
    # 1: кончик носа, 152: подбородок, 10: верх лба, 33: внешний угол L глаза, 263: внешний угол R глаза
    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD = 10
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    def __init__(self, ema_alpha: float = 0.25):
        self.ema_alpha = ema_alpha
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.baseline_roll = None
        self.calib_samples = []

        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        self.nod_start_time = None

    def reset(self):
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.baseline_roll = None
        self.calib_samples.clear()
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        self.nod_start_time = None

    def _compute_raw_angles(self, landmarks, w: int, h: int):
        p_nose = np.array([landmarks[self.NOSE_TIP].x * w, landmarks[self.NOSE_TIP].y * h, landmarks[self.NOSE_TIP].z * w])
        p_chin = np.array([landmarks[self.CHIN].x * w, landmarks[self.CHIN].y * h, landmarks[self.CHIN].z * w])
        p_fore = np.array([landmarks[self.FOREHEAD].x * w, landmarks[self.FOREHEAD].y * h, landmarks[self.FOREHEAD].z * w])
        p_l_eye = np.array([landmarks[self.LEFT_EYE_OUTER].x * w, landmarks[self.LEFT_EYE_OUTER].y * h, landmarks[self.LEFT_EYE_OUTER].z * w])
        p_r_eye = np.array([landmarks[self.RIGHT_EYE_OUTER].x * w, landmarks[self.RIGHT_EYE_OUTER].y * h, landmarks[self.RIGHT_EYE_OUTER].z * w])

        # 1. Roll (наклон головы к плечу) по линии между внешними уголками глаз
        dx_eye = p_r_eye[0] - p_l_eye[0]
        dy_eye = p_r_eye[1] - p_l_eye[1]
        roll_deg = math.degrees(math.atan2(dy_eye, max(1.0, dx_eye)))

        # 2. Yaw (поворот головы влево/вправо) по асимметрии положения носа относительно глаз
        eye_mid_x = 0.5 * (p_l_eye[0] + p_r_eye[0])
        eye_dist = max(1.0, np.linalg.norm(p_r_eye[:2] - p_l_eye[:2]))
        yaw_deg = float(((p_nose[0] - eye_mid_x) / eye_dist) * 90.0)

        # 3. Pitch (наклон головы вниз/вверх): по пропорции носа между линией глаз и подбородком
        eye_mid_y = 0.5 * (p_l_eye[1] + p_r_eye[1])
        lower_face_h = max(1.0, p_chin[1] - eye_mid_y)
        nose_ratio = (p_nose[1] - eye_mid_y) / lower_face_h
        # При наклоне головы вниз нос опускается ближе к подбородку (nose_ratio растёт)
        pitch_deg = float((nose_ratio - 0.50) * 95.0)

        return pitch_deg, yaw_deg, roll_deg

    def update(self, landmarks, w: int, h: int, now_sec: float, is_calibrating: bool) -> dict:
        raw_pitch, raw_yaw, raw_roll = self._compute_raw_angles(landmarks, w, h)

        if is_calibrating:
            self.calib_samples.append((raw_pitch, raw_yaw, raw_roll))
            return {
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
                "roll_deg": 0.0,
                "is_tilted": False,
                "nod_duration_sec": 0.0,
                "pose_status": "CALIBRATING",
            }

        if self.baseline_pitch is None:
            if self.calib_samples:
                arr = np.array(self.calib_samples)
                self.baseline_pitch = float(np.median(arr[:, 0]))
                self.baseline_yaw = float(np.median(arr[:, 1]))
                self.baseline_roll = float(np.median(arr[:, 2]))
            else:
                self.baseline_pitch, self.baseline_yaw, self.baseline_roll = raw_pitch, raw_yaw, raw_roll

        # Относительные углы от нейтральной позы пользователя
        rel_pitch = raw_pitch - self.baseline_pitch
        rel_yaw = raw_yaw - self.baseline_yaw
        rel_roll = raw_roll - self.baseline_roll

        # Медленная автоподстройка нейтрального угла при обычной рабочей позе
        if abs(rel_pitch) < 14.0 and abs(rel_roll) < 14.0:
            self.baseline_pitch = 0.995 * self.baseline_pitch + 0.005 * raw_pitch

        self.smooth_pitch = self.ema_alpha * rel_pitch + (1.0 - self.ema_alpha) * self.smooth_pitch
        self.smooth_yaw = self.ema_alpha * rel_yaw + (1.0 - self.ema_alpha) * self.smooth_yaw
        self.smooth_roll = self.ema_alpha * rel_roll + (1.0 - self.ema_alpha) * self.smooth_roll

        # Проверка сильного наклона головы вниз (Pitch > 16°) или вбок (|Roll| > 18°)
        is_nodding_down = self.smooth_pitch > 16.0
        is_rolling_side = abs(self.smooth_roll) > 18.0
        is_looking_away = abs(self.smooth_yaw) > 25.0
        is_tilted = is_nodding_down or is_rolling_side

        if is_tilted:
            if self.nod_start_time is None:
                self.nod_start_time = now_sec
            nod_dur = max(0.0, now_sec - self.nod_start_time)
        else:
            self.nod_start_time = None
            nod_dur = 0.0

        if is_nodding_down:
            pose_status = "HEAD DOWN"
        elif is_rolling_side:
            pose_status = "HEAD TILT"
        elif is_looking_away:
            pose_status = "LOOKING AWAY"
        else:
            pose_status = "NORMAL"

        return {
            "pitch_deg": float(self.smooth_pitch),
            "yaw_deg": float(self.smooth_yaw),
            "roll_deg": float(self.smooth_roll),
            "is_tilted": bool(is_tilted),
            "nod_duration_sec": float(nod_dur),
            "pose_status": pose_status,
        }

