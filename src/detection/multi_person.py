from dataclasses import dataclass, field

import numpy as np
import torch

from src.detection.drowsiness_detector import DrowsinessEvaluator
from src.detection.eye_detector import (
    DEVICE,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    StabilizedEyeTracker,
    calculate_ear,
    crop_square_eye,
)


@dataclass
class TrackedPerson:
    person_id: int
    centroid: tuple[float, float]
    first_seen_sec: float
    last_seen_sec: float
    calibration_start_sec: float
    left_tracker: StabilizedEyeTracker = field(default_factory=lambda: StabilizedEyeTracker("Left"))
    right_tracker: StabilizedEyeTracker = field(default_factory=lambda: StabilizedEyeTracker("Right"))
    evaluator: DrowsinessEvaluator = field(default_factory=lambda: DrowsinessEvaluator(window_seconds=60.0))
    last_status: dict | None = None


def _face_centroid_and_box(landmarks, frame_w: int, frame_h: int):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    cx = float(landmarks[1].x) if len(landmarks) > 1 else float(np.mean(xs))
    cy = float(landmarks[1].y) if len(landmarks) > 1 else float(np.mean(ys))
    x1 = max(0, int(min(xs) * frame_w) - 10)
    y1 = max(0, int(min(ys) * frame_h) - 14)
    x2 = min(frame_w, int(max(xs) * frame_w) + 10)
    y2 = min(frame_h, int(max(ys) * frame_h) + 10)
    return (cx, cy), (x1, y1, x2, y2)


class MultiPersonTracker:
    """
    Отслеживает от 1 до N человек в кадре одновременно.
    Сопоставляет лица между кадрами по ближайшему центроиду, сохраняя
    индивидуальную калибровку EAR/HeadPose, PERCLOS и счётчики морганий для каждого ID.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_match_dist: float = 0.22,
        stale_timeout_sec: float = 2.5,
    ):
        self.window_seconds = window_seconds
        self.max_match_dist = max_match_dist
        self.stale_timeout_sec = stale_timeout_sec
        self.sensitivity_multiplier = 1.0

        self.persons: dict[int, TrackedPerson] = {}
        self._next_id = 1

    def reset(self):
        self.persons.clear()
        self._next_id = 1

    def recalibrate_all(self, now_sec: float):
        for p in self.persons.values():
            p.calibration_start_sec = now_sec
            p.left_tracker.reset()
            p.right_tracker.reset()
            p.evaluator.reset()
            p.evaluator.sensitivity_multiplier = self.sensitivity_multiplier

    def set_sensitivity(self, multiplier: float):
        self.sensitivity_multiplier = multiplier
        for p in self.persons.values():
            p.evaluator.sensitivity_multiplier = multiplier

    def _match_faces(self, centroids: list[tuple[float, float]], now_sec: float) -> list[TrackedPerson]:
        # Удаляем давно пропавших из кадра людей
        stale_ids = [
            pid for pid, p in self.persons.items()
            if (now_sec - p.last_seen_sec) > self.stale_timeout_sec
        ]
        for pid in stale_ids:
            del self.persons[pid]

        # Жадное сопоставление пар (лицо, существующий трек) по минимальному расстоянию
        existing_ids = list(self.persons.keys())
        pairs = []
        for f_idx, (cx, cy) in enumerate(centroids):
            for pid in existing_ids:
                px, py = self.persons[pid].centroid
                dist = float(np.hypot(cx - px, cy - py))
                if dist <= self.max_match_dist:
                    pairs.append((dist, f_idx, pid))
        pairs.sort(key=lambda item: item[0])

        matched_face: dict[int, TrackedPerson] = {}
        used_pids = set()
        for _, f_idx, pid in pairs:
            if f_idx in matched_face or pid in used_pids:
                continue
            person = self.persons[pid]
            person.centroid = centroids[f_idx]
            person.last_seen_sec = now_sec
            matched_face[f_idx] = person
            used_pids.add(pid)

        # Для новых несопоставленных лиц создаём новый трек
        ordered_persons = []
        for f_idx, centroid in enumerate(centroids):
            if f_idx in matched_face:
                ordered_persons.append(matched_face[f_idx])
            else:
                pid = self._next_id
                self._next_id += 1
                ev = DrowsinessEvaluator(window_seconds=self.window_seconds)
                ev.sensitivity_multiplier = self.sensitivity_multiplier
                new_p = TrackedPerson(
                    person_id=pid,
                    centroid=centroid,
                    first_seen_sec=now_sec,
                    last_seen_sec=now_sec,
                    calibration_start_sec=now_sec,
                    evaluator=ev,
                )
                self.persons[pid] = new_p
                ordered_persons.append(new_p)

        return ordered_persons

    def process_frame(
        self,
        frame: np.ndarray,
        face_landmarks_list: list,
        model: torch.nn.Module,
        now_sec: float,
        cal_limit_sec: float = 3.5,
    ) -> list[dict]:
        """
        Обрабатывает все обнаруженные лица за один батч-проход EyeCNN.
        Возвращает список словарей с результатами по каждому человеку.
        """
        if not face_landmarks_list:
            return []

        h, w = frame.shape[:2]
        centroids = []
        face_boxes = []
        for lm in face_landmarks_list:
            c, box = _face_centroid_and_box(lm, w, h)
            centroids.append(c)
            face_boxes.append(box)

        matched_persons = self._match_faces(centroids, now_sec)

        # Собираем кропы всех глаз со всех лиц для одного батч-инференса EyeCNN
        valid_entries = []
        tensors = []
        for lm, face_box, person in zip(face_landmarks_list, face_boxes, matched_persons):
            left_crop = crop_square_eye(frame, lm, LEFT_EYE_LANDMARKS)
            right_crop = crop_square_eye(frame, lm, RIGHT_EYE_LANDMARKS)
            if left_crop is not None and right_crop is not None:
                valid_entries.append((lm, face_box, person, left_crop, right_crop))
                tensors.append(left_crop[1])
                tensors.append(right_crop[1])

        if not valid_entries:
            return []

        model_device = next(model.parameters()).device
        batch = torch.stack(tensors).to(model_device)
        with torch.inference_mode():
            all_probs = torch.softmax(model(batch), dim=1)[:, 0].cpu().tolist()

        results = []
        for idx, (lm, face_box, person, left_crop, right_crop) in enumerate(valid_entries):
            prob_l = float(all_probs[idx * 2])
            prob_r = float(all_probs[idx * 2 + 1])

            elapsed_cal = now_sec - person.calibration_start_sec
            is_calibrating = elapsed_cal < cal_limit_sec
            rem_cal = max(0.0, cal_limit_sec - elapsed_cal)

            l_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, w, h)
            r_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, w, h)

            l_closed, l_sq, _, l_ratio = person.left_tracker.update(prob_l, l_ear, is_calibrating)
            r_closed, r_sq, _, r_ratio = person.right_tracker.update(prob_r, r_ear, is_calibrating)

            both_closed = l_closed and r_closed
            mean_ear = float(np.nanmean([l_ear, r_ear]))
            mean_ratio = float(np.nanmean([l_ratio, r_ratio]))

            status = person.evaluator.update(
                now_sec=now_sec,
                elapsed_session_sec=max(0.1, elapsed_cal - cal_limit_sec),
                is_calibrating=is_calibrating,
                both_closed=both_closed,
                mean_ear=mean_ear,
                mean_ear_ratio=mean_ratio,
                landmarks=lm,
                frame_w=w,
                frame_h=h,
            )
            person.last_status = status

            results.append({
                "person_id": person.person_id,
                "landmarks": lm,
                "face_box": face_box,
                "left_crop": left_crop,
                "right_crop": right_crop,
                "prob_l": prob_l,
                "prob_r": prob_r,
                "l_ear": l_ear,
                "r_ear": r_ear,
                "l_closed": l_closed,
                "r_closed": r_closed,
                "l_sq": l_sq,
                "r_sq": r_sq,
                "both_closed": both_closed,
                "is_calibrating": is_calibrating,
                "rem_cal": rem_cal,
                "status": status,
            })

        return results
