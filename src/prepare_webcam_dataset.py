import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_WEBCAM_DIR = ROOT / "data" / "webcam_dataset"
SPLIT_WEBCAM_DIR = ROOT / "data" / "webcam_split"
BACKUP_CONFLICT_DIR = ROOT / "data" / "webcam_conflict_backup"

# Доли выборок внутри каждого временного сегмента
TRAIN_RATIO = 0.66
VAL_RATIO = 0.16
# Буфер в кадрах между выборками, чтобы исключить утечку соседних по времени кадров
BUFFER_FRAMES = 2

# Индексы переходных кадров в папке closed, где пользователь нажимал клавишу с открытыми глазами
CLOSED_OPEN_TRANSITION_INDICES = (
    set(range(0, 168))
    | set(range(264, 279))
    | set(range(299, 311))
    | set(range(427, 430))
)


def group_into_segments(frame_pairs, max_gap_ms=1000):
    """Разбивает список пар кадров на непрерывные временные сегменты."""
    if not frame_pairs:
        return []
    segments = []
    current_seg = [frame_pairs[0]]
    for prev_pair, curr_pair in zip(frame_pairs, frame_pairs[1:]):
        t_prev = int(prev_pair[0].stem.split("_")[-1])
        t_curr = int(curr_pair[0].stem.split("_")[-1])
        if t_curr - t_prev > max_gap_ms:
            if len(current_seg) >= 15:
                segments.append(current_seg)
            current_seg = [curr_pair]
        else:
            current_seg.append(curr_pair)
    if len(current_seg) >= 15:
        segments.append(current_seg)
    return segments


def split_class(cls_name: str):
    cls_dir = RAW_WEBCAM_DIR / cls_name
    if not cls_dir.exists():
        raise FileNotFoundError(f"Папка {cls_dir} не найдена.")

    left_files = sorted(cls_dir.glob("*_L_*.png"))
    clean_pairs = []
    filtered_count = 0

    for idx, lf in enumerate(left_files):
        rf = cls_dir / lf.name.replace("_L_", "_R_")
        pair = (lf, rf) if rf.exists() else (lf,)

        # Фильтрация переходных кадров с открытыми глазами в папке closed
        if cls_name == "closed" and idx in CLOSED_OPEN_TRANSITION_INDICES:
            BACKUP_CONFLICT_DIR.mkdir(parents=True, exist_ok=True)
            for f in pair:
                shutil.copy2(f, BACKUP_CONFLICT_DIR / f.name)
                filtered_count += 1
            continue

        clean_pairs.append(pair)

    segments = group_into_segments(clean_pairs)
    splits = {"train": [], "val": [], "test": []}

    for seg in segments:
        n = len(seg)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits["train"].extend(seg[:n_train])
        val_start = min(n, n_train + BUFFER_FRAMES)
        val_end = min(n, val_start + n_val)
        splits["val"].extend(seg[val_start:val_end])

        test_start = min(n, val_end + BUFFER_FRAMES)
        splits["test"].extend(seg[test_start:])

    counts = {}
    for split_name, pairs in splits.items():
        out_dir = SPLIT_WEBCAM_DIR / split_name / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        img_count = 0
        for pair in pairs:
            for src_file in pair:
                shutil.copy2(src_file, out_dir / src_file.name)
                img_count += 1
        counts[split_name] = img_count

    return counts, filtered_count, len(segments)


def main():
    if SPLIT_WEBCAM_DIR.exists():
        shutil.rmtree(SPLIT_WEBCAM_DIR)

    print("=== Очистка и стратифицированное хронологическое разделение датасета ===")
    print("Исходные файлы в data/webcam_dataset/ сохранены без изменений.\n")

    total_stats = {"train": 0, "val": 0, "test": 0}
    for cls_name in ["closed", "open"]:
        counts, filtered, n_segs = split_class(cls_name)
        print(
            f"Класс '{cls_name:6s}' ({n_segs} сегм., отфильтровано переходных: {filtered:3d}): "
            f"train = {counts['train']:4d}, "
            f"val = {counts['val']:4d}, "
            f"test = {counts['test']:4d}"
        )
        for k in total_stats:
            total_stats[k] += counts[k]

    print("-" * 72)
    print(
        f"ИТОГО ЧИСТЫХ КАДРОВ: "
        f"train = {total_stats['train']:4d}, "
        f"val = {total_stats['val']:4d}, "
        f"test = {total_stats['test']:4d}"
    )
    print(f"\nГотовый датасет сохранён в: {SPLIT_WEBCAM_DIR}")


if __name__ == "__main__":
    main()
