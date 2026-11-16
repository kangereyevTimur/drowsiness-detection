from pathlib import Path
from collections import Counter
import shutil
import random


# Исходный датасет
DATASET_DIR = Path("data/raw/mrlEyes_2018_01")

# Подготовленный датасет
OUTPUT_DIR = Path("data/processed")

RANDOM_SEED = 42

# Сколько людей используем в каждой выборке
TRAIN_SIZE = 27
VAL_SIZE = 5
TEST_SIZE = 5

# Сколько случайных вариантов разделения проверяем
SEARCH_ITERATIONS = 10000


def get_eye_state(filename: str) -> str:
    """
    Получает состояние глаза из имени файла MRL Eye.

    0 = closed
    1 = open
    """

    parts = filename.split("_")
    eye_state = parts[4]

    if eye_state == "1":
        return "open"

    return "closed"


def count_subject_images(subject_dir: Path) -> Counter:
    """
    Считает Open и Closed изображения
    для одного участника.
    """

    counts = Counter()

    for image_path in subject_dir.glob("*.png"):
        label = get_eye_state(image_path.name)
        counts[label] += 1

    return counts


def calculate_split_counts(subjects, subject_stats):
    """
    Считает количество Open и Closed
    для группы участников.
    """

    counts = Counter()

    for subject in subjects:
        counts.update(subject_stats[subject.name])

    return counts


def balance_score(counts: Counter) -> float:
    """
    Чем меньше значение, тем лучше баланс.

    0 означает идеальный баланс:
    Open = Closed.
    """

    open_count = counts["open"]
    closed_count = counts["closed"]

    total = open_count + closed_count

    if total == 0:
        return float("inf")

    return abs(open_count - closed_count) / total


def find_best_split(subjects, subject_stats):
    """
    Ищет разделение участников,
    при котором классы Open/Closed
    наиболее сбалансированы.
    """

    random.seed(RANDOM_SEED)

    best_score = float("inf")
    best_split = None

    for _ in range(SEARCH_ITERATIONS):

        shuffled = subjects.copy()
        random.shuffle(shuffled)

        train_subjects = shuffled[:TRAIN_SIZE]

        val_subjects = shuffled[
            TRAIN_SIZE:
            TRAIN_SIZE + VAL_SIZE
        ]

        test_subjects = shuffled[
            TRAIN_SIZE + VAL_SIZE:
        ]

        train_counts = calculate_split_counts(
            train_subjects,
            subject_stats
        )

        val_counts = calculate_split_counts(
            val_subjects,
            subject_stats
        )

        test_counts = calculate_split_counts(
            test_subjects,
            subject_stats
        )

        # Особое внимание Validation и Test,
        # потому что они используются для оценки модели.
        score = (
            balance_score(train_counts)
            + balance_score(val_counts) * 2
            + balance_score(test_counts) * 2
        )

        if score < best_score:

            best_score = score

            best_split = (
                train_subjects,
                val_subjects,
                test_subjects
            )

    return best_split


def copy_subject(subject_dir: Path, split: str):
    """
    Копирует изображения участника
    в нужную выборку.
    """

    for image_path in subject_dir.glob("*.png"):

        label = get_eye_state(image_path.name)

        destination = (
            OUTPUT_DIR
            / split
            / label
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


def print_split_info(name, subjects, subject_stats):

    counts = calculate_split_counts(
        subjects,
        subject_stats
    )

    open_count = counts["open"]
    closed_count = counts["closed"]

    total = open_count + closed_count

    open_percent = open_count / total * 100
    closed_percent = closed_count / total * 100

    print(f"\n{name}")

    print(
        f"Open:   {open_count} "
        f"({open_percent:.1f}%)"
    )

    print(
        f"Closed: {closed_count} "
        f"({closed_percent:.1f}%)"
    )

    print(f"Всего:  {total}")

    print(
        "Участники:",
        ", ".join(
            subject.name
            for subject in subjects
        )
    )


def main():

    subjects = sorted([
        path
        for path in DATASET_DIR.iterdir()
        if path.is_dir()
        and path.name.startswith("s")
    ])

    print(
        f"Всего участников: {len(subjects)}"
    )

    # Считаем статистику каждого человека
    subject_stats = {}

    for subject in subjects:

        subject_stats[subject.name] = (
            count_subject_images(subject)
        )

    print(
        "Поиск сбалансированного "
        "разделения..."
    )

    train_subjects, val_subjects, test_subjects = (
        find_best_split(
            subjects,
            subject_stats
        )
    )

    print_split_info(
        "TRAIN",
        train_subjects,
        subject_stats
    )

    print_split_info(
        "VALIDATION",
        val_subjects,
        subject_stats
    )

    print_split_info(
        "TEST",
        test_subjects,
        subject_stats
    )

    # Удаляем старый processed
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Создаем новые папки
    for split in [
        "train",
        "val",
        "test"
    ]:

        for label in [
            "open",
            "closed"
        ]:

            (
                OUTPUT_DIR
                / split
                / label
            ).mkdir(
                parents=True,
                exist_ok=True
            )

    print(
        "\nКопирование изображений..."
    )

    for subject in train_subjects:
        copy_subject(
            subject,
            "train"
        )

    for subject in val_subjects:
        copy_subject(
            subject,
            "val"
        )

    for subject in test_subjects:
        copy_subject(
            subject,
            "test"
        )

    print(
        "\nДатасет успешно подготовлен."
    )


if __name__ == "__main__":
    main()