
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import hashlib
import csv

# Пути к датасету и результатам
DATA_DIR = Path("data/raw/mrlEyes_2018_01")
RESULTS_DIR = Path("results/eda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Счётчики
total_images = 0
corrupted = []
invalid_labels = []
hash_groups = defaultdict(list)
class_counts = Counter()

# Получаем все изображения
image_paths = sorted(DATA_DIR.glob("s*/*.png"))

print(f"Найдено изображений: {len(image_paths)}")
print("Проверка датасета...")

for image_path in image_paths:
    total_images += 1

    # Получаем класс из названия файла
    parts = image_path.stem.split("_")

    if len(parts) != 8 or parts[4] not in ("0", "1"):
        invalid_labels.append(str(image_path))
        continue

    label = parts[4]
    class_counts[label] += 1

    try:
        # Проверяем, открывается ли изображение
        with Image.open(image_path) as img:
            img.verify()

        # Вычисляем SHA-256 содержимого файла
        sha256 = hashlib.sha256()

        with open(image_path, "rb") as file:
            for chunk in iter(lambda: file.read(65536), b""):
                sha256.update(chunk)

        file_hash = sha256.hexdigest()

        # Сохраняем хеш, класс и путь
        hash_groups[file_hash].append(
            (str(image_path), label)
        )

    except Exception as error:
        corrupted.append((str(image_path), str(error)))

    if total_images % 10000 == 0:
        print(f"Проверено: {total_images}/{len(image_paths)}")

# Ищем одинаковые файлы
duplicate_groups = {
    h: items
    for h, items in hash_groups.items()
    if len(items) > 1
}

duplicate_extra = sum(
    len(items) - 1
    for items in duplicate_groups.values()
)

# Проверяем противоречивые метки
conflicting_groups = {
    h: items
    for h, items in duplicate_groups.items()
    if len({label for _, label in items}) > 1
}

# Сохраняем дубликаты в CSV
with open(
    RESULTS_DIR / "duplicates.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["hash", "path", "label"])

    for file_hash, items in duplicate_groups.items():
        for path, label in items:
            writer.writerow([file_hash, path, label])

# Сохраняем повреждённые файлы
with open(
    RESULTS_DIR / "corrupted.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["path", "error"])
    writer.writerows(corrupted)

# Сохраняем конфликты разметки
with open(
    RESULTS_DIR / "conflicting_labels.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["hash", "path", "label"])

    for file_hash, items in conflicting_groups.items():
        for path, label in items:
            writer.writerow([file_hash, path, label])

# Вывод результатов
print("\nРЕЗУЛЬТАТЫ ПРОВЕРКИ")
print("Всего изображений:", total_images)
print("Closed:", class_counts["0"])
print("Open:", class_counts["1"])
print("Повреждённых файлов:", len(corrupted))
print("Файлов с некорректной разметкой:", len(invalid_labels))
print("Групп точных дубликатов:", len(duplicate_groups))
print("Лишних копий:", duplicate_extra)
print("Групп с противоречивыми метками:", len(conflicting_groups))
print("\nОтчёты сохранены в:", RESULTS_DIR)
