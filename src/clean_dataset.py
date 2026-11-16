
from pathlib import Path
import csv
import shutil

# Пути к данным
PROCESSED_DIR = Path("data/processed")
CONFLICTS_CSV = Path("results/eda/conflicting_labels.csv")
BACKUP_DIR = Path("data/conflict_backup")

# Проверяем наличие необходимых файлов
if not CONFLICTS_CSV.exists():
    raise FileNotFoundError("Не найден conflicting_labels.csv")

if not PROCESSED_DIR.exists():
    raise FileNotFoundError("Не найдена папка data/processed")

# Читаем список конфликтных изображений
conflict_names = set()
conflict_groups = set()

with open(CONFLICTS_CSV, encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        conflict_names.add(Path(row["path"]).name)
        conflict_groups.add(row["hash"])

print("Конфликтных групп:", len(conflict_groups))
print("Конфликтных файлов:", len(conflict_names))

# Ищем конфликтные изображения в подготовленном датасете
files_to_remove = []

for split in ["train", "val", "test"]:
    for label in ["open", "closed"]:
        folder = PROCESSED_DIR / split / label

        for image_path in folder.glob("*.png"):
            if image_path.name in conflict_names:
                files_to_remove.append(image_path)

print("Найдено в processed:", len(files_to_remove))

# Создаём резервную копию
for image_path in files_to_remove:
    relative_path = image_path.relative_to(PROCESSED_DIR)
    backup_path = BACKUP_DIR / relative_path

    backup_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(image_path, backup_path)

print("Резервные копии сохранены.")

# Удаляем конфликтные изображения из processed
for image_path in files_to_remove:
    image_path.unlink()

print("Удалено конфликтных изображений:", len(files_to_remove))

# Проверяем количество оставшихся изображений
for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    total = 0

    for label in ["open", "closed"]:
        folder = PROCESSED_DIR / split / label
        count = len(list(folder.glob("*.png")))

        print(f"{label}: {count}")
        total += count

    print("Всего:", total)

print("\nОчистка завершена.")
print("Оригинальный датасет не изменён.")
