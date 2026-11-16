
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Пути к данным и результатам
DATA_DIR = Path("data/raw/mrlEyes_2018_01")
RESULTS_DIR = Path("results/eda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

records = []
image_sizes = Counter()
image_modes = Counter()
examples = {}

# Изучаем изображения всех участников
for subject_dir in sorted(DATA_DIR.glob("s*")):
    if not subject_dir.is_dir():
        continue

    for image_path in subject_dir.glob("*.png"):
        parts = image_path.stem.split("_")

        if len(parts) != 8:
            continue

        # Извлекаем метки из имени файла
        record = {
            "subject": subject_dir.name,
            "gender": int(parts[2]),
            "glasses": int(parts[3]),
            "eye_state": int(parts[4]),
            "reflections": int(parts[5]),
            "quality": int(parts[6]),
            "sensor": parts[7],
        }

        records.append(record)

        # Сохраняем примеры двух классов
        label = record["eye_state"]
        if label not in examples:
            examples[label] = image_path

        # Проверяем размер и цветовой режим
        with Image.open(image_path) as img:
            image_sizes[img.size] += 1
            image_modes[img.mode] += 1

# Превращаем результаты в таблицу
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("Изображения не найдены. Проверь DATA_DIR.")

print("\nОБЩАЯ ИНФОРМАЦИЯ")
print("Изображений:", len(df))
print("Участников:", df["subject"].nunique())

# Названия признаков
features = {
    "eye_state": {0: "Closed", 1: "Open"},
    "gender": {0: "Male", 1: "Female"},
    "glasses": {0: "No glasses", 1: "Glasses"},
    "reflections": {0: "None", 1: "Low", 2: "High"},
    "quality": {0: "Bad", 1: "Good"},
    "sensor": {
        "01": "RealSense",
        "02": "IDS",
        "03": "Aptina"
    }
}

# Статистика и графики каждого признака
for column, mapping in features.items():
    counts = df[column].map(mapping).value_counts()

    print(f"\n{column.upper()}")
    print(counts.to_string())

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title(f"Distribution: {column}")
    plt.xlabel(column)
    plt.ylabel("Number of images")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{column}.png")
    plt.close()

# Распределение Open/Closed по участникам
subject_counts = pd.crosstab(
    df["subject"],
    df["eye_state"]
).rename(columns={0: "Closed", 1: "Open"})

subject_counts.plot(
    kind="bar",
    stacked=True,
    figsize=(15, 6)
)

plt.title("Eye states by subject")
plt.xlabel("Subject")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "subjects.png")
plt.close()

print("\nРАЗМЕРЫ ИЗОБРАЖЕНИЙ")
for size, count in image_sizes.most_common(10):
    print(size, ":", count)

print("\nЦВЕТОВЫЕ РЕЖИМЫ")
print(dict(image_modes))

# График наиболее распространённых размеров
top_sizes = image_sizes.most_common(15)
size_labels = [str(size) for size, _ in top_sizes]
size_counts = [count for _, count in top_sizes]

plt.figure(figsize=(12, 5))
plt.bar(size_labels, size_counts)
plt.title("Most common image sizes")
plt.xlabel("Image size")
plt.ylabel("Number of images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "image_sizes.png")
plt.close()

# Показываем реальные примеры Open и Closed
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, label in zip(axes, [0, 1]):
    if label in examples:
        with Image.open(examples[label]) as img:
            ax.imshow(img.convert("L"), cmap="gray")
        ax.set_title("Closed" if label == 0 else "Open")
    ax.axis("off")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "examples.png")
plt.close()

# Сохраняем таблицы для дальнейшего анализа
df.to_csv(
    RESULTS_DIR / "dataset_metadata.csv",
    index=False
)

subject_counts.to_csv(
    RESULTS_DIR / "subject_distribution.csv"
)

print("\nАнализ завершён!")
print("Результаты сохранены в:", RESULTS_DIR)
