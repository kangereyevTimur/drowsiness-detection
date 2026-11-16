
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import csv
import math

# Файл с найденными конфликтами
CSV_PATH = Path("results/eda/conflicting_labels.csv")
OUTPUT_DIR = Path("results/eda/conflict_review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Группируем изображения по SHA-256
groups = defaultdict(list)

with open(CSV_PATH, encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        groups[row["hash"]].append(row)

print("Групп с конфликтами:", len(groups))

# Создаём отдельную картинку для каждой группы
for number, (file_hash, items) in enumerate(groups.items(), 1):

    fig, axes = plt.subplots(
        1, len(items),
        figsize=(4 * len(items), 4),
        squeeze=False
    )

    for ax, item in zip(axes[0], items):
        image_path = Path(item["path"])

        with Image.open(image_path) as img:
            ax.imshow(img.convert("L"), cmap="gray")

        label = "Open" if item["label"] == "1" else "Closed"

        ax.set_title(
            f"{label}\n{image_path.name}",
            fontsize=9
        )
        ax.axis("off")

    fig.suptitle(f"Conflict group {number}")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"conflict_{number:02d}.png",
        dpi=130
    )
    plt.close()

print("Изображения сохранены в:", OUTPUT_DIR)
