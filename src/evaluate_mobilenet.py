
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Настройки
DATA_DIR = Path("data/processed/test")
MODEL_PATH = Path("models/mobilenet_best.pth")
RESULTS_DIR = Path("results/mobilenet")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 160
BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Предобработка должна совпадать с обучением
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Загружаем тестовые изображения
test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Устройство:", DEVICE)
print("Тестовых изображений:", len(test_dataset))
print("Классы:", test_dataset.class_to_idx)

# Создаём ту же архитектуру MobileNetV3 Small
# weights=None, потому что обученные веса загрузим из файла
model = models.mobilenet_v3_small(weights=None)

# Меняем выходной слой на 2 класса
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)

# Загружаем лучшую обученную модель
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model = model.to(DEVICE)
model.eval()

# Предсказания
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

# Метрики
accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

recall = recall_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

f1 = f1_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

print("\nРЕЗУЛЬТАТЫ MOBILENETV3")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# Матрица ошибок
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

display.plot(cmap="Blues", values_format="d")
plt.title("MobileNetV3 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# Сохраняем результаты
with open(
    RESULTS_DIR / "evaluation.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Test images: {len(test_dataset)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Precision (macro): {precision * 100:.2f}%\n")
    file.write(f"Recall (macro): {recall * 100:.2f}%\n")
    file.write(f"F1-score (macro): {f1 * 100:.2f}%\n\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(cm))

print("\nРезультаты сохранены в:", RESULTS_DIR)
