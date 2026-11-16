
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

# Пути к данным и сохранённой модели
DATA_DIR = Path("data/processed/test")
MODEL_PATH = Path("models/eye_cnn_best.pth")
RESULTS_DIR = Path("results/cnn")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 64

# Используем GPU Mac M1, если он доступен
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Такая же обработка изображений, как при обучении
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Архитектура должна совпадать с обученной CNN
class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Загружаем лучшую модель
model = EyeCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Устройство:", DEVICE)
print("Тестовых изображений:", len(test_dataset))
print("Классы:", test_dataset.class_to_idx)

# Получаем предсказания модели
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

# Вычисляем метрики
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, average="macro", zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, average="macro", zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, average="macro", zero_division=0
)

print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

# Отчёт отдельно по каждому классу
class_names = test_dataset.classes

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
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
    display_labels=class_names
)

display.plot(cmap="Blues", values_format="d")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Сохраняем текстовый отчёт
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
