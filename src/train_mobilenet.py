
from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Настройки обучения
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.0001

# Используем GPU Mac M1
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Устройство:", DEVICE)

# Преобразования для предобученной модели
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Загружаем обучающую и валидационную выборки
train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Классы:", train_dataset.class_to_idx)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

# Загружаем веса MobileNetV3, обученные на ImageNet
weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(weights=weights)

# Сначала замораживаем извлечение признаков
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Меняем последний слой на классификацию двух классов
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)

model = model.to(DEVICE)

# Обучаем только классификатор
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()

best_val_accuracy = 0.0

# Обучение
for epoch in range(EPOCHS):
    model.train()

    # Замороженные слои оставляем в режиме оценки
    model.features.eval()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total
    average_loss = train_loss / train_total

    # Проверка на Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Val Accuracy: {val_accuracy:.2f}%",
        flush=True
    )

    # Сохраняем лучшую модель
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_DIR / "mobilenet_best.pth"
        )

        print("Сохранена новая лучшая модель", flush=True)

print("\nОбучение завершено.")
print(f"Лучшая Validation Accuracy: {best_val_accuracy:.2f}%")
