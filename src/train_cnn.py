from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Настройки
# -----------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


# -----------------------------
# Выбор устройства
# -----------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Устройство: {DEVICE}")


# -----------------------------
# Предобработка изображений
# -----------------------------

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Небольшая аугментация
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# -----------------------------
# Загружаем датасеты
# -----------------------------

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=test_transform
)


print("Классы:", train_dataset.class_to_idx)

print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Validation: {len(val_dataset)}"
)

print(
    f"Test: {len(test_dataset)}"
)


# -----------------------------
# DataLoader
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Наша CNN
# -----------------------------

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 64x64 -> 32x32
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32x32 -> 16x16
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16x16 -> 8x8
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


model = EyeCNN().to(DEVICE)

print("\nМодель:")
print(model)


# -----------------------------
# Loss и Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Обучение
# -----------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100
        * train_correct
        / train_total
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100
        * val_correct
        / val_total
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: "
        f"{train_loss / len(train_loader):.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}% | "
        f"Val Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # Сохраняем лучшую модель
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_DIR / "eye_cnn_best.pth"
        )

        print(
            "→ Сохранена новая лучшая модель"
        )


print("\nОбучение завершено.")

print(
    f"Лучшая Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)