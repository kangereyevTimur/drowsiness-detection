import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
MRL_DIR = ROOT / "data" / "processed"
WEBCAM_DIR = ROOT / "data" / "webcam_split"
BASE_MODEL_PATH = ROOT / "models" / "eye_cnn_best.pth"
FINETUNED_MODEL_PATH = ROOT / "models" / "eye_cnn_finetuned.pth"

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0003
RANDOM_SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class EyeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Расширенная аугментация для Этапа 2: яркость, контраст, повороты, масштабирование
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.35, contrast=0.35),
    transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.88, 1.12)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Базовая модель не найдена: {BASE_MODEL_PATH}")
    if not (WEBCAM_DIR / "train").exists():
        raise FileNotFoundError("Сначала запустите python src/prepare_webcam_dataset.py")

    print(f"Устройство обучения: {DEVICE}")
    print(f"Базовая модель: {BASE_MODEL_PATH}")

    # 1. Загружаем данные веб-камеры
    webcam_train = datasets.ImageFolder(WEBCAM_DIR / "train", transform=train_transform)
    webcam_val = datasets.ImageFolder(WEBCAM_DIR / "val", transform=eval_transform)

    # Повторяем обучающую выборку веб-камеры 2 раза в эпохе (с разной случайной аугментацией)
    webcam_train_oversampled = ConcatDataset([webcam_train, webcam_train])

    # 2. Подмешиваем сбалансированную подвыборку из MRL (Experience Replay против забывания)
    mrl_train_full = datasets.ImageFolder(MRL_DIR / "train", transform=train_transform)
    mrl_indices = random.sample(range(len(mrl_train_full)), min(1500, len(mrl_train_full)))
    mrl_train_subset = Subset(mrl_train_full, mrl_indices)

    combined_train = ConcatDataset([webcam_train_oversampled, mrl_train_subset])

    train_loader = DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    webcam_val_loader = DataLoader(webcam_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Классы: {webcam_train.class_to_idx}")
    print(f"Webcam Train (x3 aug): {len(webcam_train_oversampled)} | MRL Replay: {len(mrl_train_subset)}")
    print(f"Webcam Validation: {len(webcam_val)}\n")

    # Загружаем предобученную модель
    model = EyeCNN().to(DEVICE)
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location="cpu", weights_only=True))

    _, init_val_acc = evaluate(model, webcam_val_loader)
    print(f"Точность на Webcam Validation ДО дообучения: {init_val_acc:.2f}%\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / train_total
        val_loss, val_acc = evaluate(model, webcam_val_loader)

        marker = ""
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, FINETUNED_MODEL_PATH)
            marker = " <-- Best saved"

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Webcam Val Loss: {val_loss:.4f} | Webcam Val Acc: {val_acc:.2f}%{marker}"
        )

    print(f"\nДообучение завершено! Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"Дообученная модель сохранена в: {FINETUNED_MODEL_PATH}")


if __name__ == "__main__":
    main()

