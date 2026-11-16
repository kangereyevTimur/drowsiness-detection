import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
     ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
WEBCAM_TEST_DIR = ROOT / "data" / "webcam_split" / "test"
MRL_TEST_DIR = ROOT / "data" / "processed" / "test"
RESULTS_DIR = ROOT / "results" / "cnn_finetuned"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_CNN_PATH = ROOT / "models" / "eye_cnn_best.pth"
FINETUNED_CNN_PATH = ROOT / "models" / "eye_cnn_finetuned.pth"
MOBILENET_PATH = ROOT / "models" / "mobilenet_best.pth"

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


cnn_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

mobilenet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def evaluate_model(model, test_dir: Path, tf, name: str):
    dataset = datasets.ImageFolder(test_dir, transform=tf)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_preds = []
    all_labels = []

    start_t = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    elapsed = time.perf_counter() - start_t

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    cm = confusion_matrix(all_labels, all_preds)
    fps = len(dataset) / max(elapsed, 1e-6)

    # Посчитаем отдельно точность для левого (L) и правого (R) глаза, если это webcam_split
    side_acc = {}
    for side in ["L", "R"]:
        side_indices = [i for i, (path, _) in enumerate(dataset.samples) if f"_{side}_" in Path(path).name]
        if side_indices:
            side_acc[side] = accuracy_score(all_labels[side_indices], all_preds[side_indices]) * 100

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
        "fps": fps,
        "side_acc": side_acc,
        "report": classification_report(all_labels, all_preds, target_names=dataset.classes, digits=4),
    }


def main():
    print(f"Устройство оценки: {DEVICE}\n")

    # 1. Загружаем модели
    cnn_base = EyeCNN().to(DEVICE)
    cnn_base.load_state_dict(torch.load(BASE_CNN_PATH, map_location="cpu", weights_only=True))
    cnn_base.eval()

    cnn_ft = EyeCNN().to(DEVICE)
    cnn_ft.load_state_dict(torch.load(FINETUNED_CNN_PATH, map_location="cpu", weights_only=True))
    cnn_ft.eval()

    mobilenet = models.mobilenet_v3_small(weights=None)
    in_features = mobilenet.classifier[3].in_features
    mobilenet.classifier[3] = nn.Linear(in_features, 2)
    mobilenet.load_state_dict(torch.load(MOBILENET_PATH, map_location="cpu", weights_only=True))
    mobilenet = mobilenet.to(DEVICE)
    mobilenet.eval()

    # 2. Оценка на отложенной тестовой выборке с веб-камеры (data/webcam_split/test)
    print("=== 1. Сравнение на независимой тестовой выборке с веб-камеры (Webcam Test) ===")
    res_base_web = evaluate_model(cnn_base, WEBCAM_TEST_DIR, cnn_transform, "EyeCNN (До дообучения)")
    res_ft_web = evaluate_model(cnn_ft, WEBCAM_TEST_DIR, cnn_transform, "EyeCNN (После дообучения)")
    res_mob_web = evaluate_model(mobilenet, WEBCAM_TEST_DIR, mobilenet_transform, "MobileNetV3 Small")

    for r in [res_base_web, res_ft_web, res_mob_web]:
        side_str = f" | L: {r['side_acc'].get('L', 0):.1f}%, R: {r['side_acc'].get('R', 0):.1f}%" if r["side_acc"] else ""
        print(
            f"{r['name']:26s} -> Acc: {r['accuracy']:6.2f}% | F1: {r['f1']:6.2f}% | "
            f"Prec: {r['precision']:6.2f}% | Rec: {r['recall']:6.2f}%{side_str} | Speed: {r['fps']:.0f} img/s"
        )

    # 3. Оценка на тестовой выборке MRL (сбалансированный срез 2000 изображений для быстрой проверки)
    print("\n=== 2. Проверка сохранения знаний на MRL Eye Dataset Test ===")
    mrl_ds = datasets.ImageFolder(MRL_TEST_DIR, transform=cnn_transform)
    rng = np.random.default_rng(42)
    subset_idx = rng.choice(len(mrl_ds), size=min(2000, len(mrl_ds)), replace=False)
    mrl_loader = DataLoader(torch.utils.data.Subset(mrl_ds, subset_idx), batch_size=256, shuffle=False)

    def eval_subset(model, name):
        preds, labels = [], []
        with torch.no_grad():
            for x, y in mrl_loader:
                preds.extend(model(x.to(DEVICE)).argmax(dim=1).cpu().numpy())
                labels.extend(y.numpy())
        return {
            "name": name,
            "accuracy": accuracy_score(labels, preds) * 100,
            "f1": f1_score(labels, preds, average="macro") * 100,
        }

    res_base_mrl = eval_subset(cnn_base, "EyeCNN (До дообучения)")
    res_ft_mrl = eval_subset(cnn_ft, "EyeCNN (После дообучения)")

    for r in [res_base_mrl, res_ft_mrl]:
        print(f"{r['name']:26s} -> MRL Test Acc: {r['accuracy']:6.2f}% | F1: {r['f1']:6.2f}%")

    # 4. Сохранение текстового отчета
    report_path = RESULTS_DIR / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== СРАВНЕНИЕ МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ ВЕБ-КАМЕРЫ (WEBCAM TEST) ===\n\n")
        for r in [res_base_web, res_ft_web, res_mob_web]:
            f.write(f"Модель: {r['name']}\n")
            f.write(f"Accuracy : {r['accuracy']:.2f}%\n")
            f.write(f"Precision: {r['precision']:.2f}%\n")
            f.write(f"Recall   : {r['recall']:.2f}%\n")
            f.write(f"F1-score : {r['f1']:.2f}%\n")
            if r["side_acc"]:
                f.write(f"Точность Левый глаз (L): {r['side_acc']['L']:.2f}%\n")
                f.write(f"Точность Правый глаз (R): {r['side_acc']['R']:.2f}%\n")
            f.write(f"Confusion Matrix:\n{r['cm']}\n\n")
            f.write(f"{r['report']}\n" + "-" * 60 + "\n\n")

        f.write("=== ПРОВЕРКА НА MRL EYE DATASET TEST ===\n")
        for r in [res_base_mrl, res_ft_mrl]:
            f.write(f"{r['name']}: Accuracy = {r['accuracy']:.2f}%, F1 = {r['f1']:.2f}%\n")

    # 5. Визуализация Confusion Matrix до и после дообучения
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, r in zip(axes, [res_base_web, res_ft_web, res_mob_web]):
        disp = ConfusionMatrixDisplay(confusion_matrix=r["cm"], display_labels=["closed", "open"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{r['name']}\nAcc: {r['accuracy']:.1f}% | F1: {r['f1']:.1f}%")
    plt.tight_layout()
    cm_plot_path = RESULTS_DIR / "webcam_test_confusion_matrices.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    print(f"\nОтчёт сохранён в: {report_path}")
    print(f"Матрицы ошибок сохранены в: {cm_plot_path}")


if __name__ == "__main__":
    main()

