import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["chickenpox", "eczema", "ringworm"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_split(split_dir, image_size):
    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(split_dir) / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if not is_image_file(image_path):
                continue

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
                image_array = np.asarray(image, dtype=np.float32) / 255.0

            images.append(image_array)
            labels.append(label)

    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_dataset(data_dir, image_size):
    data_dir = Path(data_dir)
    return (
        load_split(data_dir / "train", image_size),
        load_split(data_dir / "valid", image_size),
        load_split(data_dir / "test", image_size),
    )


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 54 * 54, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def make_loader(features, labels, batch_size, shuffle):
    x_tensor = torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def macro_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def save_plots(history, output_dir):
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train", marker="o")
    plt.plot(epochs, [row["valid_accuracy"] for row in history], label="Validation", marker="o")
    plt.title("Model C Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_c_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train", marker="o")
    plt.plot(epochs, [row["valid_loss"] for row in history], label="Validation", marker="o")
    plt.title("Model C Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_c_loss.png", dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model C: simple two-convolution CNN.")
    parser.add_argument("--data-dir", default="final_dataset")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size != 224:
        raise ValueError("Model C uses a fixed Linear(32 * 54 * 54, 64), so --image-size must be 224.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir) / f"model_c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (train_data, valid_data, test_data) = load_dataset(args.data_dir, args.image_size)
    train_loader = make_loader(*train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = make_loader(*valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(*test_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_valid_accuracy = -1.0
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_accuracy, _, _ = evaluate(model, valid_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "valid_loss": float(valid_loss),
                "valid_accuracy": float(valid_accuracy),
            }
        )

        if valid_accuracy > best_valid_accuracy or (
            valid_accuracy == best_valid_accuracy and valid_loss < best_valid_loss
        ):
            best_valid_accuracy = valid_accuracy
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_accuracy:.4f}"
        )

    model.load_state_dict(best_state_dict)
    test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    macro_precision, macro_recall, macro_f1 = macro_metrics(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    metrics = {
        "model": "model_c",
        "model_type": "Simple two-convolution CNN",
        "class_names": CLASS_NAMES,
        "configuration": vars(args),
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_accuracy": float(best_valid_accuracy),
        "best_valid_loss": float(best_valid_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "confusion_matrix": matrix.tolist(),
    }

    torch.save({"state_dict": model.state_dict(), **metrics}, output_dir / "model_c.pt")
    (output_dir / "model_c_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(history, output_dir)

    print(f"\nModel C test accuracy: {test_accuracy:.4f}")
    print(f"Model C test loss: {test_loss:.4f}")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
