from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["chickenpox", "eczema", "ringworm"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_split(split_dir: Path) -> list[tuple[str, int]]:
    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if is_image_file(image_path):
                samples.append((str(image_path), label))
    if not samples:
        raise ValueError(f"No images found in {split_dir}")
    return samples


class SkinDiseaseDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], img_size: int) -> None:
        self.samples = samples
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            image_tensor = image_tensor.reshape(self.img_size, self.img_size, 3)
            image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        return image_tensor, torch.tensor(label, dtype=torch.long)


class ANNClassifier(nn.Module):
    def __init__(self, img_size: int, num_classes: int) -> None:
        super().__init__()
        input_features = 3 * img_size * img_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.45),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_loader(samples: list[tuple[str, int]], img_size: int, batch_size: int, training: bool) -> DataLoader:
    dataset = SkinDiseaseDataset(samples, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=0)


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


def save_history(history, output_dir):
    with (output_dir / "model_d_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_accuracy", "valid_loss", "valid_accuracy"])
        writer.writeheader()
        writer.writerows(history)


def save_plots(history, output_dir):
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Train", marker="o")
    plt.plot(epochs, [row["valid_accuracy"] for row in history], label="Validation", marker="o")
    plt.title("Model D Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_d_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train", marker="o")
    plt.plot(epochs, [row["valid_loss"] for row in history], label="Validation", marker="o")
    plt.title("Model D Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_d_loss.png", dpi=150)
    plt.close()


def parse_train_args(parser):
    parser.add_argument("--data-dir", default="final_dataset")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def parse_predict_args(parser):
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)


def train(args):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / f"model_d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(collect_split(data_dir / "train"), args.img_size, args.batch_size, True)
    valid_loader = make_loader(collect_split(data_dir / "valid"), args.img_size, args.batch_size, False)
    test_loader = make_loader(collect_split(data_dir / "test"), args.img_size, args.batch_size, False)

    device = torch.device(args.device)
    model = ANNClassifier(img_size=args.img_size, num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=4)

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(model)

    best_valid_loss = float("inf")
    best_valid_accuracy = -1.0
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_accuracy, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step(valid_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "train_accuracy": round(float(train_accuracy), 6),
                "valid_loss": round(float(valid_loss), 6),
                "valid_accuracy": round(float(valid_accuracy), 6),
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_accuracy:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_accuracy = valid_accuracy
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation-loss improvement.")
            break

    model.load_state_dict(best_state_dict)
    test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    metrics = {
        "model": "model_d",
        "model_type": "Fully connected ANN / MLP",
        "class_names": CLASS_NAMES,
        "configuration": vars(args),
        "trainable_parameters": count_parameters(model),
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_accuracy": float(best_valid_accuracy),
        "best_valid_loss": float(best_valid_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1_score": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "img_size": args.img_size,
            "class_names": CLASS_NAMES,
            "metrics": metrics,
        },
        output_dir / "skin_disease_ann.pth",
    )
    (output_dir / "model_d_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_history(history, output_dir)
    save_plots(history, output_dir)

    print(f"\nModel D test loss: {test_loss:.4f}")
    print(f"Model D test accuracy: {test_accuracy:.4f}")
    print(f"Artifacts saved in: {output_dir}")


@torch.no_grad()
def predict(args):
    model_path = Path(args.model)
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    img_size = int(checkpoint["img_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANNClassifier(img_size=img_size, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SkinDiseaseDataset([(args.image, 0)], img_size=img_size)
    image, _ = dataset[0]
    logits = model(image.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    confidence, predicted_index = torch.max(probabilities, dim=0)

    print(f"Image: {args.image}")
    print(f"Predicted class: {class_names[int(predicted_index)]}")
    print(f"Confidence score: {confidence.item():.4f}")
    print("\nClass probabilities:")
    for class_name, probability in zip(class_names, probabilities.tolist()):
        print(f"  {class_name}: {probability:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train or run Model D: fully connected ANN.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    parse_train_args(train_parser)
    predict_parser = subparsers.add_parser("predict")
    parse_predict_args(predict_parser)
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
