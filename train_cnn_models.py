import argparse
import copy
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from model_a import build_model_a
from model_b import build_model_b


CLASS_NAMES = ["chickenpox", "eczema", "ringworm"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare two custom PyTorch CNN models for skin disease detection."
    )
    parser.add_argument("--data-dir", default="final_dataset", help="Dataset root folder.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="results", help="Folder for outputs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument(
        "--loss",
        choices=["cross-entropy", "focal"],
        default="cross-entropy",
        help="Training loss. Focal loss emphasizes hard examples.",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="Focal loss gamma.")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable horizontal-flip test-time augmentation during final evaluation.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in CrossEntropyLoss.",
    )
    parser.add_argument(
        "--models",
        choices=["a", "b", "both"],
        default="both",
        help="Choose which model to train.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device: cpu, cuda, or cuda:0.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dataset(data_dir):
    data_dir = Path(data_dir)
    split_counts = {}

    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        split_counts[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")

            count = len([p for p in class_dir.iterdir() if is_image_file(p)])
            split_counts[split][class_name] = count

    return split_counts


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_image_array(path, image_size=None):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size is not None and image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image = np.asarray(image, dtype=np.float32) / 255.0

    return image


def random_resized_crop(image, output_size, scale=(0.90, 1.0)):
    width, height = image.size
    crop_scale = random.uniform(*scale)
    crop_size = max(1, int(min(width, height) * crop_scale))
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((output_size, output_size), Image.Resampling.BILINEAR)


def augment_image(image, image_size):
    image = random_resized_crop(image, image_size)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.rotate(random.uniform(-15, 15), resample=Image.Resampling.BILINEAR)
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.90, 1.10))
    return image


def make_output_dir(base_dir):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"cnn_comparison_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class SkinDiseaseDataset(Dataset):
    def __init__(self, split_dir, image_size, mean, std, augment=False):
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.augment = augment
        self.samples = []

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = self.split_dir / class_name
            for path in sorted(class_dir.iterdir()):
                if is_image_file(path):
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No images found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.augment:
                image = augment_image(image, self.image_size)
            elif image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

            image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std
        image_array = np.ascontiguousarray(image_array, dtype=np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor, torch.tensor(label, dtype=torch.long)


def calculate_train_normalization(train_dir, image_size):
    sums = np.zeros(3, dtype=np.float64)
    squared_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for class_name in CLASS_NAMES:
        for path in sorted((Path(train_dir) / class_name).iterdir()):
            if not is_image_file(path):
                continue
            image = load_image_array(path, image_size=image_size)
            pixels = image.reshape(-1, 3)
            sums += pixels.sum(axis=0)
            squared_sums += np.square(pixels).sum(axis=0)
            pixel_count += pixels.shape[0]

    mean = sums / pixel_count
    std = np.sqrt((squared_sums / pixel_count) - np.square(mean))
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def load_datasets(data_dir, image_size, batch_size, seed):
    data_dir = Path(data_dir)
    generator = torch.Generator().manual_seed(seed)
    mean, std = calculate_train_normalization(data_dir / "train", image_size)

    train_dataset = SkinDiseaseDataset(data_dir / "train", image_size, mean, std, augment=True)
    valid_dataset = SkinDiseaseDataset(data_dir / "valid", image_size, mean, std, augment=False)
    test_dataset = SkinDiseaseDataset(data_dir / "test", image_size, mean, std, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader, mean, std


def plot_history(history, model_name, output_dir):
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Training Accuracy")
    plt.plot(epochs, history["valid_accuracy"], marker="o", label="Validation Accuracy")
    plt.title(f"{model_name}: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.plot(epochs, history["valid_loss"], marker="o", label="Validation Loss")
    plt.title(f"{model_name}: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss.png", dpi=160)
    plt.close()


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(CLASS_NAMES)
        },
    }


def calculate_macro_f1(y_true, y_pred):
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    return float(macro_f1)


def calculate_class_weights(train_loader, device):
    labels = [label for _, label in train_loader.dataset.samples]
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        cross_entropy = F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-cross_entropy.detach())
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * cross_entropy).mean()


def build_criterion(loss_name, class_weights, focal_gamma):
    if loss_name == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma, label_smoothing=0.05)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)


def plot_confusion_matrix(matrix, model_name, output_dir):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name}: Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=25, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close()


def run_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return (
        total_loss / len(data_loader.dataset),
        accuracy_score(y_true, y_pred),
        calculate_macro_f1(y_true, y_pred),
    )


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return (
        total_loss / len(data_loader.dataset),
        accuracy_score(y_true, y_pred),
        calculate_macro_f1(y_true, y_pred),
    )


def collect_predictions(model, test_loader, device, mean, std, use_tta=False):
    model.eval()
    images = []
    y_true = []
    probabilities = []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            device_images = batch_images.to(device)
            logits = model(device_images)
            if use_tta:
                flipped_logits = model(torch.flip(device_images, dims=[3]))
                logits = (logits + flipped_logits) / 2.0
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()

            images.append(tensor_batch_to_images(batch_images, mean, std))
            y_true.extend(batch_labels.numpy().tolist())
            probabilities.append(batch_probs)

    images = np.concatenate(images, axis=0)
    y_true = np.asarray(y_true)
    probabilities = np.concatenate(probabilities, axis=0)
    y_pred = probabilities.argmax(axis=1)
    return images, y_true, y_pred, probabilities


def tensor_batch_to_images(batch_images, mean, std):
    images = batch_images.permute(0, 2, 3, 1).numpy()
    images = (images * std.reshape(1, 1, 1, 3)) + mean.reshape(1, 1, 1, 3)
    return np.clip(images * 255.0, 0, 255).astype("uint8")


def save_sample_predictions(images, y_true, y_pred, probabilities, model_name, output_dir):
    correct_indices = np.where(y_true == y_pred)[0][:9]
    incorrect_indices = np.where(y_true != y_pred)[0][:9]

    save_prediction_grid(
        images,
        y_true,
        y_pred,
        probabilities,
        correct_indices,
        model_name,
        output_dir / f"{model_name}_correct_predictions.png",
        "Correct Predictions",
    )
    save_prediction_grid(
        images,
        y_true,
        y_pred,
        probabilities,
        incorrect_indices,
        model_name,
        output_dir / f"{model_name}_incorrect_predictions.png",
        "Incorrect Predictions",
    )


def save_prediction_grid(
    images,
    y_true,
    y_pred,
    probabilities,
    indices,
    model_name,
    path,
    title,
):
    if len(indices) == 0:
        plt.figure(figsize=(6, 2))
        plt.axis("off")
        plt.title(f"{model_name}: {title}")
        plt.text(0.5, 0.5, "No samples found", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return

    rows = int(np.ceil(len(indices) / 3))
    plt.figure(figsize=(10, rows * 3.2))
    for plot_idx, sample_idx in enumerate(indices, start=1):
        plt.subplot(rows, 3, plot_idx)
        plt.imshow(images[sample_idx])
        plt.axis("off")
        true_name = CLASS_NAMES[int(y_true[sample_idx])]
        pred_name = CLASS_NAMES[int(y_pred[sample_idx])]
        confidence = probabilities[sample_idx, y_pred[sample_idx]]
        plt.title(f"True: {true_name}\nPred: {pred_name} ({confidence:.2f})", fontsize=9)

    plt.suptitle(f"{model_name}: {title}", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_metrics_csv(all_metrics, output_dir):
    path = output_dir / "model_comparison_metrics.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"])
        for model_name, metrics in all_metrics.items():
            writer.writerow(
                [
                    model_name,
                    metrics["accuracy"],
                    metrics["macro_precision"],
                    metrics["macro_recall"],
                    metrics["macro_f1_score"],
                ]
            )


def train_and_evaluate(
    model_name,
    model,
    train_loader,
    valid_loader,
    test_loader,
    output_dir,
    epochs,
    device,
    optimizer,
    mean,
    std,
    patience,
    use_class_weights,
    loss_name,
    focal_gamma,
    use_tta,
):
    print(f"\n{'=' * 72}")
    print(f"Training {model_name}")
    print(f"{'=' * 72}")
    print(model)

    model = model.to(device)
    class_weights = calculate_class_weights(train_loader, device) if use_class_weights else None
    criterion = build_criterion(loss_name, class_weights, focal_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, patience // 3),
    )
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_macro_f1": [],
    }
    best_valid_accuracy = -1.0
    best_valid_macro_f1 = -1.0
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, train_macro_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        valid_loss, valid_accuracy, valid_macro_f1 = evaluate(model, valid_loader, criterion, device)
        scheduler.step(valid_macro_f1)

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))
        history["train_macro_f1"].append(float(train_macro_f1))
        history["valid_loss"].append(float(valid_loss))
        history["valid_accuracy"].append(float(valid_accuracy))
        history["valid_macro_f1"].append(float(valid_macro_f1))

        improved = valid_macro_f1 > best_valid_macro_f1
        tied_with_lower_loss = valid_macro_f1 == best_valid_macro_f1 and valid_loss < best_valid_loss
        if improved or tied_with_lower_loss:
            best_valid_accuracy = valid_accuracy
            best_valid_macro_f1 = valid_macro_f1
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} train_f1={train_macro_f1:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_accuracy:.4f} valid_f1={valid_macro_f1:.4f} "
            f"best_valid_f1={best_valid_macro_f1:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best validation macro F1 was epoch {best_epoch}.")
            break

    model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "history": history,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_valid_accuracy,
            "best_valid_macro_f1": best_valid_macro_f1,
            "best_valid_loss": best_valid_loss,
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
            "loss_name": loss_name,
            "focal_gamma": focal_gamma if loss_name == "focal" else None,
            "test_time_augmentation": use_tta,
            "normalization": {
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
        },
        output_dir / f"{model_name}.pt",
    )
    plot_history(history, model_name, output_dir)

    images, y_true, y_pred, probabilities = collect_predictions(
        model,
        test_loader,
        device,
        mean,
        std,
        use_tta=use_tta,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    metrics = calculate_metrics(y_true, y_pred)

    plot_confusion_matrix(matrix, model_name, output_dir)
    save_sample_predictions(images, y_true, y_pred, probabilities, model_name, output_dir)

    with (output_dir / f"{model_name}_metrics.json").open("w") as file:
        json.dump(
            {
                "metrics": metrics,
                "confusion_matrix": matrix.tolist(),
                "class_names": CLASS_NAMES,
                "history": history,
                "best_epoch": best_epoch,
                "best_valid_accuracy": best_valid_accuracy,
                "best_valid_macro_f1": best_valid_macro_f1,
                "best_valid_loss": best_valid_loss,
                "loss_name": loss_name,
                "focal_gamma": focal_gamma if loss_name == "focal" else None,
                "test_time_augmentation": use_tta,
            },
            file,
            indent=2,
        )

    print(f"\n{model_name} test metrics:")
    print(
        f"Best epoch: {best_epoch} with validation macro F1 {best_valid_macro_f1:.4f} "
        f"and validation accuracy {best_valid_accuracy:.4f}"
    )
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall:    {metrics['macro_recall']:.4f}")
    print(f"F1-score:  {metrics['macro_f1_score']:.4f}")
    print("Confusion matrix:")
    print(matrix)
    return metrics


def write_run_summary(output_dir, args, split_counts, all_metrics, mean, std):
    best_model = max(all_metrics, key=lambda name: all_metrics[name]["macro_f1_score"])
    summary = {
        "configuration": vars(args),
        "dataset_split_counts": split_counts,
        "class_names": CLASS_NAMES,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "frameworks": {
            "training": "PyTorch",
            "metrics": "scikit-learn",
        },
        "models": {
            "custom_cnn_a": {
                "architecture": "VGG-style plain Conv2d blocks with BatchNorm, MaxPool2d, Dropout2d, and average/max global pooling",
                "activation": "ReLU hidden layers, raw logits output",
                "optimizer": f"AdamW learning_rate={args.learning_rate} weight_decay={args.weight_decay}",
                "loss": f"{args.loss} with label_smoothing=0.05 and optional inverse-frequency class weights",
            },
            "custom_cnn_b": {
                "architecture": "Deeper plain Conv2d CNN stages with BatchNorm, MaxPool2d, and average/max global pooling",
                "activation": "SiLU hidden layers, raw logits output",
                "optimizer": f"AdamW learning_rate={args.learning_rate} weight_decay={args.weight_decay}",
                "loss": f"{args.loss} with label_smoothing=0.05 and optional inverse-frequency class weights",
            },
        },
        "metrics": all_metrics,
        "best_model_by_macro_f1": best_model,
    }

    with (output_dir / "run_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "README_results.txt").open("w") as file:
        file.write("Skin Disease Detection CNN Comparison\n")
        file.write("=====================================\n\n")
        file.write("Framework: PyTorch training with scikit-learn metrics\n")
        file.write(f"Epochs: {args.epochs}\n")
        file.write(f"Batch size: {args.batch_size}\n")
        file.write(f"Image size: {args.image_size}x{args.image_size}\n")
        file.write(f"Early stopping patience: {args.patience}\n")
        file.write(f"Loss: {args.loss}\n")
        file.write(f"Test-time augmentation: {args.tta}\n")
        file.write("Classes: chickenpox, eczema, ringworm\n\n")
        file.write("Best model by macro F1-score: " + best_model + "\n\n")
        file.write("See PNG files for accuracy/loss curves, confusion matrices, and qualitative samples.\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    split_counts = check_dataset(args.data_dir)
    output_dir = make_output_dir(args.output_dir)

    print("Dataset structure:")
    for split, counts in split_counts.items():
        print(f"  {split}: {counts} total={sum(counts.values())}")
    print(f"Device: {device}")
    print(f"Outputs will be saved to: {output_dir}")

    train_loader, valid_loader, test_loader, mean, std = load_datasets(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.seed,
    )
    print(f"Train normalization mean: {mean.round(4).tolist()}")
    print(f"Train normalization std:  {std.round(4).tolist()}")

    model_a = build_model_a(args.image_size, len(CLASS_NAMES))
    model_b = build_model_b(args.image_size, len(CLASS_NAMES))
    training_jobs = {}
    if args.models in ("a", "both"):
        training_jobs["custom_cnn_a"] = (
            model_a,
            torch.optim.AdamW(
                model_a.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
        )
    if args.models in ("b", "both"):
        training_jobs["custom_cnn_b"] = (
            model_b,
            torch.optim.AdamW(
                model_b.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
        )

    all_metrics = {}
    for model_name, (model, optimizer) in training_jobs.items():
        all_metrics[model_name] = train_and_evaluate(
            model_name,
            model,
            train_loader,
            valid_loader,
            test_loader,
            output_dir,
            args.epochs,
            device,
            optimizer,
            mean,
            std,
            args.patience,
            not args.no_class_weights,
            args.loss,
            args.focal_gamma,
            args.tta,
        )

    save_metrics_csv(all_metrics, output_dir)
    write_run_summary(output_dir, args, split_counts, all_metrics, mean, std)

    print("\nFinal comparison:")
    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['macro_precision']:.4f}, "
            f"recall={metrics['macro_recall']:.4f}, "
            f"f1={metrics['macro_f1_score']:.4f}"
        )
    print(f"\nAll artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
