import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss


CLASS_NAMES = ["chickenpox", "eczema", "ringworm"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_image_features(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array.flatten()


def load_split(split_dir: Path, image_size: int):
    features = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if is_image_file(image_path):
                features.append(load_image_features(image_path, image_size))
                labels.append(label)

    return np.stack(features), np.asarray(labels)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(labels), num_classes), dtype=np.float64)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def train(args):
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)

    x_train, y_train = load_split(data_dir / "train", args.image_size)
    x_valid, y_valid = load_split(data_dir / "valid", args.image_size)
    x_test, y_test = load_split(data_dir / "test", args.image_size)

    print("Model E: simple raw-pixel LinearRegression baseline")
    print("This is not a natural multiclass image classifier; it is included only as a weak baseline.")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Train / Valid / Test: {len(y_train)} / {len(y_valid)} / {len(y_test)}")

    model = LinearRegression()
    model.fit(x_train, one_hot(y_train, len(CLASS_NAMES)))

    valid_scores = model.predict(x_valid)
    valid_probabilities = softmax(valid_scores)
    valid_predictions = valid_scores.argmax(axis=1)
    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    valid_loss = log_loss(y_valid, valid_probabilities, labels=list(range(len(CLASS_NAMES))))

    test_scores = model.predict(x_test)
    test_probabilities = softmax(test_scores)
    test_predictions = test_scores.argmax(axis=1)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_loss = log_loss(y_test, test_probabilities, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_test,
        test_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, test_predictions, labels=list(range(len(CLASS_NAMES))))

    joblib.dump(
        {
            "model": model,
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
            "model_type": "raw_pixel_linear_regression",
        },
        args.model_path,
    )

    metrics = {
        "model": "model_e",
        "model_type": "Simple raw-pixel LinearRegression baseline",
        "note": "Linear regression is not a natural multiclass image classifier; prediction uses argmax over one-hot regression scores.",
        "image_size": args.image_size,
        "epochs_run": 1,
        "validation_accuracy": float(valid_accuracy),
        "validation_loss": float(valid_loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1_score": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Validation accuracy: {valid_accuracy:.4f}")
    print(f"Validation loss:     {valid_loss:.4f}")
    print(f"Test accuracy:       {test_accuracy:.4f}")
    print(f"Test loss:           {test_loss:.4f}")
    print(f"Saved model:         {args.model_path}")
    print(f"Saved metrics:       {args.metrics_path}")


def predict(args):
    data = joblib.load(args.model)
    model = data["model"]
    class_names = data["class_names"]
    image_size = int(data["image_size"])
    features = load_image_features(Path(args.image), image_size).reshape(1, -1)
    scores = model.predict(features)
    probabilities = softmax(scores)[0]
    predicted_index = int(np.argmax(probabilities))

    print(
        json.dumps(
            {
                "predicted": class_names[predicted_index],
                "confidence": float(probabilities[predicted_index]),
                "probabilities": {
                    class_name: float(probabilities[index])
                    for index, class_name in enumerate(class_names)
                },
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run Model E: simple linear regression baseline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", default="final_dataset")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--image-size", type=int, default=8)
    train_parser.add_argument("--model-path", default="simple_linear_model.joblib")
    train_parser.add_argument("--metrics-path", default="simple_linear_metrics.json")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--model", default="simple_linear_model.joblib")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
