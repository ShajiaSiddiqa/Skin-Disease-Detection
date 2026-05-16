import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_image_features(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - MEAN) / STD
    return image_array.flatten()


def load_split(split_dir: Path, class_to_idx: dict[str, int], image_size: int):
    features = []
    labels = []

    for class_name, label in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
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


def run_prediction(image_path: str, model_path: str = "simple_linear_model.joblib") -> dict:
    data = joblib.load(model_path)
    scaler = data["scaler"]
    pca = data["pca"]
    model = data["linear_regression"]
    class_to_idx = data["class_to_idx"]
    image_size = data.get("image_size", 64)
    classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    flat = load_image_features(Path(image_path), image_size).reshape(1, -1)
    features = pca.transform(scaler.transform(flat))
    scores = model.predict(features)
    probabilities = softmax(scores)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted": classes[predicted_index],
        "index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {classes[index]: float(probabilities[index]) for index in range(len(classes))},
        "raw_scores": {classes[index]: float(scores[0][index]) for index in range(len(classes))},
        "pca_components": int(pca.n_components_),
        "image_size": image_size,
    }


def train(args):
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    classes = sorted(path.name for path in (data_dir / "train").iterdir() if path.is_dir())
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    num_classes = len(classes)
    feature_dim = args.image_size * args.image_size * 3

    print("Libraries    : scikit-learn + numpy + Pillow")
    print(f"Classes      : {classes}")
    print(f"Image size   : {args.image_size}x{args.image_size} | Raw features: {feature_dim}")
    print("Loading images ...", flush=True)

    x_train, y_train = load_split(data_dir / "train", class_to_idx, args.image_size)
    x_valid, y_valid = load_split(data_dir / "valid", class_to_idx, args.image_size)
    x_test, y_test = load_split(data_dir / "test", class_to_idx, args.image_size)
    print(f"Train / Valid / Test : {len(y_train)} / {len(y_valid)} / {len(y_test)}")

    n_components = min(args.pca_components, x_train.shape[0] - 1, x_train.shape[1] - 1)
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=args.seed)

    x_train_scaled = scaler.fit_transform(x_train)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_valid_pca = pca.transform(scaler.transform(x_valid))
    x_test_pca = pca.transform(scaler.transform(x_test))

    print(
        f"PCA kept {pca.n_components_} components "
        f"({pca.explained_variance_ratio_.sum() * 100:.1f}% variance)"
    )

    print("\nTraining Simple Linear Regression ...")
    linear_model = LinearRegression()
    linear_model.fit(x_train_pca, one_hot(y_train, num_classes))

    valid_scores = linear_model.predict(x_valid_pca)
    valid_probabilities = softmax(valid_scores)
    valid_predictions = np.argmax(valid_scores, axis=1)
    valid_accuracy = accuracy_score(y_valid, valid_predictions)
    valid_loss = log_loss(y_valid, valid_probabilities, labels=list(range(num_classes)))
    print(f"  Validation accuracy : {valid_accuracy * 100:.2f}%")
    print(f"  Validation log loss : {valid_loss:.4f}")

    print("Training Logistic Regression reference ...")
    logistic_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=args.seed,
        class_weight="balanced",
        solver="lbfgs",
    )
    logistic_model.fit(x_train_pca, y_train)
    logistic_valid_accuracy = accuracy_score(y_valid, logistic_model.predict(x_valid_pca))
    print(f"  Logistic Regression validation accuracy : {logistic_valid_accuracy * 100:.2f}%")

    print("\nRetraining linear regression on train+valid ...")
    x_train_valid_pca = np.concatenate([x_train_pca, x_valid_pca])
    y_train_valid = np.concatenate([y_train, y_valid])
    final_model = LinearRegression()
    final_model.fit(x_train_valid_pca, one_hot(y_train_valid, num_classes))

    test_scores = final_model.predict(x_test_pca)
    test_probabilities = softmax(test_scores)
    test_predictions = np.argmax(test_scores, axis=1)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_loss = log_loss(y_test, test_probabilities, labels=list(range(num_classes)))
    report = classification_report(y_test, test_predictions, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, test_predictions, labels=list(range(num_classes))).tolist()

    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "linear_regression": final_model,
            "class_to_idx": class_to_idx,
            "image_size": args.image_size,
            "num_classes": num_classes,
            "libraries": "scikit-learn + numpy + Pillow",
        },
        model_path,
    )

    metrics = {
        "model": "model_e",
        "model_type": "Simple Linear Regression on PCA pixel features",
        "libraries": "scikit-learn + numpy + Pillow",
        "feature_extraction": "Raw resized RGB pixels normalized with ImageNet mean/std, then StandardScaler and PCA",
        "image_size": args.image_size,
        "pca_components": int(pca.n_components_),
        "pca_variance_explained_percent": round(float(pca.explained_variance_ratio_.sum()) * 100, 2),
        "classes": classes,
        "validation_accuracy": float(valid_accuracy),
        "validation_loss": float(valid_loss),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1_score": float(report["macro avg"]["f1-score"]),
        "logistic_regression_valid_accuracy": float(logistic_valid_accuracy),
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run Model E: simple linear regression classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", default="final_dataset")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--image-size", type=int, default=64)
    train_parser.add_argument("--pca-components", type=int, default=150)
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
        print(json.dumps(run_prediction(args.image, args.model), indent=2))


if __name__ == "__main__":
    main()
