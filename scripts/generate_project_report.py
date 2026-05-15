import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "skin_disease_detection_report.pdf"

MODEL_A_RUN = ROOT / "results" / "cnn_comparison_20260515_213452"
MODEL_B_RUN = ROOT / "results" / "cnn_comparison_20260515_200631"
CLASS_NAMES = ["chickenpox", "eczema", "ringworm"]


def load_json(path):
    with path.open() as file:
        return json.load(file)


def add_header(fig, title):
    fig.text(0.06, 0.955, title, fontsize=18, fontweight="bold", va="top")
    fig.text(0.06, 0.93, "Skin Disease Detection CNN Project", fontsize=9, color="#555555", va="top")
    fig.lines.append(
        plt.Line2D([0.06, 0.94], [0.912, 0.912], transform=fig.transFigure, color="#222222", linewidth=0.8)
    )


def wrapped_lines(text, width=96):
    lines = []
    for paragraph in text.strip().split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width))
    return lines


def text_page(pdf, title, sections):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    add_header(fig, title)
    y = 0.875
    for section_title, body in sections:
        fig.text(0.07, y, section_title, fontsize=13, fontweight="bold", va="top")
        y -= 0.03
        for line in wrapped_lines(body):
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                add_header(fig, title + " (continued)")
                y = 0.875
            fig.text(0.08, y, line, fontsize=10, va="top")
            y -= 0.021
        y -= 0.018
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(pdf, title, table_title, columns, rows, note=None):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    add_header(fig, title)
    fig.text(0.07, 0.875, table_title, fontsize=13, fontweight="bold", va="top")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="upper center",
        cellLoc="center",
        bbox=[0.07, 0.49, 0.86, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")
    if note:
        fig.text(0.08, 0.45, note, fontsize=9, color="#444444", va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_page(pdf, title, image_paths):
    fig, axes = plt.subplots(len(image_paths), 1, figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    if len(image_paths) == 1:
        axes = [axes]
    add_header(fig, title)
    for ax, (caption, path) in zip(axes, image_paths):
        ax.axis("off")
        if path.exists():
            image = Image.open(path)
            ax.imshow(image)
            ax.set_title(caption, fontsize=11, pad=8)
        else:
            ax.text(0.5, 0.5, f"Missing image: {path.name}", ha="center", va="center")
    plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.9])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pipeline_figure(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    add_header(fig, "Complete Pipeline (Figure)")
    steps = [
        "Raw disease images",
        "Preprocess / resize",
        "Train / valid / test split",
        "Dataset normalization",
        "Training augmentation",
        "Model A / Model B CNN",
        "Validation macro F1 checkpoint",
        "Test evaluation",
        "Metrics + plots + predictions",
    ]
    x = 0.5
    ys = np.linspace(0.82, 0.18, len(steps))
    for i, (step, y) in enumerate(zip(steps, ys)):
        box = plt.Rectangle((0.22, y - 0.025), 0.56, 0.05, fill=True, color="#F1F4F8", ec="#2F3A45", lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, step, ha="center", va="center", fontsize=11, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x, ys[i + 1] + 0.03),
                xytext=(x, y - 0.03),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#2F3A45"),
            )
    ax.text(
        0.5,
        0.10,
        "The pipeline is implemented with PyTorch for CNN training and scikit-learn for evaluation metrics.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def format_percent(value):
    return f"{value * 100:.2f}%"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_a = load_json(MODEL_A_RUN / "custom_cnn_a_metrics.json")
    model_b = load_json(MODEL_B_RUN / "custom_cnn_b_metrics.json")
    summary_a = load_json(MODEL_A_RUN / "run_summary.json")

    split_counts = summary_a["dataset_split_counts"]
    total_images = sum(sum(class_counts.values()) for class_counts in split_counts.values())

    with PdfPages(OUTPUT_PATH) as pdf:
        text_page(
            pdf,
            "Project Title",
            [
                (
                    "Skin Disease Detection Using Custom CNN Models",
                    "A PyTorch-based image classification project for detecting chickenpox, eczema, and ringworm from skin images. The work compares two custom CNN architectures trained from scratch under coursework constraints: no TensorFlow/Keras, no pretrained models, no transformers, no attention modules, and no ResNet/residual skip connections.",
                )
            ],
        )

        text_page(
            pdf,
            "Introduction",
            [
                (
                    "Overview",
                    "Skin disease recognition from images is a computer vision task where visual patterns such as color, lesion texture, shape, and local contrast are used to classify disease categories. This project builds a supervised image classification pipeline for three skin disease classes: chickenpox, eczema, and ringworm.",
                ),
                (
                    "Objective",
                    "The objective is to train and compare custom CNN models, evaluate them on held-out test data, and analyze both quantitative results and qualitative prediction examples.",
                ),
            ],
        )

        text_page(
            pdf,
            "Applications (Motivation)",
            [
                (
                    "Motivation",
                    "A lightweight image classifier can support educational skin-disease screening experiments, demonstrate CNN-based visual feature learning, and provide a reproducible coursework pipeline for preprocessing, training, validation, and evaluation.",
                ),
                (
                    "Important Limitation",
                    "This project is not a medical diagnostic tool. It is intended for academic experimentation only. Real clinical deployment would require expert-labeled data, broader disease coverage, bias analysis, and medical validation.",
                ),
            ],
        )

        dataset_rows = []
        for split in ["train", "valid", "test"]:
            counts = split_counts[split]
            dataset_rows.append(
                [
                    split,
                    counts["chickenpox"],
                    counts["eczema"],
                    counts["ringworm"],
                    sum(counts.values()),
                ]
            )
        table_page(
            pdf,
            "Dataset Collection Details",
            "Dataset Split and Class Counts",
            ["Split", "Chickenpox", "Eczema", "Ringworm", "Total"],
            dataset_rows,
            note=f"Total images used in this project: {total_images}. The dataset is organized in final_dataset/train, final_dataset/valid, and final_dataset/test.",
        )

        text_page(
            pdf,
            "Pre-processing (if any)",
            [
                (
                    "Image Preparation",
                    "Images are loaded as RGB, resized to a square input size, normalized using training-set RGB mean and standard deviation, and converted to PyTorch tensors in channel-first format.",
                ),
                (
                    "Training Augmentation",
                    "During training, images use random resized cropping, horizontal flipping, small rotations, brightness adjustment, contrast adjustment, and color adjustment. Validation and test images are not randomly augmented.",
                ),
                (
                    "Normalization Values",
                    f"For the latest Model A run, mean={summary_a['normalization']['mean']} and std={summary_a['normalization']['std']}.",
                ),
            ],
        )

        text_page(
            pdf,
            "Model Training (Methodology)",
            [
                (
                    "Frameworks",
                    "Training is implemented in PyTorch. Accuracy, precision, recall, F1-score, and confusion matrices are computed with scikit-learn.",
                ),
                (
                    "Optimization",
                    "The models use AdamW optimization, weight decay, label smoothing, inverse-frequency class weights, ReduceLROnPlateau scheduling, and early stopping. Checkpoints are selected using validation macro F1-score so that weaker classes are considered during model selection.",
                ),
                (
                    "Model A",
                    "Model A is a VGG-style plain CNN with standard Conv2d blocks, BatchNorm2d, ReLU, MaxPool2d, Dropout2d, and combined AdaptiveAvgPool2d plus AdaptiveMaxPool2d before the classifier.",
                ),
                (
                    "Model B",
                    "Model B is a deeper plain CNN using standard Conv2d stages, BatchNorm2d, SiLU, MaxPool2d, Dropout2d, and combined global average/max pooling. It does not use residual connections or attention.",
                ),
            ],
        )

        pipeline_figure(pdf)

        text_page(
            pdf,
            "Experimental Details",
            [
                (
                    "Latest Model A Run",
                    "Run folder: results/cnn_comparison_20260515_213452. Command: python -u train_cnn_models.py --models a --image-size 224 --batch-size 8 --epochs 60 --patience 14.",
                ),
                (
                    "Latest Model B Run",
                    "Run folder: results/cnn_comparison_20260515_200631. This is the latest available completed run that includes Model B metrics.",
                ),
                (
                    "Evaluation Protocol",
                    "The models are evaluated on the held-out test split. Reported accuracy and macro F1 are test-set metrics, while validation loss and validation macro F1 are used for checkpoint selection.",
                ),
            ],
        )

        a_metrics = model_a["metrics"]
        b_metrics = model_b["metrics"]
        result_rows = [
            [
                "Model A",
                format_percent(a_metrics["accuracy"]),
                f"{a_metrics['macro_precision']:.4f}",
                f"{a_metrics['macro_recall']:.4f}",
                f"{a_metrics['macro_f1_score']:.4f}",
                f"{model_a.get('best_valid_loss', model_a['history']['valid_loss'][model_a['best_epoch'] - 1]):.4f}",
            ],
            [
                "Model B",
                format_percent(b_metrics["accuracy"]),
                f"{b_metrics['macro_precision']:.4f}",
                f"{b_metrics['macro_recall']:.4f}",
                f"{b_metrics['macro_f1_score']:.4f}",
                f"{model_b.get('best_valid_loss') or model_b['history']['valid_loss'][model_b['best_epoch'] - 1]:.4f}",
            ],
        ]
        table_page(
            pdf,
            "Results (Quantitative / Qualitative)",
            "Quantitative Test Results",
            ["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Val Loss"],
            result_rows,
            note="Model A is currently the stronger model by test accuracy and macro F1-score.",
        )

        class_rows = []
        for class_name in CLASS_NAMES:
            a_class = a_metrics["per_class"][class_name]
            b_class = b_metrics["per_class"][class_name]
            class_rows.append(
                [
                    class_name,
                    f"{a_class['recall']:.4f}",
                    f"{a_class['f1_score']:.4f}",
                    f"{b_class['recall']:.4f}",
                    f"{b_class['f1_score']:.4f}",
                ]
            )
        table_page(
            pdf,
            "Results (Quantitative / Qualitative)",
            "Per-Class Recall and F1-score",
            ["Class", "A Recall", "A F1", "B Recall", "B F1"],
            class_rows,
        )

        image_page(
            pdf,
            "Results (Quantitative / Qualitative)",
            [
                ("Model A Confusion Matrix", MODEL_A_RUN / "custom_cnn_a_confusion_matrix.png"),
                ("Model B Confusion Matrix", MODEL_B_RUN / "custom_cnn_b_confusion_matrix.png"),
            ],
        )
        image_page(
            pdf,
            "Results (Quantitative / Qualitative)",
            [
                ("Model A Accuracy Curve", MODEL_A_RUN / "custom_cnn_a_accuracy.png"),
                ("Model A Loss Curve", MODEL_A_RUN / "custom_cnn_a_loss.png"),
            ],
        )
        image_page(
            pdf,
            "Results (Quantitative / Qualitative)",
            [
                ("Model A Correct Prediction Samples", MODEL_A_RUN / "custom_cnn_a_correct_predictions.png"),
                ("Model A Incorrect Prediction Samples", MODEL_A_RUN / "custom_cnn_a_incorrect_predictions.png"),
            ],
        )

        text_page(
            pdf,
            "Conclusion",
            [
                (
                    "Summary",
                    f"Model A achieved the best latest test performance with {format_percent(a_metrics['accuracy'])} accuracy and {a_metrics['macro_f1_score']:.4f} macro F1-score. Model B achieved {format_percent(b_metrics['accuracy'])} accuracy and {b_metrics['macro_f1_score']:.4f} macro F1-score in its latest completed run.",
                ),
                (
                    "Observation",
                    "The results show that the improved VGG-style CNN is more effective for the current dataset than the deeper plain CNN. Eczema remains the hardest class, indicating that more high-quality eczema samples or improved dataset cleaning would likely help more than simply increasing model complexity.",
                ),
                (
                    "Future Work",
                    "Future improvements should focus on collecting more balanced images, removing mislabeled or low-quality samples, testing multiple random splits, and increasing data diversity while staying within the coursework restrictions.",
                ),
            ],
        )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
