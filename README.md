# Skin Disease Detection

PyTorch project for training and comparing custom CNN models on skin disease image classes. The training code uses Pillow/NumPy for image preprocessing and scikit-learn for evaluation metrics.

The current training pipeline supports three classes:

- `chickenpox`
- `eczema`
- `ringworm`

This project is for coursework and experimentation only. It is not a medical diagnostic tool.

## Project Structure

```text
.
├── train_cnn_models.py      # Main PyTorch training/evaluation pipeline
├── model_a.py               # Improved VGG-style PyTorch CNN architecture
├── model_b.py               # Deeper plain PyTorch CNN architecture
├── main.py                  # Image resize/preprocessing helper
├── main2.py                 # Dataset split helper
├── pipeline.py              # Legacy TXT-based dataset loader
├── textcreation.py          # Generates train/valid/test TXT manifests
├── scripts/
│   └── generate_project_report.py
├── reports/
│   └── skin_disease_detection_report.pdf
├── requirements.txt         # Python dependencies
├── dataset/                 # Raw class images, ignored by Git
├── processed_dataset/       # Resized/preprocessed images, ignored by Git
├── final_dataset/           # Train/valid/test image folders, ignored by Git
├── results/                 # Training outputs, ignored by Git
├── train.txt                # Optional legacy manifest
├── valid.txt                # Optional legacy manifest
└── test.txt                 # Optional legacy manifest
```

## Dataset Layout

The main training script expects this folder structure:

```text
final_dataset/
├── train/
│   ├── chickenpox/
│   ├── eczema/
│   └── ringworm/
├── valid/
│   ├── chickenpox/
│   ├── eczema/
│   └── ringworm/
└── test/
    ├── chickenpox/
    ├── eczema/
    └── ringworm/
```

Each class folder should contain image files for that split.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses PyTorch for model training and scikit-learn for metrics. It does not use TensorFlow, Keras, transformers, attention modules, pretrained models, or ResNet/residual skip connections.

## Train and Compare Models

Run the full training and evaluation pipeline:

```bash
python train_cnn_models.py
```

Useful options:

```bash
python train_cnn_models.py \
  --data-dir final_dataset \
  --epochs 60 \
  --batch-size 8 \
  --image-size 224 \
  --patience 14 \
  --output-dir results
```

The script trains both CNN models, restores the best validation-accuracy checkpoint for each model, evaluates that checkpoint on the test split, and writes artifacts to a timestamped folder under `results/`.

The current training pipeline includes:

- Training-split RGB mean/std normalization
- Pillow bilinear resizing instead of nearest-neighbor resizing
- Random crop, horizontal flip, rotation, brightness, contrast, and color augmentation for training images
- AdamW optimization with weight decay
- Reduce-on-plateau learning-rate scheduling
- Cross-entropy loss by default, with label smoothing and inverse-frequency class weights
- Best-checkpoint selection by validation macro F1-score
- Optional horizontal-flip test-time augmentation during final evaluation
- Early stopping controlled by `--patience`

Both models are custom CNNs trained from scratch. On a small dataset of only a few hundred training images, 80%+ test accuracy is not guaranteed. If accuracy remains limited, the most reliable allowed improvement is expanding and cleaning the dataset.

You can train only one model during experiments:

```bash
python train_cnn_models.py --models a
python train_cnn_models.py --models b
python train_cnn_models.py --models both
```

Class weights can be disabled for ablation runs:

```bash
python train_cnn_models.py --models a --no-class-weights
```

Focal loss can be used for comparison:

```bash
python train_cnn_models.py --models a --loss focal
```

The strongest measured Model A run used:

```bash
python -u train_cnn_models.py \
  --models a \
  --image-size 224 \
  --batch-size 8 \
  --epochs 60 \
  --patience 14
```

That run reached `69.09%` test accuracy and `0.6831` macro F1. The best validation accuracy reached `72.73%`.

## Model Architectures

### Model A: Improved VGG-Style CNN

Model A is a plain CNN. It does not use residual connections, attention, transformers, or pretrained layers.

```text
Input: RGB image

Conv Block 1:
Conv2d 3 -> 16, BatchNorm2d, ReLU
Conv2d 16 -> 16, BatchNorm2d, ReLU
MaxPool2d, Dropout2d

Conv Block 2:
Conv2d 16 -> 32, BatchNorm2d, ReLU
Conv2d 32 -> 32, BatchNorm2d, ReLU
MaxPool2d, Dropout2d

Conv Block 3:
Conv2d 32 -> 64, BatchNorm2d, ReLU
Conv2d 64 -> 64, BatchNorm2d, ReLU
MaxPool2d, Dropout2d

Conv Block 4:
Conv2d 64 -> 128, BatchNorm2d, ReLU
Conv2d 128 -> 128, BatchNorm2d, ReLU
MaxPool2d, Dropout2d

Global pooling:
AdaptiveAvgPool2d + AdaptiveMaxPool2d

Classifier:
Linear 256 -> 128, BatchNorm1d, ReLU, Dropout
Linear 128 -> 64, BatchNorm1d, ReLU, Dropout
Linear 64 -> 3 classes
```

The main architecture improvement over the earlier Model A is the combined average and max global pooling, which gives the classifier both overall texture information and the strongest detected lesion features. The training pipeline also uses class-weighted cross-entropy loss and selects the saved checkpoint by validation macro F1-score.

Latest measured Model A metrics:

| Metric | Value |
|---|---:|
| Test accuracy | `69.09%` |
| Macro precision | `0.6865` |
| Macro recall | `0.6876` |
| Macro F1-score | `0.6831` |
| Best validation accuracy | `72.73%` |
| Best validation macro F1 | `0.7259` |
| Validation loss at best epoch | `0.8066` |
| Training loss at best epoch | `0.7602` |
| Final training loss | `0.7558` |
| Final validation loss | `0.7675` |

### Model B: Deeper Plain CNN

Model B is also a plain CNN. It uses standard Conv2d stages only, without residual connections or attention.

```text
Input: RGB image

Initial Conv:
Conv2d 3 -> 32, BatchNorm2d, SiLU

Plain CNN Stages:
32 -> 48, two Conv2d layers, MaxPool2d, Dropout2d
48 -> 72, two Conv2d layers, MaxPool2d, Dropout2d
72 -> 96, two Conv2d layers, MaxPool2d, Dropout2d
96 -> 128, two Conv2d layers, MaxPool2d, Dropout2d
128 -> 160, Conv2d, BatchNorm2d, SiLU, Dropout2d

Global pooling:
AdaptiveAvgPool2d + AdaptiveMaxPool2d

Classifier:
Linear 320 -> 128, BatchNorm1d, SiLU, Dropout
Linear 128 -> 64, BatchNorm1d, SiLU, Dropout
Linear 64 -> 3 classes
```

Latest measured Model B metrics:

| Metric | Value |
|---|---:|
| Test accuracy | `54.55%` |
| Macro precision | `0.5148` |
| Macro recall | `0.5303` |
| Macro F1-score | `0.5127` |
| Best validation accuracy | `69.09%` |
| Validation loss at best epoch | `0.8986` |
| Training loss at best epoch | `0.7954` |
| Final training loss | `0.7016` |
| Final validation loss | `0.8800` |

### Model Comparison

| Model | Test Accuracy | Macro F1 | Best Validation Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| Model A | `69.09%` | `0.6831` | `72.73%` | `0.8066` |
| Model B | `54.55%` | `0.5127` | `69.09%` | `0.8986` |

Model A is currently the better model for this dataset.

## Outputs

Each training run creates files such as:

- Saved PyTorch checkpoints: `custom_cnn_a.pt`, `custom_cnn_b.pt`
- Accuracy and loss plots
- Confusion matrices
- Correct and incorrect prediction sample grids
- Per-model metrics JSON files
- `model_comparison_metrics.csv`
- `run_summary.json`

Each checkpoint also stores the best epoch, best validation accuracy, class names, training history, and normalization values needed for inference.

## Project Report

The generated PDF report is included at:

```text
reports/skin_disease_detection_report.pdf
```

To regenerate it after new experiments:

```bash
python scripts/generate_project_report.py
```

## Helper Scripts

- `main.py`: resizes images from a raw class folder into `processed_dataset/`.
- `main2.py`: splits `processed_dataset/` into `final_dataset/train`, `final_dataset/valid`, and `final_dataset/test`.
- `textcreation.py`: creates TXT manifests for the legacy loader in `pipeline.py`.
- `pipeline.py`: loads images from TXT manifests into NumPy arrays for legacy experiments.

Check and adjust folder names inside these helper scripts before running them on a new dataset.

## Notes

- Raw datasets, processed datasets, trained models, and result images are ignored by Git.
- Keep reproducible source code and dependency files in Git.
- Store large datasets or model artifacts separately if they need to be shared.

## Push to GitHub from VS Code

1. Open this project folder in VS Code.
2. Open the Source Control panel from the left sidebar.
3. If Git is not initialized, click `Initialize Repository`.
4. Create a new empty repository on GitHub. Do not add a README there because this project already has one.
5. In VS Code terminal, connect the local repo to GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git branch -M main
```

6. Stage the files from the Source Control panel, or run:

```bash
git add .
```

7. Commit:

```bash
git commit -m "Initial skin disease detection project"
```

8. Push:

```bash
git push -u origin main
```

After this, future updates can be pushed from VS Code using Source Control: stage changes, write a commit message, click Commit, then Sync/Push.
