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
├── model_c.py               # Simple two-layer PyTorch CNN training script
├── model_d.py               # Fully connected PyTorch ANN/MLP training and prediction script
├── model_e.py               # scikit-learn linear regression training and prediction script
├── pipeline.py              # Legacy TXT-based dataset loader
├── reports/
│   └── skin_disease_detection_report.pdf
├── requirements.txt         # Python dependencies
├── dataset/                 # Raw class images, ignored by Git
├── processed_dataset/       # Resized/preprocessed images, ignored by Git
├── final_dataset/           # Train/valid/test image folders, ignored by Git
└── results/                 # Training outputs, ignored by Git

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

The script trains Model A and Model B, restores the best validation macro-F1 checkpoint for each model, evaluates that checkpoint on the test split, and writes artifacts to a timestamped folder under `results/`.

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

All five models are trained from scratch or from raw pixel features. On a small dataset of only a few hundred training images, 80%+ test accuracy is not guaranteed. If accuracy remains limited, the most reliable allowed improvement is expanding and cleaning the dataset.

You can train only one model during experiments:

```bash
python train_cnn_models.py --models a
python train_cnn_models.py --models b
python train_cnn_models.py --models c --image-size 224
python train_cnn_models.py --models d --image-size 128
python train_cnn_models.py --models e --linear-image-size 8
python train_cnn_models.py --models both
python train_cnn_models.py --models all --image-size 224
```

The standalone scripts are also kept for direct experiments:

```bash
python model_c.py --data-dir final_dataset --epochs 30 --batch-size 16 --image-size 224
python model_d.py train --data-dir final_dataset --img-size 128 --batch-size 32 --epochs 50 --patience 8
python model_e.py train --data-dir final_dataset --image-size 8
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

Model B is also a plain CNN. It uses standard Conv2d stages only, without residual connections, attention, or dropout. Dropout was removed because the small training split was already strongly regularized by augmentation, class weighting, label smoothing, and AdamW weight decay.

```text
Input: RGB image

Initial Conv:
Conv2d 3 -> 32, BatchNorm2d, SiLU

Plain CNN Stages:
32 -> 48, two Conv2d layers, MaxPool2d
48 -> 72, two Conv2d layers, MaxPool2d
72 -> 96, two Conv2d layers, MaxPool2d
96 -> 128, two Conv2d layers, MaxPool2d
128 -> 160, Conv2d, BatchNorm2d, SiLU

Global pooling:
AdaptiveAvgPool2d + AdaptiveMaxPool2d

Classifier:
Linear 320 -> 128, BatchNorm1d, SiLU
Linear 128 -> 64, BatchNorm1d, SiLU
Linear 64 -> 3 classes
```

Recommended Model B command for the 75-80% target range:

```bash
python train_cnn_models.py --models b --image-size 128 --epochs 25 --patience 8
```

Prior Model B runs with the same dataset and `128x128` images reached `78.18%` test accuracy over 50 epochs and `83.64%` test accuracy over 25 epochs. Because the test split has only 55 images, each prediction changes accuracy by about 1.82 percentage points, so reruns can move noticeably with the same code.

Latest measured Model B metrics after removing dropout:

| Metric | Value |
|---|---:|
| Test accuracy | `74.55%` |
| Macro precision | `0.7444` |
| Macro recall | `0.7398` |
| Macro F1-score | `0.7408` |
| Best validation accuracy | `73.64%` |
| Validation loss at best epoch | `0.7276` |
| Training loss at best epoch | `0.6069` |
| Final training loss | `0.5570` |
| Final validation loss | `0.8758` |

### Model C: Simple CNN

Model C is the direct simple CNN implementation: two convolution layers, max pooling, one hidden fully connected layer, and a final 3-class output layer. It uses raw resized `224x224` images scaled to `[0, 1]`, no augmentation, no BatchNorm, no dropout, and no class weighting.

```text
Input: RGB image, 224x224
Conv2d 3 -> 16, ReLU, MaxPool2d
Conv2d 16 -> 32, ReLU, MaxPool2d
Flatten
Linear 32*54*54 -> 64, ReLU
Linear 64 -> 3 classes
```

Latest measured Model C metrics:

| Metric | Value |
|---|---:|
| Test accuracy | `56.36%` |
| Test loss | `1.6501` |
| Macro F1-score | `0.5399` |
| Best validation accuracy | `56.36%` |
| Best validation loss | `1.3903` |

### Model D: ANN / MLP

Model D is a fully connected artificial neural network. It flattens each `128x128` RGB image directly into a vector and classifies it with dense layers. It has many more parameters than the CNNs because the first linear layer receives every image pixel.

```text
Input: RGB image, 128x128
Flatten
Linear 49152 -> 1024, ReLU, BatchNorm1d, Dropout
Linear 1024 -> 512, ReLU, BatchNorm1d, Dropout
Linear 512 -> 128, ReLU, Dropout
Linear 128 -> 3 classes
```

Latest measured Model D metrics:

| Metric | Value |
|---|---:|
| Test accuracy | `49.09%` |
| Test loss | `1.0320` |
| Macro F1-score | `0.4399` |
| Best validation accuracy | `52.73%` |
| Best validation loss | `1.0031` |
| Trainable parameters | `50,926,595` |

### Model E: Simple Linear Regression

Model E is a deliberately weak non-neural baseline. Linear regression is not a natural multiclass image-classification model. It is included only to show how a simple raw-pixel regression baseline compares against CNN and ANN models. Each image is resized to `8x8`, flattened, fitted with `LinearRegression` on one-hot class targets, and predicted with `argmax`.

```text
Input: RGB image, 8x8
Scale pixels to [0, 1]
Flatten raw pixels
LinearRegression on one-hot class targets
Argmax over 3 output scores
```

Latest measured Model E metrics:

| Metric | Value |
|---|---:|
| Test accuracy | `45.45%` |
| Test loss | `1.1966` |
| Macro F1-score | `0.4306` |
| Validation accuracy | `43.64%` |
| Validation loss | `1.1261` |

### Key Differences

| Model | Implementation type | Main feature extractor | Regularization / preprocessing | Epochs run | Best epoch |
|---|---|---|---|---:|---:|
| Model A | PyTorch CNN | Four VGG-style conv blocks with global avg+max pooling | Augmentation, BatchNorm, Dropout, AdamW, class weights | `60` | `41` |
| Model B | PyTorch CNN | Deeper plain conv stages with global avg+max pooling | Augmentation, BatchNorm, AdamW, class weights; dropout removed | `25` | `22` |
| Model C | PyTorch CNN | Two basic conv layers and a large flatten layer | Raw pixel scaling only in standalone run | `30` | `9` |
| Model D | PyTorch ANN | Dense layers over flattened pixels | BatchNorm and Dropout | `11` with early stopping | `3` |
| Model E | scikit-learn baseline | Raw `8x8` pixels and LinearRegression | No deep learning; no true epochs | `1 fit` | `1` |

### Model Comparison

| Rank | Model | Test Accuracy | Macro F1 | Best/Final Validation Accuracy | Validation Loss | Test Loss |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Model B | `74.55%` | `0.7408` | `73.64%` | `0.7276` | Not saved |
| 2 | Model A | `69.09%` | `0.6831` | `72.73%` | `0.8066` | Not saved |
| 3 | Model C | `56.36%` | `0.5399` | `56.36%` | `1.3903` | `1.6501` |
| 4 | Model D | `49.09%` | `0.4399` | `52.73%` | `1.0031` | `1.0320` |
| 5 | Model E | `45.45%` | `0.4306` | `43.64%` | `1.1261` | `1.1966` |

Model B is currently the best measured implementation because it has the highest test accuracy, highest macro F1, and lowest validation loss. Model A is second. Model C is third by accuracy, but its high validation/test loss shows overfitting. Model D has lower loss than Model C but weaker accuracy and F1. Model E is last because linear regression is only a simple raw-pixel baseline, not an appropriate primary multiclass image classifier.

## Outputs

Each unified training run creates files such as:

- Saved PyTorch checkpoints for Models A-D
- Saved `joblib` artifact for Model E
- Accuracy and loss plots
- Confusion matrices
- Correct and incorrect prediction sample grids
- Per-model metrics JSON files
- `model_comparison_metrics.csv`
- `run_summary.json`

PyTorch checkpoints store the best epoch, best validation accuracy, class names, training history, and normalization values needed for inference. The Model E `joblib` file stores the fitted linear regression model, class names, and image size.

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
