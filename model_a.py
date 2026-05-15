import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, dropout=0.0):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class CustomCNNA(nn.Module):
    """VGG-style plain CNN with average and max global pooling."""

    def __init__(self, image_size, num_classes):
        super().__init__()
        del image_size

        self.features = nn.Sequential(
            conv_block(3, 16, dropout=0.03),
            conv_block(16, 32, dropout=0.06),
            conv_block(32, 64, dropout=0.10),
            conv_block(64, 128, dropout=0.12),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1)
        return self.classifier(x)


def build_model_a(image_size, num_classes):
    return CustomCNNA(image_size, num_classes)
