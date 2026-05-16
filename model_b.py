import torch
import torch.nn as nn


def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1):
    padding = kernel_size // 2
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    ]
    return nn.Sequential(*layers)


class PlainCNNStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stage = nn.Sequential(
            conv_bn_act(in_channels, out_channels),
            conv_bn_act(out_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.stage(x)


class CustomCNNB(nn.Module):
    """Deeper plain CNN without residual connections, attention, or transformers."""

    def __init__(self, image_size, num_classes):
        super().__init__()
        del image_size

        self.features = nn.Sequential(
            conv_bn_act(3, 32),
            PlainCNNStage(32, 48),
            PlainCNNStage(48, 72),
            PlainCNNStage(72, 96),
            PlainCNNStage(96, 128),
            conv_bn_act(128, 160),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1)
        return self.classifier(x)


def build_model_b(image_size, num_classes):
    return CustomCNNB(image_size, num_classes)
