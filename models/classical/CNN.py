"""PyTorch reimplementation of https://arxiv.org/pdf/2401"""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=32,
                kernel_size=6, stride=1, padding=0,
            ),

            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),

            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=4, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.1),

            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),

            nn.Flatten(),

            nn.Linear(12800, 64),
            nn.ReLU(),

            nn.Linear(
                64, 2
            ),
        )

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.model.apply(init_weights)
        all_params = sum(p.numel() for p in self.parameters())
        print(f"Instantiated model with {all_params} params.")

    def forward(self, x):
        return self.model(x)
