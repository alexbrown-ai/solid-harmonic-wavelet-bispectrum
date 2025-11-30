import torch
import torch.nn as nn


class ScatteringAutoencoder(nn.Module):
    """
    Autoencoder that reconstructs images from pre-computed scattering coefficients.
    """
    def __init__(self, input_dim):
        super(ScatteringAutoencoder, self).__init__()

        self.start_channels = 512
        self.start_spatial = 4
        fc_out_dim = self.start_channels * self.start_spatial * self.start_spatial

        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, fc_out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fc_out_dim),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.start_channels, 256, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 64->128
            nn.Sigmoid()
        )

        all_params = sum(p.numel() for p in self.parameters())
        print(f"Instantiated model with {all_params:,} parameters.")

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.start_channels, self.start_spatial, self.start_spatial)
        x = self.decoder(x)
        x = x.squeeze()
        return x


