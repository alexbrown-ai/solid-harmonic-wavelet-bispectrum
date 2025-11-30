from torch import nn
import torch

class ScatteringCNN(nn.Module):
    def __init__(self, input_shape, scattering, feature_dropout=0.1, classifier_dropout=0.1):
        super(ScatteringCNN, self).__init__()

        self.scattering = scattering

        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])
        dummy_output = scattering(dummy_input)
        self.output_dim = dummy_output.shape[-1]
        dummy_output = dummy_output.view(dummy_output.size(0), -1, self.output_dim, self.output_dim)
        self.K = dummy_output.shape[1]
        print(self.K)

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(self.K),

            nn.Conv2d(in_channels=self.K, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(feature_dropout),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(feature_dropout),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, groups=128),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(feature_dropout),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0, groups=256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(feature_dropout),
        )

        with torch.no_grad():
            dummy_features = self.feature_extractor(dummy_output)
            flattened_dim = dummy_features.view(1, -1).shape[1]

        print(flattened_dim)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.model = nn.Sequential(self.feature_extractor, self.classifier)
        self.apply(self._init_weights)

        all_params = sum(p.numel() for p in self.parameters())
        print(f"Instantiated model with {all_params} params.")

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1, 1) and m.in_channels == self.K:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            else:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 0.8)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.scattering(x)
        x = x.view(x.size(0), self.K, self.output_dim, self.output_dim)
        return self.model(x)