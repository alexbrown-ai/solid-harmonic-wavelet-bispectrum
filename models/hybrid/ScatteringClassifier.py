from torch import nn
import torch


class ScatteringClassifier(nn.Module):
    def __init__(self, input_shape, scattering, reduced_units, dropout, output_type="regression"):
        super(ScatteringClassifier, self).__init__()

        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])
        dummy_output = scattering.scatter(dummy_input)
        self.scattering_dim = dummy_output.shape[-1]

        self.scattering = scattering

        self.model = nn.Sequential(
            nn.BatchNorm1d(self.scattering_dim),
            nn.Linear(self.scattering_dim, reduced_units),
            nn.BatchNorm1d(reduced_units),
            nn.Softmax(dim=1),
            nn.Dropout(p=dropout),
            nn.Linear(reduced_units, 1 if output_type == "regression" else 2),
        )

        all_params = sum(p.numel() for p in self.parameters())
        print(f"Instantiated model with {all_params} params.")

    def forward(self, x):
        x = self.scattering.scatter(x)
        return self.model(x)
