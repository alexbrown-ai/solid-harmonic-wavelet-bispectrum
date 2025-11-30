from torch import nn
import torch


class MultiLinear(nn.Module):
    def __init__(self, input_shape, scattering, order=2, num_features=1, num_classes=2, dropout=0.5):
        super(MultiLinear, self).__init__()

        dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1])
        dummy_output = scattering.scatter(dummy_input)
        self.scattering_dim = dummy_output.shape[-1]

        self.scattering = scattering

        self.order = order
        self.num_features = num_features

        self.bn = nn.BatchNorm1d(self.scattering_dim)

        self.w = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.scattering_dim, num_features),
                nn.BatchNorm1d(num_features),
                nn.Dropout(dropout)
            )
            for _ in range(order)
        ])

        self.v = nn.Linear(num_features, num_classes)

        all_params = sum(p.numel() for p in self.parameters())
        print(f"Instantiated model with {all_params} params.")

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.scattering.scatter(x)
        x = self.bn(x)
        product = None
        for z in range(self.order):
            linear_result = self.w[z](x).squeeze(-1)
            product = linear_result if product is None else product * linear_result

        return self.v(product).squeeze(-1)


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
