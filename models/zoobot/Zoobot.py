from zoobot.pytorch.training.finetune import (
    FinetuneableZoobotClassifier,
    FinetuneableZoobotRegressor,
)


class _Regressor(FinetuneableZoobotRegressor):
    def forward(self, x):
        # Zoobot's head ends in a bare .squeeze(), which turns a batch of one into a scalar
        return super().forward(x).reshape(-1)


ENCODER = "hf_hub:mwalmsley/zoobot-encoder-convnext_nano"


def zoobot_classifier(lr=1e-4, weight_decay=0.05, encoder=ENCODER, num_classes=2):
    return FinetuneableZoobotClassifier(
        name=encoder, 
        num_classes=num_classes, 
        label_col="label",
        learning_rate=lr, 
        weight_decay=weight_decay,
        prog_bar=False
    )


def zoobot_regressor(label_col="label", lr=1e-4, weight_decay=0.05, encoder=ENCODER):
    return _Regressor(
        name=encoder, 
        label_col=label_col,
        learning_rate=lr, 
        weight_decay=weight_decay, 
        prog_bar=False
    )
