"""Fine-tune Zoobot as documented at https://zoobot.readthedocs.io"""
import lightning as L
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from zoobot.pytorch.training import finetune

from models.zoobot.Zoobot import zoobot_classifier, zoobot_regressor


class ImageDataset(Dataset):
    """Images (N, C, H, W) with C = 1 or 3, as 3-channel 224x224 zoobot batch dicts."""

    def __init__(self, X, y, augmentations=None, size=224):
        self.X, self.y, self.augmentations = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y), augmentations
        self.resize = T.Resize((size, size), antialias=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i] if self.augmentations is None else self.augmentations(self.X[i])
        return {"image": self.resize(x.expand(3, -1, -1)), "label": self.y[i]}


def finetune_zoobot(model, train, val, test, save_dir, batch_size=32, max_epochs=100, patience=10):
    """Fits on `train`, early-stops on `val`, and returns the best checkpoint's predictions on `test`."""
    dl = lambda ds, shuffle=False: DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    trainer = finetune.get_trainer(save_dir, max_epochs=max_epochs, patience=patience,
                                   logger=L.pytorch.loggers.CSVLogger(save_dir), enable_progress_bar=False)
    trainer.fit(model, dl(train, True), dl(val))
    return torch.cat(trainer.predict(model, dl(test), ckpt_path="best")).numpy()


def train_zoobot_classifier(X_train, y_train, X_val, y_val, X_test, save_dir, augmentations=None, **kwargs):
    """Returns test class probabilities (n, 2). `augmentations`: optional callable for training images."""
    make = lambda X, y, aug=None: ImageDataset(X, np.asarray(y, dtype=np.int64), aug)
    return finetune_zoobot(zoobot_classifier(), make(X_train, y_train, augmentations), make(X_val, y_val),
                           make(X_test, np.zeros(len(X_test))), save_dir, **kwargs)


def train_zoobot_regressor(X_train, y_train, X_val, y_val, X_test, save_dir, augmentations=None, **kwargs):
    """Returns test predictions (n,). `augmentations`: optional callable for training images."""
    make = lambda X, y, aug=None: ImageDataset(X, np.asarray(y, dtype=np.float32), aug)
    return finetune_zoobot(zoobot_regressor(), make(X_train, y_train, augmentations), make(X_val, y_val),
                           make(X_test, np.zeros(len(X_test))), save_dir, **kwargs)
