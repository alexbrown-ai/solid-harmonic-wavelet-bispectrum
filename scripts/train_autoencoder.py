"""Training script for scattering-based autoencoders."""
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from models import ScatteringAutoencoder


class ScatteredFaceDataset(Dataset):
    """Dataset for pre-computed scattering coefficients and images."""
    
    def __init__(self, h5_path):
        """
        :param h5_path: Path to HDF5 file containing 'images' and 'scattering_coef' datasets
        """
        self.h5_path = h5_path
        self.hf = None

        with h5py.File(self.h5_path, 'r') as hf:
            self.length = hf['images'].shape[0]

    def __getitem__(self, index):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        img = self.hf['images'][index]
        img = torch.from_numpy(img)
        scattering_coef = self.hf['scattering_coef'][index]
        scattering_coef = torch.from_numpy(scattering_coef)

        return scattering_coef, img

    def __len__(self):
        return self.length

    def __del__(self):
        if self.hf is not None:
            self.hf.close()


def get_loss(mse_loss, l1_loss, ssim_loss):
    return 0.5 * mse_loss + 0.2 * l1_loss + 0.3 * ssim_loss


def train_and_validate(model, train_loader, val_loader, num_epochs, device, save_dir='checkpoints',
                       image_name="images", checkpoint_path=None):
    """
    Train and validate the autoencoder.
    
    :param model: Autoencoder model
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param num_epochs: Number of training epochs
    :param device: torch device
    :param save_dir: Directory to save checkpoints
    :param image_name: Name prefix for saved images
    :param checkpoint_path: Path to checkpoint to resume from
        
    :return best_val_loss, train_losses, val_losses
    """
    mse_criterion = nn.MSELoss().to(device)
    l1_criterion = nn.L1Loss().to(device)
    ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=4e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=20)

    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_l1 = 0.0
        train_ssim_loss = 0.0

        for batch_idx, (scattered, images) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')):
            scattered = scattered.to(device)
            images = images.to(device)
            optimiser.zero_grad()

            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                recon = model(scattered)
                mse_loss = mse_criterion(recon, images)
                l1_loss = l1_criterion(recon, images)
                ssim_loss = 1 - ssim(recon.unsqueeze(1), images.unsqueeze(1))
                loss = get_loss(mse_loss, l1_loss, ssim_loss)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

            train_loss += loss.item() * images.size(0)
            train_mse += mse_loss.item() * images.size(0)
            train_l1 += l1_loss.item() * images.size(0)
            train_ssim_loss += ssim_loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        train_l1 /= len(train_loader.dataset)
        train_ssim_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_l1 = 0.0
        val_ssim_loss = 0.0

        with torch.no_grad():
            for batch_idx, (scattered, images) in enumerate(
                    tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')):
                scattered = scattered.to(device)
                images = images.to(device)
                with torch.amp.autocast(device_type="cuda" if device.type == 'cuda' else "cpu"):
                    recon = model(scattered)
                    mse_loss = mse_criterion(recon, images)
                    l1_loss = l1_criterion(recon, images)
                    ssim_loss = 1 - ssim(recon.unsqueeze(1), images.unsqueeze(1))
                    loss = get_loss(mse_loss, l1_loss, ssim_loss)

                val_loss += loss.item() * images.size(0)
                val_mse += mse_loss.item() * images.size(0)
                val_l1 += l1_loss.item() * images.size(0)
                val_ssim_loss += ssim_loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_mse /= len(val_loader.dataset)
        val_l1 /= len(val_loader.dataset)
        val_ssim_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(
            f'Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, L1: {train_l1:.4f}, SSIM: {train_ssim_loss:.4f}). '
            f'Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, L1: {val_l1:.4f}, SSIM: {val_ssim_loss:.4f})')
        print(f'Learning Rate: {optimiser.param_groups[0]["lr"]:.6f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved best model with Val Loss: {best_val_loss:.4f}')
            plot_reconstructions(model, val_loader, device, num_samples=8, experiment_name=image_name, save_dir=save_dir)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
            print(f'Saved checkpoint at epoch {epoch + 1}')

    return best_val_loss, train_losses, val_losses


def get_dataloaders(dataset, batch_size=32, train_split=0.7, val_split=0.2, seed=42):
    """
    Split dataset into train/val/test loaders.

    :param dataset: Dataset instance
    :param batch_size: Batch size
    :param train_split: Fraction for training
    :param val_split: Fraction for validation
    :param seed: Random seed

    :return: train_loader, val_loader, test_loader
    """
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def plot_reconstructions(model, test_loader, device, num_samples=8, experiment_name="", save_dir="checkpoints"):
    """
    Plot and save reconstruction examples.
    
    :param model: Trained autoencoder model
    :param test_loader: Data loader for test set
    :param device: torch device
    :param num_samples: Number of samples to visualize
    :param experiment_name: Name prefix for saved file
    :param save_dir: Directory to save images
    """
    model.eval()
    images, recons = [], []
    with torch.no_grad():
        for batch_idx, (scattering, imgs) in enumerate(test_loader):
            imgs = imgs.to(device)
            scattering = scattering.to(device)
            recons_batch = model(scattering)
            images.append(imgs.cpu())
            recons.append(recons_batch.cpu())
            if (batch_idx + 1) * test_loader.batch_size >= num_samples:
                break

    images = torch.cat(images)[:num_samples]
    recons = torch.cat(recons)[:num_samples]

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        axes[1, i].imshow(recons[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')

    plt.tight_layout()

    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.png" if experiment_name else f"reconstructions_{timestamp}.png"
    plt.savefig(os.path.join(save_dir, "images", filename))
    print(f"Saved reconstruction plot to {os.path.join(save_dir, 'images', filename)}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train scattering-based autoencoder")
    parser.add_argument("--hdf5_path", type=str, required=True, help="Path to HDF5 file with scattering coefficients")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    dataset = ScatteredFaceDataset(args.hdf5_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scattering_dim = dataset[0][0].shape[-1]
    model = ScatteringAutoencoder(scattering_dim).to(device)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, args.batch_size)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    best_val_loss, train_losses, val_losses = train_and_validate(
        model, train_loader, val_loader, args.num_epochs, device, args.save_dir, 
        checkpoint_path=args.checkpoint
    )

