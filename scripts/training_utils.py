import torch
import torchvision.transforms as T
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def train_classifier(model, device, train_loader, val_loader, optimizer, scheduler, criterion, epochs,
                     early_stopping=10):
    model = model.to(device)

    best_val_loss = float('inf')
    counter = 0
    best_model = None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, dim=1)
            correct += (predicted == target).sum()
            total += target.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = (correct / total).item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, dim=1)
                val_correct += (predicted == target).sum()
                val_total += target.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = (val_correct / val_total).item()

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{epoch + 1}/{epochs}]. "
              f"Train Loss: {train_loss:.4f}. Train accuracy: {train_acc:.4f}. "
              f"Val Loss: {val_loss:.4f}. Val accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            counter = 0
            print(
                f"Val loss decreased from {best_val_loss:.4f} to {val_loss:.4f}. Saving new model for val acc: {val_acc}.")

            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        else:
            counter += 1
            if counter >= early_stopping:
                print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                break
        if scheduler:
            scheduler.step(val_loss)

    model.load_state_dict(best_model)
    return model, history


def evaluate_classifier(model, device, test_loader):
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    topk_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, preds = torch.max(probabilities, dim=1)

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted"
    )
    top_k_accuracy = topk_correct / (len(test_loader.dataset))

    print(
        f"Top-1: {accuracy:.4f}. Top-5: {top_k_accuracy:.4f}. Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return accuracy, precision, recall, f1


def train_regressor(model, device, train_loader, val_loader, optimizer, scheduler, criterion, epochs,
                    early_stopping=10, tol=1e-3, verbose=True):
    model = model.to(device)

    best_val_loss = float('inf')
    counter = 0
    best_model = None

    history = {
        'train_loss': [],
        'val_loss': [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()
            optimizer.zero_grad()
            output = model(data)

            if output.dim() > 1 and output.shape[1] == 1:
                output = output.squeeze(1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).float()
                output = model(data)

                if output.dim() > 1 and output.shape[1] == 1:
                    output = output.squeeze(1)

                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)

        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0:
            if verbose:
                print(f"[{epoch + 1}/{epochs}]. "
                      f"Train Loss: {train_loss:.4f}. "
                      f"Val Loss: {val_loss:.4f}.")

        if best_val_loss - val_loss > tol:
            counter = 0
            if verbose: print(f"Val loss decreased from {best_val_loss:.4f} to {val_loss:.4f}. Saving new model.")
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        else:
            counter += 1
            if counter >= early_stopping:
                if verbose: print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                break

        if scheduler:
            scheduler.step(val_loss)

    model.load_state_dict(best_model)
    return model, history


def evaluate_regressor(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)

            if output.dim() > 1 and output.shape[1] == 1:
                output = output.squeeze(1)

            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(output.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    mse = mean_squared_error(all_targets, all_preds)
    rmse = root_mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f"Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return all_targets, all_preds


_default_augmentations = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=None,
        shear=None,
        fill=0
    )
])


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augmentations=_default_augmentations, scattering=False, classification=False):
        self.X = torch.tensor(X, dtype=torch.float32)

        if classification:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

        self.augmentations = augmentations

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.augmentations:
            image = self.augmentations(image)

        return image, label
