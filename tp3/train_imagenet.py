import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pipeline d'entraînement
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Pipeline de validation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...")

    train_dataset = ImageNet(
        root=data_dir,
        split="train",
        transform=train_transform
    )

    val_dataset = ImageNet(
        root=data_dir,
        split="val",
        transform=val_transform
    )

    # Sous-ensemble pour les tests rapides
    if subset_size is not None:
        generator = torch.Generator().manual_seed(42)

        train_indices = torch.randperm(
            len(train_dataset), generator=generator
        )[:subset_size]

        val_indices = torch.randperm(
            len(val_dataset), generator=generator
        )[:subset_size // 4]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True
    )

    return train_loader, val_loader

class MLP(nn.Module):
    def __init__(self, input_size=3 * 227 * 227,
                 hidden_size=256, num_classes=1000):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



class AlexNetDeep(AlexNet):
    def __init__(self, num_classes=1000):
        super().__init__(num_classes=num_classes)

        # Ajouter 3 convolutions après Conv5 et avant MaxPool3
        layers = list(self.features.children())

        extra_layers = [
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        self.features = nn.Sequential(
            *layers[:-1],
            *extra_layers,
            layers[-1]
        )




@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_accuracy = correct / total

    return val_loss, val_accuracy



def train_model(model_name="alexnet", batch_size=256, lr=1e-4,
                num_epochs=20, weight_decay=1e-5,
                patience=5, subset_size=20000):

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        subset_size=subset_size
    )


    if model_name == "alexnet":
        model = AlexNet(num_classes=1000).to(device)
    elif model_name == "alexnet_deep":
        model = AlexNetDeep(num_classes=1000).to(device)
    elif model_name == "mlp":
        model = MLP(num_classes=1000).to(device)
    else:
        raise ValueError("Modèle inconnu")


    print(
        f"Modèle {model_name} initialisé avec "
        f"{count_parameters(model):,} paramètres."
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    run_tag = f"{model_name}_sub{subset_size}_lr{lr}"
    writer = SummaryWriter(log_dir=f"runs/{run_tag}")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{run_tag}_best.pth"

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        val_loss, val_acc = evaluate(
            val_loader, model, criterion
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        epoch_time = time.time() - start_time
        writer.add_scalar("Time/Epoch", epoch_time, epoch)

        writer.flush()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} "
            f"(Acc: {train_acc:.4f}) | "
            f"Val Loss: {val_loss:.4f} "
            f"(Acc: {val_acc:.4f}) | "
            f"Time: {epoch_time:.2f}s"
        )

        # --- EARLY STOPPING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping déclenché !")
            break

    writer.close()

    # Restaurer les meilleurs poids
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )

    return model





if __name__ == "__main__":
    print("Device:", device)

    print("\n========== Training AlexNet Full ImageNet ==========")

    alexnet = train_model(
        model_name="alexnet",
        batch_size=256,
        lr=1e-4,
        num_epochs=50,
        weight_decay=1e-5,
        patience=5,
        subset_size=None
    )

