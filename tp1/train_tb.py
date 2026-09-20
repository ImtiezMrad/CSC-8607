import os
import random
import datetime

import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ==================================================
# HYPERPARAMETERS
# ==================================================

hparams = dict(
    model="MLP",
    batch_size=128,
    lr=1e-1,
    seed=0,
    weight_decay=0.0
)


# ==================================================
# TENSORBOARD
# ==================================================

run_name = (
    f"{hparams['model']}/"
    f"bs{hparams['batch_size']}_"
    f"lr{hparams['lr']}_"
    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

logdir = os.path.join("runs", run_name)

print("Logdir:", logdir)

writer = SummaryWriter(log_dir=logdir)


# ==================================================
# CIFAR-10
# ==================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

trainset = datasets.CIFAR10(
    root="./data2",
    train=True,
    download=False,
    transform=transform
)

testset = datasets.CIFAR10(
    root="./data2",
    train=False,
    download=False,
    transform=transform
)


# ==================================================
# DATALOADER CONFIGURATION
# ==================================================

def get_num_workers(default=2, cap=4):
    try:
        n = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    except Exception:
        n = default

    return max(0, min(cap, n))


num_workers = get_num_workers()


# ==================================================
# TRAIN / VALIDATION SPLIT
# ==================================================

N = len(trainset)

val_size = int(0.1 * N)
train_size = N - val_size

train_subset, val_subset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(hparams["seed"])
)


# ==================================================
# DATALOADERS
# ==================================================

trainloader = DataLoader(
    train_subset,
    batch_size=hparams["batch_size"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

valloader = DataLoader(
    val_subset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# ==================================================
# VALIDATION METRICS
# ==================================================

@torch.no_grad()
def epoch_metrics(loader, model, criterion, device):

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)

        pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    loss_avg = loss_sum / total
    acc = correct / total

    return loss_avg, acc


# ==================================================
# MODEL
# ==================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


# ==================================================
# SEED + DEVICE
# ==================================================

torch.manual_seed(hparams["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hparams["seed"])

random.seed(hparams["seed"])

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# ==================================================
# MODEL / LOSS / OPTIMIZER
# ==================================================

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hparams["lr"],
    momentum=0.9,
    weight_decay=hparams["weight_decay"]
)


# ==================================================
# TRAINING
# ==================================================

EPOCHS = 10
global_step = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss_sum = 0.0
    running_total = 0

    for b, (inputs, labels) in enumerate(trainloader):

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # TensorBoard : loss batch
        if b % 10 == 0:
            writer.add_scalar(
                "Loss/train_step",
                loss.item(),
                global_step
            )

        running_loss_sum += loss.item() * inputs.size(0)
        running_total += labels.size(0)

        global_step += 1

    # Loss moyenne train
    train_loss = running_loss_sum / running_total

    # Validation
    val_loss, val_acc = epoch_metrics(
        valloader,
        model,
        criterion,
        device
    )

    # TensorBoard
    writer.add_scalar(
        "Loss/train",
        train_loss,
        epoch
    )

    writer.add_scalar(
        "Loss/val",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "Accuracy/val",
        val_acc,
        epoch
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )


# ==================================================
# CLOSE TENSORBOARD WRITER
# ==================================================

writer.close()