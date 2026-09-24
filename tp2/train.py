import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from dataset import CardioDataset


# ============================================================
# Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


# ============================================================
# Dataset
# ============================================================

dataset = CardioDataset("data/cardio_train.csv")

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [0.8, 0.1, 0.1],
    generator=generator
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


# ============================================================
# Modèle MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Entraînement
# ============================================================

def train_model(optimizer_name, learning_rate=0.001, epochs=30):

    input_size = dataset.features.shape[1]

    model = MLP(
        input_size=input_size,
        hidden_size=128
    ).to(device)

    criterion = nn.BCELoss()

    # --------------------------------------------------------
    # Choix de l'optimiseur
    # --------------------------------------------------------

    if optimizer_name == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    elif optimizer_name == "Momentum":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    elif optimizer_name == "RMSprop":

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif optimizer_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:

        raise ValueError(
            f"Optimiseur inconnu : {optimizer_name}"
        )

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    writer = SummaryWriter(
        f"runs/cardio_{optimizer_name}_lr{learning_rate}"
    )

    # --------------------------------------------------------
    # Boucle d'entraînement
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        writer.add_scalar(
            "Training Loss",
            epoch_loss,
            epoch
        )

        print(
            f"[{optimizer_name}] "
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f}"
        )

    writer.close()

    return model


# ============================================================
# Évaluation sur le jeu de test
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_targets = []
    all_preds_probs = []

    with torch.no_grad():

        for batch in test_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(inputs)

            all_targets.extend(
                targets.cpu().numpy()
            )

            all_preds_probs.extend(
                outputs.cpu().numpy()
            )

    # Conversion en tableaux numpy

    all_targets = np.array(
        all_targets
    ).flatten()

    all_preds_probs = np.array(
        all_preds_probs
    ).flatten()

    # --------------------------------------------------------
    # Prédictions binaires avec seuil 0.5
    # --------------------------------------------------------

    all_preds_classes = (
        all_preds_probs > 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Calcul des métriques
    # --------------------------------------------------------

    precision = precision_score(
        all_targets,
        all_preds_classes
    )

    recall = recall_score(
        all_targets,
        all_preds_classes
    )

    f1 = f1_score(
        all_targets,
        all_preds_classes
    )

    auc = roc_auc_score(
        all_targets,
        all_preds_probs
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print("\n===== Test Metrics =====")

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"AUC:       {auc:.4f}"
    )

    print("========================\n")


# ============================================================
# Exercice 3 : comparaison des optimiseurs
# ============================================================

# Décommenter cette partie si besoin pour refaire
# les quatre expériences TensorBoard.

# for optimizer in ["SGD", "Momentum", "RMSprop", "Adam"]:
#     train_model(
#         optimizer,
#         learning_rate=0.001,
#         epochs=30
#     )


# ============================================================
# Exercice 4 : évaluation du meilleur modèle
# ============================================================

best_model = train_model(
    "Adam",
    learning_rate=0.001,
    epochs=30
)

evaluate_model(
    best_model,
    test_loader
)