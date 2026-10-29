
import os
import json
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Reproductibilité
torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemin vers les données
DATA_DIR = "/mnt/ssd/irishman"

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Chargement des données
train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Device : {device}")
print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")

print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])


# ============================================================
# Exercice 1 - Étape 2 : Vocabulaire et Padding
# ============================================================

# 1. Extraction des caractères uniques
all_text = "".join(song["abc notation"] for song in train_data + val_data)

unique_chars = sorted(list(set(all_text)))
vocab_size = len(unique_chars)

print(f"Taille du vocabulaire : {vocab_size} caractères uniques")

# 2. Correspondance caractère <-> indice
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

def vectorize_string(s):
    return [char_to_index[char] for char in s]

# 3. Statistiques de longueur
lengths = [len(song["abc notation"]) for song in train_data]

max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")

# 4. Padding et troncature
SEQ_LENGTH = 500

def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence.ljust(max_length, " ")
        


# ============================================================
# Exercice 2 - Étape 1 : MusicDataset auto-régressif
# ============================================================

class MusicDataset(Dataset):
    def __init__(self, raw_data, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        self.data = [
            vectorize_string(
                pad_or_truncate(song["abc notation"], seq_length)
            )
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]

        # Entrée : tous les caractères sauf le dernier
        x = torch.tensor(seq[:-1], dtype=torch.long)

        # Cible : tous les caractères sauf le premier
        y = torch.tensor(seq[1:], dtype=torch.long)

        return x, y


# ============================================================
# Vérification du Dataset
# ============================================================

BATCH_SIZE = 128

# Sous-ensemble de 10 000 partitions pour les expériences
train_dataset = MusicDataset(train_data, subset_size=10000)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

x, y = next(iter(train_loader))

print("\n--- Vérification MusicDataset ---")
print(f"Nombre de partitions : {len(train_dataset)}")
print(f"Shape x : {x.shape}")
print(f"Shape y : {y.shape}")
print(f"Type x : {x.dtype}")
print(f"Type y : {y.dtype}")

print("\n--- Vérification du décalage ---")
print("5 premiers caractères de x :", x[0, :5].tolist())
print("5 premiers caractères de y :", y[0, :5].tolist())
print("Décalage correct :", torch.equal(x[0, 1:], y[0, :-1]))





# ============================================================
# Exercice 2 - Étape 2 : Architecture RNN vs LSTM
# ============================================================

class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256,
                 hidden_size=512, rnn_type="lstm"):
        super().__init__()

        self.rnn_type = rnn_type.lower()

        # Couche d'Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Couche récurrente
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                batch_first=True
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_size,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")

        # Couche de sortie
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x : (batch_size, seq_len)

        embeds = self.embedding(x)

        out, hidden_state = self.rnn(embeds, hidden_state)

        logits = self.fc(out)

        return logits, hidden_state


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Vérification des modèles
# ============================================================

rnn_model = MusicModel(
    vocab_size,
    embedding_dim=256,
    hidden_size=512,
    rnn_type="rnn"
).to(device)

lstm_model = MusicModel(
    vocab_size,
    embedding_dim=256,
    hidden_size=512,
    rnn_type="lstm"
).to(device)

print("\n--- Comparaison RNN vs LSTM ---")

for name, model in [("RNN", rnn_model), ("LSTM", lstm_model)]:
    print(f"\nModèle : {name}")
    print(f"Embedding : {count_parameters(model.embedding)} paramètres")
    print(f"Récurrent : {count_parameters(model.rnn)} paramètres")
    print(f"Linear : {count_parameters(model.fc)} paramètres")
    print(f"Total : {count_parameters(model)} paramètres")

# Test du passage avant (forward)
x_test, y_test = next(iter(train_loader))
x_test = x_test.to(device)

with torch.no_grad():
    logits_rnn, hidden_rnn = rnn_model(x_test)
    logits_lstm, hidden_lstm = lstm_model(x_test)

print("\n--- Vérification des dimensions ---")
print(f"Logits RNN : {logits_rnn.shape}")
print(f"Hidden RNN : {hidden_rnn.shape}")
print(f"Logits LSTM : {logits_lstm.shape}")
print(f"Hidden LSTM (h) : {hidden_lstm[0].shape}")
print(f"Cell LSTM (c) : {hidden_lstm[1].shape}")




# ============================================================
# Exercice 3 - Évaluation et entraînement
# ============================================================

@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, vocab_size):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, _ = model(x)

        # CrossEntropyLoss attend (N, C) et (N,)
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        total_loss += loss.item() * y.numel()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

    return total_loss / total_tokens, total_correct / total_tokens


def train_model(
    rnn_type="lstm",
    subset_size=10000,
    num_epochs=15,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=512,
    max_grad_norm=None,
    patience=5,
):
    train_dataset = MusicDataset(
        train_data, subset_size=subset_size
    )
    val_dataset = MusicDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = MusicModel(
        vocab_size,
        embedding_dim,
        hidden_size,
        rnn_type=rnn_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    clip_tag = (
        f"clip{max_grad_norm}"
        if max_grad_norm is not None
        else "noclip"
    )

    run_name = (
        f"{rnn_type}_sub{subset_size}_lr{lr}_"
        f"{clip_tag}_{time.strftime('%H%M%S')}"
    )

    os.makedirs("checkpoints", exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs_rnn/{run_name}")
    save_path = f"checkpoints/best_{run_name}.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0
        epoch_grad_norms = []

        for x, y in tqdm(
            train_loader,
            desc=f"[{run_name}] Epoch {epoch}/{num_epochs}",
            leave=False
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward
            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            # Backpropagation
            loss.backward()

            # Norme du gradient et clipping éventuel
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float("inf")
                )

            epoch_grad_norms.append(grad_norm.item())

            if global_step % 20 == 0:
                writer.add_scalar(
                    "GradNorm/step",
                    grad_norm.item(),
                    global_step
                )

            # Mise à jour des poids
            optimizer.step()

            total_train_loss += loss.item() * y.numel()
            total_train_tokens += y.numel()
            global_step += 1

        avg_train_loss = total_train_loss / total_train_tokens
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)

        val_loss, val_acc = evaluate_model(
            model, val_loader, loss_fn, vocab_size
        )

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("GradNorm/Epoch_Avg", avg_grad_norm, epoch)
        writer.flush()

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Avg GradNorm: {avg_grad_norm:.3f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            print(f"Meilleur modèle sauvegardé : {save_path}")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping à l'époque {epoch}.")
                break

    writer.close()
    return model



# ============================================================
# Exercice 4 - Étape 1 : Génération auto-régressive
# ============================================================

@torch.no_grad()
def generate_music(
    model,
    start_sequence,
    output_length=300,
    temperature=1.0
):
    model.eval()

    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    generated = start_sequence

    # Transformer l'amorce en indices
    indices = vectorize_string(start_sequence)

    input_tensor = torch.tensor(
        [indices],
        dtype=torch.long,
        device=device
    )

    # Initialiser le LSTM avec l'amorce complète
    logits, hidden_state = model(input_tensor)

    # Génération d'un caractère à la fois
    for _ in range(output_length):

        # Appliquer la température
        next_logits = logits[:, -1, :] / temperature

        # Calculer les probabilités
        probabilities = F.softmax(next_logits, dim=-1)

        # Tirer le prochain caractère
        next_index = torch.multinomial(
            probabilities, num_samples=1
        ).item()

        next_char = index_to_char[next_index]
        generated += next_char

        # Réinjecter uniquement le nouveau caractère
        input_tensor = torch.tensor(
            [[next_index]],
            dtype=torch.long,
            device=device
        )

        # Conserver l'état caché du LSTM
        logits, hidden_state = model(
            input_tensor,
            hidden_state
        )

    return generated




# ============================================================
# Exercise 4 - Long LSTM training
# ============================================================

if __name__ == "__main__":

    print("\n========== EXERCISE 4: LONG LSTM TRAINING ==========")

    train_model(
        rnn_type="lstm",
        subset_size=None,      # All 214122 training songs
        num_epochs=30,       # Maximum; early stopping enabled
        batch_size=256,
        lr=5e-3,
        embedding_dim=256,
        hidden_size=1024,
        max_grad_norm=1.0,
        patience=5
    )
