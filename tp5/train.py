
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import train_loader, test_loader
from text_utils import pretrained_embeddings, idx2word, word2idx, tokenize, untokenize
from model import ImageCaptioningModel, resnet_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)

model = ImageCaptioningModel(
    resnet_pretrained,
    pretrained_embeddings,
    lstm_hidden_dim=512
).to(device)

# Gel des poids pré-entraînés
for param in model.cnn.parameters():
    param.requires_grad = False

for param in model.embeddings.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

criterion = nn.CrossEntropyLoss(ignore_index=0)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1
)

writer = SummaryWriter("runs/captioning")

num_epochs = 15
global_step = 0

for epoch in range(num_epochs):

    model.train()
    model.cnn.eval()

    total_loss = 0.0
    running_loss = 0.0

    for i, (images, captions) in enumerate(train_loader):

        images = images.to(device)

        caps = torch.LongTensor([
            tokenize(x, word2idx) for x in captions[0]
        ]).to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images, caps[:, :-1])

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            caps[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        global_step += 1

        if (i + 1) % 10 == 0:
            writer.add_scalar(
                "Loss/Train_Step",
                running_loss / 10,
                global_step
            )
            running_loss = 0.0

    train_loss = total_loss / len(train_loader)

    writer.add_scalar(
        "Loss/Train_Epoch", train_loss, epoch + 1
    )

    # Evaluation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for images, captions in test_loader:

            images = images.to(device)

            caps = torch.LongTensor([
                tokenize(x, word2idx) for x in captions[0]
            ]).to(device)

            outputs = model(images, caps[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1)
            )

            val_loss += loss.item()

        predictions = outputs[-1].argmax(dim=-1)

        pred_text = " ".join(
            untokenize(predictions.cpu().tolist(), idx2word)
        )

        true_text = " ".join(
            untokenize(caps[-1][1:].cpu().tolist(), idx2word)
        )

    val_loss /= len(test_loader)

    writer.add_scalar(
        "Loss/Validation", val_loss, epoch + 1
    )

    current_lr = optimizer.param_groups[0]["lr"]

    writer.add_scalar("LR", current_lr, epoch + 1)

    writer.add_text(
        "Captions/Prediction",
        pred_text,
        epoch + 1
    )

    writer.add_text(
        "Captions/Target",
        true_text,
        epoch + 1
    )

    print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
    print(f"Train Loss : {train_loss:.4f}", flush=True)
    print(f"Val Loss   : {val_loss:.4f}", flush=True)
    print(f"LR         : {current_lr:.6f}", flush=True)
    print(f"Prediction : {pred_text}", flush=True)
    print(f"Target     : {true_text}", flush=True)

    scheduler.step()
    writer.flush()

writer.close()
print("\nEntraînement terminé !", flush=True)
