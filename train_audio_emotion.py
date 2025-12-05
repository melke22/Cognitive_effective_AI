import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Fix Python path for importing modules from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# NOTE: since SRC_DIR is on sys.path, we import without "src."
from preprocessing.ravdess_loader import load_ravdess
from preprocessing.audio_dataset import RavdessAudioDataset
from models.audio_cnn import AudioEmotionCNN


def main():
    # 1. Load metadata from RAVDESS
    df = load_ravdess(os.path.join(CURRENT_DIR, "data"))

    # Filter for audio only
    df_audio = df[df["modality"] == "audio"].reset_index(drop=True)

    print("Total audio samples:", len(df_audio))
    print(df_audio.head())

    # TEMP: use only a subset for quick testing
    # df_audio = df_audio.sample(n=200, random_state=42).reset_index(drop=True)
    # print("Using subset of samples for quick test:", len(df_audio))

    # 2. Train/validation split
    train_df, val_df = train_test_split(
        df_audio,
        test_size=0.2,
        random_state=42,
        stratify=df_audio["emotion"]
    )

    # 3. Datasets & Dataloaders
    train_dataset = RavdessAudioDataset(train_df)
    val_dataset = RavdessAudioDataset(val_df)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model, Loss, Optimizer
    n_classes = len(train_dataset.emotions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = AudioEmotionCNN(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop
    num_epochs = 20 # Change to desired number of epochs
    #num_epochs = 5  # TEMP: for quick testing

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch} - Batch {batch_idx + 1}/{len(train_loader)} "
                    f"- Loss: {loss.item():.4f}"
                )

        # End of epoch: average loss
        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(batch_y.cpu().numpy().tolist())

        val_acc = accuracy_score(val_targets, val_preds)

        print(
            f"Epoch [{epoch}/{num_epochs}] - "
            f"Train Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

    # 6. Save model
    save_path = os.path.join(CURRENT_DIR, "results", "audio_emotion_cnn.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)


if __name__ == "__main__":
    main()
