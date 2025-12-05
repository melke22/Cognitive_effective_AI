import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.video_dataset import RavdessFramesDataset
from models.video_resnet import VideoEmotionResNet
import pandas as pd


def main():
    csv_path = os.path.join(CURRENT_DIR, "data", "video_frames", "frames_metadata.csv")
    df = pd.read_csv(csv_path)

    # Train/val split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["emotion"]
    )

    # Save temporary CSVs
    train_csv = os.path.join(CURRENT_DIR, "data", "video_frames", "train_frames.csv")
    val_csv = os.path.join(CURRENT_DIR, "data", "video_frames", "val_frames.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_dataset = RavdessFramesDataset(train_csv)
    val_dataset = RavdessFramesDataset(val_csv)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    n_classes = len(train_dataset.emotions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = VideoEmotionResNet(n_classes=n_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save model
    save_path = os.path.join(CURRENT_DIR, "results", "video_emotion_resnet.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", save_path)


if __name__ == "__main__":
    main()