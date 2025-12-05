import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.video_dataset import RavdessFramesDataset
from models.video_resnet import VideoEmotionResNet


def main():
    # Use the same val CSV created during training
    val_csv = os.path.join(CURRENT_DIR, "data", "video_frames", "val_frames.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}. "
                                "Run train_video_emotion.py first to create it.")

    val_dataset = RavdessFramesDataset(val_csv)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    n_classes = len(val_dataset.emotions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Rebuild model & load weights
    model = VideoEmotionResNet(n_classes=n_classes, pretrained=False).to(device)
    model_path = os.path.join(CURRENT_DIR, "results", "video_emotion_resnet.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    emotions = [val_dataset.idx_to_emotion[i] for i in range(n_classes)]

    print("\nClassification Report (Video / Facial Emotion):")
    print(classification_report(all_targets, all_preds, target_names=emotions))

    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Video Emotion Confusion Matrix (Validation Set)")
    plt.tight_layout()

    out_path = os.path.join(CURRENT_DIR, "results", "video_confusion_matrix.png")
    plt.savefig(out_path)
    print(f"\nConfusion matrix image saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
