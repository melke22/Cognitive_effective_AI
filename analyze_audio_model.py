import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Make src importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.ravdess_loader import load_ravdess
from preprocessing.audio_dataset import RavdessAudioDataset
from models.audio_cnn import AudioEmotionCNN

def main():
    # Load metadata & filter audio
    df = load_ravdess(os.path.join(CURRENT_DIR, "data"))
    df_audio = df[df["modality"] == "audio"].reset_index(drop=True)

    # Use the SAME split strategy as training
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df_audio,
        test_size=0.2,
        random_state=42,
        stratify=df_audio["emotion"]
    )

    # Validation dataset & loader
    val_dataset = RavdessAudioDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    n_classes = len(val_dataset.emotions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Rebuild model & load weights
    model = AudioEmotionCNN(n_classes=n_classes).to(device)
    model_path = os.path.join(CURRENT_DIR, "results", "audio_emotion_cnn.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    emotions = [val_dataset.idx_to_emotion[i] for i in range(n_classes)]

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=emotions))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Audio Emotion Confusion Matrix (Validation Set)")
    plt.tight_layout()

    # Save & show
    out_path = os.path.join(CURRENT_DIR, "results", "audio_confusion_matrix.png")
    plt.savefig(out_path)
    print(f"\nConfusion matrix image saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
