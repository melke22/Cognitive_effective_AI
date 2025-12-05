import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RavdessFramesDataset(Dataset):
    """
    PyTorch Dataset for single frames extracted from RAVDESS videos.
    Uses frames_metadata.csv.
    """
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.emotions = sorted(self.df["emotion"].unique())
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx_to_emotion = {i: e for e, i in self.emotion_to_idx.items()}

        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        emotion = row["emotion"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.emotion_to_idx[emotion], dtype=torch.long)
        return img, label