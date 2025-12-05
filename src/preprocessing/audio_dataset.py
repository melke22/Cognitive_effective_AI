import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

class RavdessAudioDataset(Dataset):
    """
    PyTorch Dataset for RAVDESS audio files.
    It:
      - filters only audio samples
      - extracts MFCC features
      - maps emotion labels to integer IDs
    """
    def __init__(self, df: pd.DataFrame, max_len: int = 300, n_mfcc: int = 40):
        # Keep only audio rows
        self.df = df[df["modality"] == "audio"].reset_index(drop=True)

        # Unique emotions & mapping
        self.emotions = sorted(self.df["emotion"].unique())
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        self.idx_to_emotion = {i: e for e, i in self.emotion_to_idx.items()}

        self.max_len = max_len
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.df)

    def _load_mfcc(self, file_path: str) -> np.ndarray:
        """
        Load audio file and convert to MFCC features.
        Output shape: (n_mfcc, max_len)
        """
        y, sr = librosa.load(file_path, sr=16000)  # resample to 16kHz

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)  # (n_mfcc, T)

        # Pad or truncate to fixed length in time dimension
        if mfcc.shape[1] < self.max_len:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :self.max_len]

        return mfcc.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        emotion_label = row["emotion"]

        mfcc = self._load_mfcc(file_path)  # (n_mfcc, max_len)
        x = torch.tensor(mfcc).unsqueeze(0)  # -> (1, n_mfcc, max_len)
        y = torch.tensor(self.emotion_to_idx[emotion_label], dtype=torch.long)

        return x, y
