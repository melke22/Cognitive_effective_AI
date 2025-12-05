import os
import pandas as pd

EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_filename(filename):
    parts = filename.split("-")
    emotion = EMOTION_LABELS.get(parts[2], "unknown")
    return emotion

def load_ravdess(data_path):
    file_records = []

    for root, _, files in os.walk(data_path):
        for filename in files:
            if filename.endswith((".wav", ".mp4")):
                file_path = os.path.join(root, filename)

                modality = "audio" if filename.endswith(".wav") else "video"
                emotion = parse_filename(filename)

                file_records.append({
                    "file_path": file_path,
                    "modality": modality,
                    "emotion": emotion
                })

    df = pd.DataFrame(file_records)
    return df


if __name__ == "__main__":
    data_folder = "../../data/"  # adjust if needed
    df = load_ravdess(data_folder)
    print(df.head())
    print("Total samples loaded:", len(df))
