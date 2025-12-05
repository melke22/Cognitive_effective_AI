import os
import sys
import cv2
import pandas as pd

# Make src importable when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing.ravdess_loader import load_ravdess


def extract_middle_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"No frames in video: {video_path}")
        cap.release()
        return False

    middle_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read middle frame: {video_path}")
        return False

    # Optional: resize to something like 224x224 here if you want
    # frame = cv2.resize(frame, (224, 224))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return True


def main():
    # Load all metadata (audio+video) and filter videos
    df = load_ravdess(os.path.join(ROOT_DIR, "data"))
    df_video = df[df["modality"] == "video"].reset_index(drop=True)

    print("Total video samples:", len(df_video))

    output_root = os.path.join(ROOT_DIR, "data", "video_frames")
    records = []

    for idx, row in df_video.iterrows():
        video_path = row["file_path"]
        emotion = row["emotion"]

        # Create file name
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_root, emotion)
        out_path = os.path.join(out_dir, f"{base_name}_frame.png")

        success = extract_middle_frame(video_path, out_path)
        if success:
            records.append({"image_path": out_path, "emotion": emotion})

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(df_video)} videos")

    # Save a CSV with metadata
    if records:
        df_frames = pd.DataFrame(records)
        csv_path = os.path.join(output_root, "frames_metadata.csv")
        df_frames.to_csv(csv_path, index=False)
        print(f"Saved frame metadata to: {csv_path}")
        print("Total extracted frames:", len(df_frames))
    else:
        print("No frames were extracted!")


if __name__ == "__main__":
    main()
