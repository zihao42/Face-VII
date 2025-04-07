import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio
import pandas as pd

#############################
# Global Open-Set Combination Definitions
#############################
# RAVDESS emotion labels (0-indexed) are:
# 0: Neutral, 1: Calm, 2: Happy, 3: Sad, 4: Angry, 5: Fearful, 6: Disgust, 7: Surprise
# For each combination, we define:
#   "known": set of emotion labels (0-indexed) that are considered known.
#   "unknown": set of emotion labels that are considered unknown.
# In this version, all unknown files are remapped to label 8.
COMBINATION_SPLITS = {
    1: {"known": {1, 7, 5, 4, 6}, "unknown": {0, 2, 3}},
    2: {"known": {1, 2, 3, 4, 6}, "unknown": {0, 7, 5}},
    3: {"known": {1, 7, 3, 5, 6}, "unknown": {0, 2, 4}},
    4: {"known": {0, 2, 3, 5, 4}, "unknown": {1, 7, 6}},
    5: {"known": {1, 2, 5, 6, 7}, "unknown": {0, 3, 4}},
    6: {"known": {0, 2, 3, 4, 7}, "unknown": {1, 5, 6}},
    7: {"known": {0, 1, 3, 6, 7}, "unknown": {2, 5, 4}},
    8: {"known": {0, 1, 2, 5, 4}, "unknown": {7, 3, 6}},
    9: {"known": {0, 7, 3, 4, 6}, "unknown": {1, 2, 5}},
    10: {"known": {0, 2, 3, 5, 6}, "unknown": {1, 7, 4}},
}

#############################
# Parsing Functions
#############################
def parse_ravdess_info(filename):
    """
    Parse a RAVDESS filename and return a dict with:
      - modality: first field (e.g., "02")
      - vocal_channel: second field (e.g., "01")
      - emotion: third field as a 0-indexed int (e.g., "06" becomes 5)
    Assumes filename is in the format: "XX-XX-XX-XX-XX-XX-XX.ext"
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) < 7:
        raise ValueError("Filename does not have the required 7 parts")
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': int(parts[2]) - 1  # convert to 0-indexed
    }

#############################
# Video Processing Functions
#############################
def load_video_frames(video_path, num_frames=16):
    """
    Load video frames from video_path using OpenCV.
    Uniformly sample num_frames frames and return them as RGB numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_id += 1
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames

#############################
# Audio Processing Functions
#############################
def load_audio_file(audio_path, target_sample_rate=16000):
    """
    Load an audio file using torchaudio and resample if needed.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                   new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

#############################
# File Listing Functions
#############################
def get_file_list(video_dir, extension='.mp4'):
    """
    Retrieve a list of filenames with the given extension from video_dir.
    Only include files where vocal_channel == "01" (speech).
    """
    files = []
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(extension):
            parts = f.split('-')
            if len(parts) < 2:
                continue
            if parts[1] == '01':
                files.append(f)
    random.shuffle(files)
    return files

#############################
# CSV Generation & Open-set Splitting
#############################
def generate_openset_csv(video_dir, combination, output_csv):
    """
    Generate a CSV file for the open-set split.
    The CSV will have three columns:
      - filename (e.g., "02-01-06-01-02-01-12.mp4")
      - category ("train" or "test")
      - emo_label (for known files, the original 0-indexed label; for unknown files, fixed to 8)
    
    For known files, we randomly shuffle and split them so that:
      - 6/7 of known files are used for training (marked as "train")
      - 1/7 of known files are reserved for final testing (marked as "test")
    All unknown files are marked as "test" with label 8.
    
    Returns two lists: train_files and test_files.
    """
    combo = COMBINATION_SPLITS.get(combination)
    if combo is None:
        raise ValueError(f"Invalid combination {combination}. Must be 1-10.")
    known_set = combo["known"]
    unknown_set = combo["unknown"]

    file_list = get_file_list(video_dir, extension='.mp4')
    known_files = []
    unknown_files = []
    
    # Separate files based on the emotion extracted from the filename.
    for f in file_list:
        info = parse_ravdess_info(f)
        emo = info['emotion']
        if emo in known_set:
            known_files.append(f)
        elif emo in unknown_set:
            unknown_files.append(f)
        else:
            continue

    # Shuffle known files and split: 6/7 for training, 1/7 for final testing.
    random.shuffle(known_files)
    n_known = len(known_files)
    n_train = int((6/7) * n_known)
    train_known = known_files[:n_train]
    test_known = known_files[n_train:]
    # Final test set is the union of reserved known (1/7) and all unknown files.
    final_test = test_known + unknown_files

    rows = []
    # Mark training known files as "train"
    for f in train_known:
        info = parse_ravdess_info(f)
        rows.append({"filename": f, "category": "train", "emo_label": info['emotion']})
    # Mark the reserved known files as "test"
    for f in test_known:
        info = parse_ravdess_info(f)
        rows.append({"filename": f, "category": "test", "emo_label": info['emotion']})
    # Mark all unknown files as "test" with label 8.
    for f in unknown_files:
        rows.append({"filename": f, "category": "test", "emo_label": 8})

    df = pd.DataFrame(rows, columns=["filename", "category", "emo_label"])
    df.to_csv(output_csv, index=False)
    print(f"CSV file generated with {len(rows)} entries at {output_csv}")
    
    return train_known, final_test

#############################
# Dataset Class for Open-set (Using CSV-based Label Mapping)
#############################
class RAVDESSOpenSetDataset(Dataset):
    def __init__(self, video_dir, audio_dir, file_list, label_mapping,
                 modality='both', num_frames=16, video_transform=None,
                 audio_transform=None, target_sample_rate=16000):
        """
        Args:
            video_dir (str): Directory containing video (.mp4) files.
            audio_dir (str or None): Directory containing audio (.wav) files.
                If None, audio is loaded from the video file.
            file_list (list): List of filenames.
            label_mapping (dict): Mapping from filename to emo_label (from CSV).
            modality (str): 'video', 'audio', or 'both'.
            num_frames (int): Number of video frames to sample.
            video_transform: Transform to apply to each video frame.
            audio_transform: Transform to apply to the audio waveform.
            target_sample_rate (int): Audio sample rate.
        """
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.file_list = file_list
        self.label_mapping = label_mapping
        self.modality = modality
        self.num_frames = num_frames
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        if filename not in self.label_mapping:
            raise KeyError(f"Label for {filename} not found.")
        label = self.label_mapping[filename]
        data = {}
        # Process video if required.
        if self.modality in ['video', 'both']:
            video_path = os.path.join(self.video_dir, filename)
            frames = load_video_frames(video_path, self.num_frames)
            if self.video_transform:
                frames = [self.video_transform(Image.fromarray(frame)) for frame in frames]
            else:
                frames = [transforms.ToTensor()(Image.fromarray(frame)) for frame in frames]
            data['video'] = torch.stack(frames)
        # Process audio if required.
        if self.modality in ['audio', 'both']:
            base, _ = os.path.splitext(filename)
            audio_filename = base + '.wav'
            audio_path = os.path.join(self.audio_dir, audio_filename) if self.audio_dir is not None else os.path.join(self.video_dir, filename)
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file {audio_path} not found.")
            waveform = load_audio_file(audio_path, self.target_sample_rate)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
            data['audio'] = waveform
        return data, label

#############################
# DataLoader Helper Function for Open-set with Train/Eval Split
#############################
def get_openset_dataloaders(video_dir, audio_dir, modality, combination, output_csv,
                            batch_size=16, num_frames=16, video_transform=None,
                            audio_transform=None, target_sample_rate=16000, num_workers=4,
                            train_eval_split=0.8):
    """
    For a given open-set combination (1-10), generate the CSV splits and build three dataloaders:
      - train_loader: for model updating (subset of known files)
      - eval_loader: for evaluation during training (subset of known files held out from training)
      - test_loader: for final testing (union of reserved known and all unknown files)
    
    The CSV will have three columns: filename, category, and emo_label.
    Known files are split so that:
      - train: train_eval_split (e.g. 80%) of known files (marked "train")
      - eval: remaining known files (20%) are used for evaluation during training (marked "eval")
    All unknown files are marked as "test" with label 8.
    
    Returns: train_loader, eval_loader, test_loader.
    """
    # Generate CSV; note that generate_openset_csv marks reserved known as "test".
    # We will re-split the "train" category into two subsets: training and evaluation.
    train_known, final_test = generate_openset_csv(video_dir, combination, output_csv)
    
    # Load the CSV file.
    df = pd.read_csv(output_csv)
    
    # Build initial mappings:
    # Files with category "train" (these are known files intended for training/evaluation).
    full_train_files = [row['filename'] for _, row in df.iterrows() if row['category'] == 'train']
    full_train_mapping = {row['filename']: row['emo_label'] for _, row in df.iterrows() if row['category'] == 'train'}
    # Files with category "test" (these include reserved known and unknown).
    test_files = [row['filename'] for _, row in df.iterrows() if row['category'] == 'test']
    test_mapping = {row['filename']: row['emo_label'] for _, row in df.iterrows() if row['category'] == 'test'}
    
    # Now split the full_train_files into training and evaluation subsets.
    random.shuffle(full_train_files)
    n_full = len(full_train_files)
    n_train = int(train_eval_split * n_full)
    train_files = full_train_files[:n_train]
    eval_files = full_train_files[n_train:]
    
    # Build corresponding mappings.
    train_mapping = {f: full_train_mapping[f] for f in train_files}
    eval_mapping = {f: full_train_mapping[f] for f in eval_files}
    
    # Define default video transform if not provided.
    if video_transform is None:
        video_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    
    # Create dataset objects.
    train_dataset = RAVDESSOpenSetDataset(video_dir, audio_dir, train_files, train_mapping,
                                           modality, num_frames, video_transform,
                                           audio_transform, target_sample_rate)
    eval_dataset = RAVDESSOpenSetDataset(video_dir, audio_dir, eval_files, eval_mapping,
                                          modality, num_frames, video_transform,
                                          audio_transform, target_sample_rate)
    test_dataset = RAVDESSOpenSetDataset(video_dir, audio_dir, test_files, test_mapping,
                                          modality, num_frames, video_transform,
                                          audio_transform, target_sample_rate)
    
    # Create dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, eval_loader, test_loader

#############################
# Example Usage
#############################
if __name__ == '__main__':
    video_directory = "/path/to/ravdess/videos"    # Directory with video files (flat folder expected here)
    audio_directory = "/path/to/ravdess/audios"      # Directory with corresponding audio files
    modality = 'both'  # Options: 'video', 'audio', or 'both'
    combination_index = 1  # Choose a combination number between 1 and 10
    output_csv_path = "openset_split.csv"
    
    # Get three dataloaders: train (for updating), eval (for training-time evaluation), and test (final testing)
    train_loader, eval_loader, test_loader = get_openset_dataloaders(video_directory, audio_directory,
                                                                     modality, combination_index, output_csv_path,
                                                                     batch_size=16, num_frames=16,
                                                                     target_sample_rate=16000, num_workers=4,
                                                                     train_eval_split=0.8)
    
    # train_loader, eval_loader, and test_loader are now ready for your training pipeline.
