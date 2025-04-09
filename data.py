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
      - modality: first field (e.g., "01" = full-AV, "02" = video-only, "03" = audio-only)
      - vocal_channel: second field (e.g., "01" for speech)
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
        'emotion': int(parts[2]) - 1
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
def get_file_list(data_dir, allowed_modalities=None):
    """
    Retrieve a list of filenames from data_dir with extensions .mp4 or .wav.
    Only include files where vocal_channel == "01" (speech).
    Optionally filter files based on the modality field (first field).
    
    Args:
        data_dir (str): Directory containing the files.
        allowed_modalities (set or None): E.g., {"02", "03"} to filter files.
    Returns:
        list: Filtered and shuffled list of filenames.
    """
    allowed_ext = {'.mp4', '.wav'}
    files = []
    for f in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in allowed_ext:
            continue
        parts = f.split('-')
        if len(parts) < 2:
            continue
        if parts[1] != '01':  # Only select speech files.
            continue
        if allowed_modalities is not None and parts[0] not in allowed_modalities:
            continue
        files.append(f)
    random.shuffle(files)
    return files

#############################
# CSV Generation & Open-set Splitting
#############################
def generate_openset_csv(data_dir, combination, output_csv, allowed_modalities=None):
    """
    Generate a CSV file for the open-set split.
    The CSV will have columns:
      - filename (e.g., "02-01-06-01-02-01-12.mp4" or "03-01-06-01-02-01-12.wav")
      - category ("train" or "test")
      - emo_label (for known files, the original 0-indexed label; for unknown files, fixed to 8)
    
    Only files whose modality is in allowed_modalities will be included.
    Known files are randomly split: 6/7 for training and 1/7 reserved for test.
    
    Returns:
        (list, list): train_files, final_test_files
    """
    combo = COMBINATION_SPLITS.get(combination)
    if combo is None:
        raise ValueError(f"Invalid combination {combination}. Must be 1-10.")
    known_set = combo["known"]
    unknown_set = combo["unknown"]

    file_list = get_file_list(data_dir, allowed_modalities=allowed_modalities)
    known_files = []
    unknown_files = []
    
    for f in file_list:
        info = parse_ravdess_info(f)
        emo = info['emotion']
        if emo in known_set:
            known_files.append(f)
        elif emo in unknown_set:
            unknown_files.append(f)
        else:
            continue

    random.shuffle(known_files)
    n_known = len(known_files)
    n_train = int((6/7) * n_known)
    train_known = known_files[:n_train]
    test_known = known_files[n_train:]
    final_test = test_known + unknown_files

    rows = []
    for f in train_known:
        info = parse_ravdess_info(f)
        rows.append({"filename": f, "category": "train", "emo_label": info['emotion']})
    for f in test_known:
        info = parse_ravdess_info(f)
        rows.append({"filename": f, "category": "test", "emo_label": info['emotion']})
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
    def __init__(self, data_dir, file_list, label_mapping,
                 modality='both', num_frames=16, video_transform=None,
                 audio_transform=None, target_sample_rate=16000):
        """
        Args:
            data_dir (str): Directory containing the data files (.mp4 and .wav).
            file_list (list): List of filenames.
            label_mapping (dict): Mapping from filename to emo_label (from CSV).
            modality (str): 'video', 'audio', or 'both'. Select which modality(ies) to load.
            num_frames (int): Number of video frames to sample.
            video_transform: Transform to apply to video frames.
            audio_transform: Transform to apply to the audio waveform.
            target_sample_rate (int): Audio sample rate.
        """
        self.data_dir = data_dir
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
        info = parse_ravdess_info(filename)
        file_mod = info['modality']  # "02" (video-only) or "03" (audio-only)
        file_ext = os.path.splitext(filename)[1].lower()  # .mp4 or .wav
        
        # Video Processing: Only load if modality is video-only ("02")
        if self.modality in ['video', 'both']:
            if file_ext == '.mp4' and file_mod == "02":
                video_path = os.path.join(self.data_dir, filename)
                frames = load_video_frames(video_path, self.num_frames)
                if self.video_transform:
                    frames = [self.video_transform(Image.fromarray(frame)) for frame in frames]
                else:
                    frames = [transforms.ToTensor()(Image.fromarray(frame)) for frame in frames]
                data['video'] = torch.stack(frames)
            elif self.modality == 'video':
                raise FileNotFoundError(f"No video available for file {filename} in selected modalities.")

        # Audio Processing: Only load if modality is audio-only ("03")
        if self.modality in ['audio', 'both']:
            if file_ext == '.wav' and file_mod == "03":
                audio_path = os.path.join(self.data_dir, filename)
            elif self.modality == 'audio':
                raise FileNotFoundError(f"No audio available for file {filename} in selected modalities.")
            else:
                audio_path = None
            if audio_path is not None:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file {audio_path} not found for file {filename}")
                waveform = load_audio_file(audio_path, self.target_sample_rate)
                if self.audio_transform:
                    waveform = self.audio_transform(waveform)
                data['audio'] = waveform

        return data, label

#############################
# DataLoader Helper Function for Open-set with Train/Eval Split
#############################
def get_openset_dataloaders(data_dir, combination, output_csv_dir,
                            modality='both', batch_size=16, num_frames=16,
                            video_transform=None, audio_transform=None,
                            target_sample_rate=16000, num_workers=4,
                            train_eval_split=0.8,
                            train_allowed_modalities={"02", "03"}):
    """
    For a given open-set combination, generate CSV splits and build three dataloaders.
    The CSV file is automatically created in output_csv_dir with a name that includes the combination.
    
    Args:
        data_dir (str): Directory containing the data files.
        combination (int): Open-set combination number (1-10).
        output_csv_dir (str): Directory where the output CSV file should be created.
        modality (str): 'video', 'audio', or 'both'. 
        batch_size (int): Batch size for the DataLoaders.
        num_frames (int): Number of video frames to sample.
        video_transform: Video transform function.
        audio_transform: Audio transform function.
        target_sample_rate (int): Target audio sample rate.
        num_workers (int): Number of workers for DataLoader.
        train_eval_split (float): Fraction of training files to use for actual training.
        train_allowed_modalities (set or None): E.g., {"02", "03"} to include only video-only and audio-only files.
    Returns:
        train_loader, eval_loader, test_loader
    """
    # Automatically construct the CSV file path in output_csv_dir.
    output_csv = os.path.join(output_csv_dir, f"openset_split_combination_{combination}.csv")
    
    train_known, final_test = generate_openset_csv(data_dir, combination, output_csv,
                                                   allowed_modalities=train_allowed_modalities)
    
    df = pd.read_csv(output_csv)
    
    full_train_files = [row['filename'] for _, row in df.iterrows() if row['category'] == 'train']
    full_train_mapping = {row['filename']: row['emo_label'] for _, row in df.iterrows() if row['category'] == 'train'}
    test_files = [row['filename'] for _, row in df.iterrows() if row['category'] == 'test']
    test_mapping = {row['filename']: row['emo_label'] for _, row in df.iterrows() if row['category'] == 'test'}
    
    random.shuffle(full_train_files)
    n_full = len(full_train_files)
    n_train = int(train_eval_split * n_full)
    train_files = full_train_files[:n_train]
    eval_files = full_train_files[n_train:]
    
    train_mapping = {f: full_train_mapping[f] for f in train_files}
    eval_mapping = {f: full_train_mapping[f] for f in eval_files}
    
    if video_transform is None:
        video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_dataset = RAVDESSOpenSetDataset(data_dir, train_files, train_mapping,
                                           modality, num_frames, video_transform,
                                           audio_transform, target_sample_rate)
    eval_dataset = RAVDESSOpenSetDataset(data_dir, eval_files, eval_mapping,
                                          modality, num_frames, video_transform,
                                          audio_transform, target_sample_rate)
    test_dataset = RAVDESSOpenSetDataset(data_dir, test_files, test_mapping,
                                          modality, num_frames, video_transform,
                                          audio_transform, target_sample_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, eval_loader, test_loader
