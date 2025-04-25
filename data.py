import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
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
    1: {"known": {1,7,5,4,6}, "unknown": {0,2,3}},
    2: {"known": {1,2,3,4,6}, "unknown": {0,7,5}},
    3: {"known": {1,7,3,5,6}, "unknown": {0,2,4}},
    4: {"known": {0,2,3,5,4}, "unknown": {1,7,6}},
    5: {"known": {1,2,5,6,7}, "unknown": {0,3,4}},
    6: {"known": {0,2,3,4,7}, "unknown": {1,5,6}},
    7: {"known": {0,1,3,6,7}, "unknown": {2,5,4}},
    8: {"known": {0,1,2,5,4}, "unknown": {7,3,6}},
    9: {"known": {0,7,3,4,6}, "unknown": {1,2,5}},
    10:{"known": {0,2,3,5,6}, "unknown": {1,7,4}},
}

#############################
# Collate function for audio padding
#############################
def collate_fn_audio(batch):
    """
    对 batch 中的 audio waveform 进行 padding，保证同一 batch 内所有样本长度一致。
    假设每个 waveform 的 shape 为 (1, num_samples)（单声道）
    """
    waveforms, labels = zip(*batch)
    max_length = max(w.shape[1] for w in waveforms)
    padded = []
    for w in waveforms:
        pad_len = max_length - w.shape[1]
        padded.append(nn.functional.pad(w, (0, pad_len), mode='constant', value=0))
    return torch.stack(padded, dim=0), torch.tensor(labels)

def collate_fn_default(batch):
    """
    Collate function for both video and audio: 
    - stacks video tensors (all are [num_frames, C, H, W]) into [B, num_frames, C, H, W]
    - pads each audio tensor (shape [num_frames, L_i]) along dim=1 to the batch max length, 
      then stacks into [B, num_frames, max_L]
    """
    samples, labels = zip(*batch)

    # Stack video
    video_batch = torch.stack([s['video'] for s in samples], dim=0)

    # Collect and pad audio
    waveforms = [s['audio'] for s in samples]  # each is [num_frames, L_i]
    max_len = max(w.shape[1] for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        pad_len = max_len - w.shape[1]
        padded_waveforms.append(nn.functional.pad(w, (0, pad_len), mode='constant', value=0))
    audio_batch = torch.stack(padded_waveforms, dim=0)

    return {'video': video_batch, 'audio': audio_batch}, torch.tensor(labels)



#############################
# Parsing filename → metadata
#############################
def parse_ravdess_info(filename):
    """
    Parse a RAVDESS filename and return:
      - modality: "02" or "03"
      - vocal_channel
      - emotion: 0–7
    """
    base, _ = os.path.splitext(os.path.basename(filename))
    parts = base.split('-')
    if len(parts) < 7:
        raise ValueError("Filename missing fields")
    return {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': int(parts[2]) - 1
    }

#############################
# Video loader (32 frames)
#############################
def load_video_frames(video_path, num_frames=32, transform=None):
    """
    读取视频文件并均匀采样 num_frames 帧
    返回 Tensor (num_frames, C, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")

    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames, fid = [], 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        if fid in idxs:
            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = transform(img) if transform else transforms.ToTensor()(img)
            frames.append(img)
        fid += 1
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.stack(frames)

#############################
# Audio loader
#############################
def load_audio_file(audio_path, target_sample_rate=16000):
    """
    Load waveform and resample if needed.
    """
    wv, sr = torchaudio.load(audio_path)
    if sr != target_sample_rate:
        wv = torchaudio.transforms.Resample(sr, target_sample_rate)(wv)
    return wv

#############################
# List files in dir
#############################
def get_file_list(data_dir, allowed_modalities=None):
    """
    List .mp4/.wav files with vocal_channel '01' only.
    """
    exts = {'.mp4', '.wav'}
    fl = []
    for f in sorted(os.listdir(data_dir)):
        e = os.path.splitext(f)[1].lower()
        if e in exts:
            parts = f.split('-')
            if len(parts)>=2 and parts[1]=='01':
                if not allowed_modalities or parts[0] in allowed_modalities:
                    fl.append(f)
    random.shuffle(fl)
    return fl

#############################
# CSV Generation: standalone function
#############################
def generate_openset_csv(data_dir, combination, csv_dir, allowed_modalities={"02","03"}):
    """
    Generate CSV with columns: video_filename, audio_filename, category, emo_label
    and save to csv_dir as multimodal-{combination}.csv
    """
    combo = COMBINATION_SPLITS.get(combination)
    if not combo:
        raise ValueError("combination 1–10 only")
    known, unknown = combo['known'], combo['unknown']

    files = get_file_list(data_dir, allowed_modalities)
    utt2emo = {}
    for f in files:
        info = parse_ravdess_info(f)
        u = os.path.splitext(f)[0].split('-',1)[1]
        utt2emo[u] = info['emotion']

    valid = []
    for u, emo in utt2emo.items():
        v = f"02-{u}.mp4"; a = f"03-{u}.wav"
        if os.path.exists(os.path.join(data_dir, v)) and os.path.exists(os.path.join(data_dir, a)):
            valid.append(u)

    k_utts = [u for u in valid if utt2emo[u] in known]
    un_utts= [u for u in valid if utt2emo[u] in unknown]
    random.shuffle(k_utts)
    n_train = int((6/7)*len(k_utts))
    train_k, test_k = k_utts[:n_train], k_utts[n_train:]

    rows = []
    for u in train_k:
        rows.append({'video_filename': f"02-{u}.mp4",
                     'audio_filename': f"03-{u}.wav",
                     'category': 'train',
                     'emo_label': utt2emo[u]})
    for u in test_k:
        rows.append({'video_filename': f"02-{u}.mp4",
                     'audio_filename': f"03-{u}.wav",
                     'category': 'test',
                     'emo_label': utt2emo[u]})
    for u in un_utts:
        rows.append({'video_filename': f"02-{u}.mp4",
                     'audio_filename': f"03-{u}.wav",
                     'category': 'test',
                     'emo_label': 8})

    df = pd.DataFrame(rows, columns=['video_filename','audio_filename','category','emo_label'])
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"multimodal-{combination}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV generated: {len(rows)} rows → {csv_path}")

#############################
# Paired Dataset
#############################
class RAVDESSOpenSetDataset(Dataset):
    def __init__(self, data_dir, video_files, audio_files, labels,
                 modality='both', num_frames=32,
                 video_transform=None, audio_transform=None,
                 target_sample_rate=16000):
        self.data_dir = data_dir
        self.vfiles = video_files
        self.afiles = audio_files
        self.labels = labels
        self.modality = modality
        self.num_frames = num_frames
        self.vt = video_transform
        self.at = audio_transform
        self.sr = target_sample_rate

    def __len__(self):
        return len(self.vfiles)

    def __getitem__(self, idx):
        sample, lab = {}, self.labels[idx]
        if self.modality in ['video','both']:
            sample['video'] = load_video_frames(
                os.path.join(self.data_dir, self.vfiles[idx]),
                num_frames=self.num_frames,
                transform=self.vt)
        if self.modality in ['audio','both']:
            wv = load_audio_file(
                os.path.join(self.data_dir, self.afiles[idx]), self.sr)
            if wv.size(0)>1:
                wv = wv.mean(dim=0, keepdim=True)
            total = wv.size(1)
            fl = total // self.num_frames
            if fl == 0:
                raise ValueError("Audio too short")
            desired = fl * self.num_frames
            wv = (wv if total>=desired else nn.functional.pad(wv, (0, desired-total)))[:, :desired]
            wv = wv.view(1, self.num_frames, fl).squeeze(0)
            if self.at:
                wv = torch.stack([self.at(x) for x in wv])
            sample['audio'] = wv
        return sample, lab

#############################
# DataLoader builder: custom splits
#############################
def get_openset_dataloaders(data_dir, csv_dir,
                            combination, modality='both',
                            batch_size=4, num_frames=32,
                            video_transform=None, audio_transform=None,
                            target_sample_rate=16000, num_workers=4):
    """
    Train loader: all category=='train'.
    Val loader: all category=='test' & emo_label != 8.
    Test loader: union of val set and all emo_label == 8.
    """
    # Load and split CSV
    csv_path = os.path.join(csv_dir, f"multimodal-combination-{combination}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    train_df = df[df.category == 'train'].reset_index(drop=True)
    val_df   = df[(df.category == 'test') & (df.emo_label != 8)].reset_index(drop=True)
    unk_df   = df[(df.category == 'test') & (df.emo_label == 8)].reset_index(drop=True)
    test_df  = pd.concat([val_df, unk_df]).reset_index(drop=True)

    # Default video transform if none
    if video_transform is None:
        video_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    # Build datasets
    train_ds = RAVDESSOpenSetDataset(data_dir,
                                     train_df.video_filename.tolist(),
                                     train_df.audio_filename.tolist(),
                                     train_df.emo_label.tolist(),
                                     modality, num_frames,
                                     video_transform, audio_transform,
                                     target_sample_rate)
    val_ds   = RAVDESSOpenSetDataset(data_dir,
                                     val_df.video_filename.tolist(),
                                     val_df.audio_filename.tolist(),
                                     val_df.emo_label.tolist(),
                                     modality, num_frames,
                                     video_transform, audio_transform,
                                     target_sample_rate)
    test_ds  = RAVDESSOpenSetDataset(data_dir,
                                     test_df.video_filename.tolist(),
                                     test_df.audio_filename.tolist(),
                                     test_df.emo_label.tolist(),
                                     modality, num_frames,
                                     video_transform, audio_transform,
                                     target_sample_rate)

    # Choose collate function: audio uses padding, others use default stack
    if modality == 'audio':
        collate = collate_fn_audio
    else:
        collate = collate_fn_default

    # Create DataLoaders with consistent collate_fn
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate)

    return train_loader, val_loader, test_loader


