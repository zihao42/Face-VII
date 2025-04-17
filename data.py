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
# CSV Gen: video, audio, category, emo_label
#############################
def generate_openset_csv(data_dir, combination, output_csv, allowed_modalities=None):
    """
    Generate CSV with columns: video_filename, audio_filename, category, emo_label
    """
    combo = COMBINATION_SPLITS.get(combination)
    if not combo:
        raise ValueError("combination 1–10 only")
    known, unknown = combo['known'], combo['unknown']

    files = get_file_list(data_dir, allowed_modalities)
    # map utt_id -> emotion
    utt2emo = {}
    for f in files:
        info = parse_ravdess_info(f)
        u = os.path.splitext(f)[0].split('-',1)[1]
        utt2emo[u] = info['emotion']

    # keep only utts with both modalities
    valid = []
    for u, emo in utt2emo.items():
        v = f"02-{u}.mp4"; a = f"03-{u}.wav"
        if os.path.exists(os.path.join(data_dir, v)) and \
           os.path.exists(os.path.join(data_dir, a)):
            valid.append(u)

    # split known vs unknown
    k_utts = [u for u in valid if utt2emo[u] in known]
    un_utts= [u for u in valid if utt2emo[u] in unknown]
    random.shuffle(k_utts)
    n_train = int((6/7)*len(k_utts))
    train_k, test_k = k_utts[:n_train], k_utts[n_train:]
    rows = []

    for u in train_k:
        rows.append({
            'video_filename': f"02-{u}.mp4",
            'audio_filename': f"03-{u}.wav",
            'category': 'train',
            'emo_label': utt2emo[u]
        })
    for u in test_k:
        rows.append({
            'video_filename': f"02-{u}.mp4",
            'audio_filename': f"03-{u}.wav",
            'category': 'test',
            'emo_label': utt2emo[u]
        })
    for u in un_utts:
        rows.append({
            'video_filename': f"02-{u}.mp4",
            'audio_filename': f"03-{u}.wav",
            'category': 'test',
            'emo_label': 8
        })

    df = pd.DataFrame(rows, columns=[
        'video_filename','audio_filename','category','emo_label'
    ])
    df.to_csv(output_csv, index=False)
    print(f"CSV generated: {len(rows)} rows → {output_csv}")

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
        sample = {}
        lab = self.labels[idx]

        if self.modality in ['video','both']:
            vf = self.vfiles[idx]
            sample['video'] = load_video_frames(
                os.path.join(self.data_dir, vf),
                num_frames=self.num_frames,
                transform=self.vt
            )
        if self.modality in ['audio','both']:
            af = self.afiles[idx]
            wv = load_audio_file(os.path.join(self.data_dir, af), self.sr)
            if wv.size(0)>1:
                wv = wv.mean(dim=0, keepdim=True)
            total = wv.size(1)
            fl = total // self.num_frames
            if fl==0:
                raise ValueError("Audio too short")
            desired = fl*self.num_frames
            wv = (wv if total>=desired else nn.functional.pad(wv,(0,desired-total)))[:,:desired]
            wv = wv.view(1,self.num_frames,fl).squeeze(0)
            if self.at:
                wv = torch.stack([self.at(x) for x in wv])
            sample['audio'] = wv

        return sample, lab

#############################
# DataLoader builder
#############################
def get_openset_dataloaders(data_dir, combination, output_csv_dir,
                            modality='both', batch_size=4, num_frames=32,
                            video_transform=None, audio_transform=None,
                            target_sample_rate=16000, num_workers=4,
                            train_eval_split=0.8,
                            train_allowed_modalities={"02","03"}):
    """
    Generate paired video+audio dataloaders.
    """
    os.makedirs(output_csv_dir, exist_ok=True)
    csv_path = os.path.join(
        output_csv_dir, f"multimodal-{combination}.csv"
    )
    generate_openset_csv(
        data_dir, combination, csv_path,
        allowed_modalities=train_allowed_modalities
    )

    df = pd.read_csv(csv_path)
    train_df = df[df.category=='train'].sample(frac=1).reset_index(drop=True)
    test_df  = df[df.category=='test'].reset_index(drop=True)

    cut = int(train_eval_split * len(train_df))
    df_tr = train_df.iloc[:cut]
    df_val= train_df.iloc[cut:]

    train_ds = RAVDESSOpenSetDataset(
        data_dir,
        df_tr.video_filename.tolist(),
        df_tr.audio_filename.tolist(),
        df_tr.emo_label.tolist(),
        modality, num_frames,
        video_transform, audio_transform,
        target_sample_rate
    )
    val_ds = RAVDESSOpenSetDataset(
        data_dir,
        df_val.video_filename.tolist(),
        df_val.audio_filename.tolist(),
        df_val.emo_label.tolist(),
        modality, num_frames,
        video_transform, audio_transform,
        target_sample_rate
    )
    test_ds = RAVDESSOpenSetDataset(
        data_dir,
        test_df.video_filename.tolist(),
        test_df.audio_filename.tolist(),
        test_df.emo_label.tolist(),
        modality, num_frames,
        video_transform, audio_transform,
        target_sample_rate
    )

    collate = collate_fn_audio if modality=='audio' else None

    if video_transform is None:
        video_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

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
