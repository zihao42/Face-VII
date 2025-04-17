#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt

from audio_feature_extract import extract_audio_features
from visual_feature_extract import extract_frame_features
from feature_fusion import MultimodalTransformer


def load_video_frames(video_path, num_frames=32):
    """
    使用 OpenCV 从视频文件中均匀采样 num_frames 帧，返回 Tensor [T, C, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total-1, 0), num_frames, dtype=int)
    frames = []
    idx_set = set(indices.tolist())
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fid in idx_set:
            # BGR -> RGB, to tensor
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)
        fid += 1
    cap.release()
    # 如果帧数不足，重复最后一帧
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames)  # [T, C, H, W]


class RAVDESSMultimodalDataset(Dataset):
    def __init__(self, csv_file, media_dir, split, num_frames=32):
        df = pd.read_csv(csv_file, header=None, names=['file', 'split', 'label'])
        df = df[df['split'] == split]
        # 核心 ID = 文件名去掉扩展名
        self.samples = [(os.path.splitext(row['file'])[0], int(row['label']))
                        for _, row in df.iterrows()]
        self.media_dir = media_dir
        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        core, label = self.samples[idx]
        wav_path = os.path.join(self.media_dir, core + '.wav')
        mp4_path = os.path.join(self.media_dir, core + '.mp4')
        # 加载音频
        waveform, sr = torchaudio.load(wav_path)  # [C, S]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # 加载视频帧
        frames = load_video_frames(mp4_path, num_frames=self.num_frames)  # [T, C, H, W]
        return waveform, frames, label


def collate_fn_modality(batch):
    """
    批内对齐音频长度并堆叠视频帧
    """
    waveforms, videos, labels = zip(*batch)
    # 找到最长的音频长度，做 zero-pad
    max_len = max(wf.shape[1] for wf in waveforms)
    padded = []
    for wf in waveforms:
        pad_len = max_len - wf.shape[1]
        wf_p = F.pad(wf, (0, pad_len), mode='constant', value=0)
        # 如果是单通道，移除通道维度
        if wf_p.size(0) == 1:
            wf_p = wf_p.squeeze(0)
        padded.append(wf_p)
    audio_batch = torch.stack(padded, dim=0)  # [B, L]
    video_batch = torch.stack(videos, dim=0)  # [B, T, C, H, W]
    label_batch = torch.tensor(labels, dtype=torch.long)
    return audio_batch, video_batch, label_batch


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device,
                       num_epochs=10, video_comb=1, audio_comb=1):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    model.to(device)
    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_loss = total_correct = total_samples = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for wavs, vids, labels in loop:
            wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat_a = extract_audio_features(wavs, audio_comb, device=device)
                feat_v = extract_frame_features(vids, video_comb, device=device)
            logits, _ = model([feat_a, feat_v])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            loop.set_postfix(loss=loss.item())
        train_losses.append(total_loss / total_samples)
        train_accs.append(total_correct / total_samples)

        # 验证
        model.eval()
        val_loss = val_correct = val_samples = 0
        with torch.no_grad():
            for wavs, vids, labels in val_loader:
                wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
                feat_a = extract_audio_features(wavs, audio_comb, device=device)
                feat_v = extract_frame_features(vids, video_comb, device=device)
                logits, _ = model([feat_a, feat_v])
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)
        val_losses.append(val_loss / val_samples)
        val_accs.append(val_correct / val_samples)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.4f} | "
              f"Val Loss {val_losses[-1]:.4f}, Acc {val_accs[-1]:.4f}")

    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "train_accs": train_accs, "val_accs": val_accs}


def main():
    parser = argparse.ArgumentParser(description="Train multimodal fusion on RAVDESS")
    parser.add_argument("--csv_file", type=str,
                        default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/openset_split_combination_1.csv")
    parser.add_argument("--media_dir", type=str,
                        default="/media/data1/ningtong/wzh/datasets/RAVDESS")
    parser.add_argument("--output_dir", type=str,
                        default="/media/data1/ningtong/wzh/projects/Face-VII/weights/model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Number of frames sampled per video")
    args = parser.parse_args()

    prefix = os.path.splitext(os.path.basename(args.csv_file))[0]
    comb = int(prefix.split('_')[-1]) if prefix.split('_')[-1].isdigit() else 1

    train_ds = RAVDESSMultimodalDataset(args.csv_file, args.media_dir, 'train', args.num_frames)
    val_ds   = RAVDESSMultimodalDataset(args.csv_file, args.media_dir, 'test',  args.num_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_modality)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn_modality)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}, sampling {args.num_frames} frames/video")

    model = MultimodalTransformer(modality_num=2, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trained_model, metrics = train_and_evaluate(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        num_epochs=args.epochs,
        video_comb=comb, audio_comb=comb
    )

    os.makedirs(args.output_dir, exist_ok=True)
    weight_path = os.path.join(args.output_dir, f"{prefix}.pth")
    torch.save(trained_model.state_dict(), weight_path)
    print(f"Saved model weights to {weight_path}")

    plt.figure()
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Val Loss')
    plt.plot(metrics['train_accs'], label='Train Acc')
    plt.plot(metrics['val_accs'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plot_path = os.path.join(args.output_dir, f"{prefix}.png")
    plt.savefig(plot_path)
    print(f"Saved training curves to {plot_path}")

if __name__ == '__main__':
    main()
