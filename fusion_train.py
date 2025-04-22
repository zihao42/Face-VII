#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm

from data import load_video_frames, load_audio_file
from feature_fusion import MultimodalTransformer
from audio_feature_extract import load_audio_backbone, extract_audio_features_from_backbone
from visual_feature_extract import load_timesformer_backbone, extract_frame_features_from_backbone


class RAVDESSMultimodalDataset(Dataset):
    """
    样本列表驱动的数据集：
    samples: list of (audio_filename, video_filename, raw_emo_label)
    label_map: dict 从 raw_emo_label 到 紧凑的 0..C-1
    """
    def __init__(self, samples, media_dir, num_frames, label_map, video_transform=None):
        self.media_dir = media_dir
        self.num_frames = num_frames
        self.label_map = label_map
        # 应用标签映射
        self.samples = [(a, v, label_map[lbl]) for (a, v, lbl) in samples]
        # 视频预处理
        if video_transform is None:
            self.video_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.video_transform = video_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_fn, video_fn, label = self.samples[idx]
        wav_path = os.path.join(self.media_dir, audio_fn)
        mp4_path = os.path.join(self.media_dir, video_fn)

        # 加载音频，并合并多通道
        waveform = load_audio_file(wav_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 加载视频帧序列
        frames = load_video_frames(
            mp4_path,
            num_frames=self.num_frames,
            transform=self.video_transform
        )
        return waveform, frames, label


def collate_fn_modality(batch):
    waveforms, videos, labels = zip(*batch)
    # 统一音频长度
    max_len = max(wf.shape[1] for wf in waveforms)
    padded = []
    for wf in waveforms:
        pad_len = max_len - wf.shape[1]
        wf_p = F.pad(wf, (0, pad_len), mode='constant', value=0)
        if wf_p.dim() == 2:
            wf_p = wf_p.squeeze(0)
        padded.append(wf_p)
    audio_batch = torch.stack(padded, dim=0)
    video_batch = torch.stack(videos, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)
    return audio_batch, video_batch, label_batch


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    num_epochs: int = 10,
    video_comb: int = 1,
    audio_comb: int = 1,
    weights_dir_visual: str = "weights/backbones/visual",
    weights_dir_audio: str = "weights/backbones/audio"
):
    # 预加载视觉与音频 backbone
    v_path = os.path.join(
        weights_dir_visual,
        f"openset_split_combination_{video_comb}_timesformer_backbone.pth"
    )
    a_path = os.path.join(
        weights_dir_audio,
        f"openset_split_combination_{audio_comb}_wav2vec_backbone.pth"
    )
    backbone_v = load_timesformer_backbone(v_path, device)
    backbone_a = load_audio_backbone(a_path, device)

    model.to(device)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = total_correct = total_samples = 0
        for wavs, vids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat_v = extract_frame_features_from_backbone(vids, backbone_v)
                feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
            logits, _ = model([feat_a, feat_v])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_losses.append(total_loss / total_samples)
        train_accs.append(total_correct / total_samples)

        # 验证阶段（只评估已知类别）
        model.eval()
        v_loss = v_correct = v_samples = 0
        with torch.no_grad():
            for wavs, vids, labels in val_loader:
                wavs, vids, labels = wavs.to(device), vids.to(device), labels.to(device)
                feat_v = extract_frame_features_from_backbone(vids, backbone_v)
                feat_a = extract_audio_features_from_backbone(wavs, backbone_a)
                logits, _ = model([feat_a, feat_v])
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                v_loss += loss.item() * labels.size(0)
                v_correct += (preds == labels).sum().item()
                v_samples += labels.size(0)

        val_losses.append(v_loss / v_samples)
        val_accs.append(v_correct / v_samples)

        print(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.4f} | "
            f"Val Loss {val_losses[-1]:.4f}, Acc {val_accs[-1]:.4f}"
        )

    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "train_accs": train_accs, "val_accs": val_accs}


def main():
    parser = argparse.ArgumentParser(description="Train multimodal fusion on RAVDESS")
    parser.add_argument("--csv_file", type=str,
                        default="/media/data1/ningtong/wzh/datasets/RAVDESS/csv/multimodel/multimodal-combination-1.csv",
                        help="包含 audio/video 文件名、category、emo_label 的 CSV 文件路径")
    parser.add_argument("--media_dir", type=str,
                        default="/media/data1/ningtong/wzh/datasets/RAVDESS/data",
                        help="存放 .wav/.mp4 文件的目录")
    parser.add_argument("--output_dir", type=str,
                        default="./weights",
                        help="保存模型权重和曲线图的输出目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_frames", type=int, default=32,
                        help="每段视频采样帧数")
    parser.add_argument("--video_comb", type=int, default=1,
                        help="视觉 backbone 对应的组合 ID")
    parser.add_argument("--audio_comb", type=int, default=1,
                        help="音频 backbone 对应的组合 ID")
    args = parser.parse_args()

    # 1. 读取原始 CSV，构建全局标签映射
    raw_df = pd.read_csv(args.csv_file)
    unique_labels = sorted(raw_df['emo_label'].unique())
    label_map = {orig: i for i, orig in enumerate(unique_labels)}
    num_classes = len(label_map)

    # 2. 拆分训练/验证集：验证集只保留已知类别
    train_df = raw_df[raw_df['category'] == 'train']
    val_df   = raw_df[(raw_df['category'] == 'test') & (raw_df['emo_label'] != 8)]

    train_samples = list(zip(
        train_df['audio_filename'], train_df['video_filename'], train_df['emo_label']
    ))
    val_samples   = list(zip(
        val_df['audio_filename'],   val_df['video_filename'],   val_df['emo_label']
    ))

    # 3. 视频预处理（同 Dataset 内部）
    video_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. 构造 Dataset 和 DataLoader
    train_ds = RAVDESSMultimodalDataset(
        train_samples, args.media_dir, args.num_frames, label_map, video_transform
    )
    val_ds   = RAVDESSMultimodalDataset(
        val_samples,   args.media_dir, args.num_frames, label_map, video_transform
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_modality
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_modality
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}, sampling {args.num_frames} frames/video")

    # 5. 模型、损失、优化器
    model = MultimodalTransformer(modality_num=2, num_classes=num_classes, num_layers=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. 训练与验证
    trained_model, metrics = train_and_evaluate(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        num_epochs=args.epochs,
        video_comb=args.video_comb,
        audio_comb=args.audio_comb,
        weights_dir_visual=os.path.join(args.output_dir, "backbones", "visual"),
        weights_dir_audio=os.path.join(args.output_dir, "backbones", "audio")
    )

    # 7. 保存权重和曲线图
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(args.csv_file))[0]
    wpath = os.path.join(args.output_dir, f"{prefix}.pth")
    torch.save(trained_model.state_dict(), wpath)
    print(f"Saved model weights to {wpath}")

    plt.figure()
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(args.output_dir, f"{prefix}_loss.png"))
    print(f"Saved loss curve to {args.output_dir}/{prefix}_loss.png")

    plt.figure()
    plt.plot(metrics['train_accs'], label='Train Acc')
    plt.plot(metrics['val_accs'],   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(os.path.join(args.output_dir, f"{prefix}_acc.png"))
    print(f"Saved accuracy curve to {args.output_dir}/{prefix}_acc.png")


if __name__ == '__main__':
    main()
