#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torchaudio

# Import the pretrained wav2vec2 model from transformers
from transformers import Wav2Vec2Model

# Reuse the audio loading function from your data.py
from data import load_audio_file

# -------------------- 自定义 Collate 函数 --------------------
def collate_fn_audio(batch):
    """
    对 batch 中的 audio waveform 进行 padding，保证同一 batch 内所有样本长度一致。
    假设每个 waveform 的 shape 为 (1, num_samples)（单声道）
    """
    waveforms, labels = zip(*batch)
    max_length = max(waveform.shape[1] for waveform in waveforms)
    padded_waveforms = []
    for waveform in waveforms:
        pad_length = max_length - waveform.shape[1]
        padded_waveform = nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)
        padded_waveforms.append(padded_waveform)
    padded_waveforms = torch.stack(padded_waveforms, dim=0)  # (batch, 1, max_length)
    labels = torch.tensor(labels)
    return padded_waveforms, labels

# ===================== Audio Dataset Class =====================
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, target_sample_rate=16000):
        """
        初始化音频数据集。
        参数：
          csv_file: CSV 文件路径，包含 "filename", "category", 和 "emo_label" 列。
          audio_dir: 存放音频文件的目录。
          transform: 对 waveform 进行额外预处理的函数（可选）。
          target_sample_rate: 目标采样率。
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        
        df = pd.read_csv(csv_file)
        df = df[df['emo_label'] != 8].reset_index(drop=True)
        # 仅保留 .wav 文件
        df = df[df['filename'].str.lower().str.endswith('.wav')].reset_index(drop=True)
        
        unique_labels = sorted(df['emo_label'].unique())
        self.label_mapping = {orig: idx for idx, orig in enumerate(unique_labels)}
        
        self.samples = []
        for _, row in df.iterrows():
            filename = row['filename']
            label_orig = row['emo_label']
            label = self.label_mapping[label_orig]
            audio_path = os.path.join(audio_dir, filename)
            self.samples.append((audio_path, label))
        
        self.num_classes = len(unique_labels)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found.")
        waveform = load_audio_file(audio_path, target_sample_rate=self.target_sample_rate)
        # 如果音频为多通道，则取均值转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# ===================== Define Wav2Vec Classifier with Regularization and Feature Alignment =====================
class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes, target_time=32, dropout_rate=0.3):
        """
        初始化 Wav2Vec 分类器，用于提取对齐后的时间序列特征以便与视频特征做多模态融合。
        参数：
          num_classes: 输出类别数。
          target_time: 对齐后的时间步数（例如 16，与视频采样的帧数保持一致）。
          dropout_rate: 分类头中 Dropout 的比率。
        """
        super(Wav2VecClassifier, self).__init__()
        # 加载预训练的 wav2vec2 模型
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_size = self.backbone.config.hidden_size
        # 加入 Dropout 正则化后再接线性层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        self.target_time = target_time
        
    def forward(self, x):
        """
        Args:
          x (Tensor): 音频 waveform，形状 (batch_size, 1, samples)。
        Returns:
          logits: 分类输出 (batch, num_classes)。
          aligned_features: 时间对齐后的特征 (batch, target_time, hidden_size)，便于多模态融合。
        """
        # 将 (batch, 1, samples) squeeze 成 (batch, samples)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        outputs = self.backbone(x)
        hidden_states = outputs.last_hidden_state  # (batch, T, hidden_size)
        # 自适应池化：将 T 维固定为 target_time
        hidden_states_t = hidden_states.transpose(1, 2)  # (batch, hidden_size, T)
        aligned_features = nn.functional.adaptive_avg_pool1d(hidden_states_t, self.target_time)
        aligned_features = aligned_features.transpose(1, 2)  # (batch, target_time, hidden_size)
        # 平均池化对齐后的特征得到全局特征
        pooled = aligned_features.mean(dim=1)
        logits = self.classifier(pooled)
        return logits, aligned_features

# ===================== Training and Evaluation Function =====================
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")
        for waveforms, labels in epoch_iter:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * waveforms.size(0)
            _, preds = torch.max(logits, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)
            epoch_iter.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                logits, _ = model(waveforms)
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * waveforms.size(0)
                _, preds = torch.max(logits, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Val   Acc: {epoch_val_acc:.4f}")
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    return model, metrics

# ===================== Main Function =====================
def main():
    # ---------------------- 路径设置 ----------------------
    csv_dir = "/media/data1/ningtong/wzh/datasets/RAVDESS/csv/"
    audio_dir = "/media/data1/ningtong/wzh/datasets/RAVDESS"
    backbone_save_dir = "/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/audio"
    os.makedirs(backbone_save_dir, exist_ok=True)
    
    # ---------------------- 训练参数 ----------------------
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-5
    val_split = 0.2  # 验证集比例
    
    # ---------------------- 设备设置 ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------- 读取 CSV 并训练 ----------------------
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for csv_file_name in csv_files:
        full_csv_path = os.path.join(csv_dir, csv_file_name)
        print(f"Processing CSV file: {csv_file_name}")
        
        dataset = AudioDataset(csv_file=full_csv_path, audio_dir=audio_dir, target_sample_rate=16000)
        print(f"  Total samples: {len(dataset)}, Number of classes: {dataset.num_classes}")
        
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_audio)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_audio)
        
        # 指定 target_time 与视频采样帧数一致（例如 16）
        model = Wav2VecClassifier(num_classes=dataset.num_classes, target_time=32, dropout_rate=0.3)
        model = model.to(device)
        
        # 加入 L2 正则化（weight_decay）到 Adam 优化器中
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print(f"  Training on {csv_file_name} ...")
        model, metrics = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)
        
        # 绘制训练和验证曲线
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics['train_losses'], label="Train Loss")
        plt.plot(epochs, metrics['val_losses'], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics['train_accs'], label="Train Acc")
        plt.plot(epochs, metrics['val_accs'], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")
        
        plt.tight_layout()
        plot_save_path = os.path.join(backbone_save_dir, f"{os.path.splitext(csv_file_name)[0]}_training_curve.png")
        plt.savefig(plot_save_path)
        plt.close()
        print(f"  Saved training curve to {plot_save_path}")
        
        # 训练结束后，将分类头替换为 Identity，只保存 backbone 权重
        model.classifier = nn.Identity()
        backbone_save_path = os.path.join(backbone_save_dir, f"{os.path.splitext(csv_file_name)[0]}_wav2vec_backbone.pth")
        torch.save(model.backbone.state_dict(), backbone_save_path)
        print(f"  Saved backbone weights to {backbone_save_path}\n")

if __name__ == '__main__':
    main()
