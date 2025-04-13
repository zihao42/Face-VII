#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm

# 从 transformers 导入 TimesformerModel
from transformers import TimesformerModel

# ===================== 视频帧采样和预处理函数 =====================
def load_video_frames(video_path, num_frames=16, transform=None):
    """
    读取视频文件并均匀采样 num_frames 帧
    参数：
        video_path: 视频文件路径
        num_frames: 采样帧数（默认16帧）
        transform: 对 PIL 图像进行的预处理变换
    返回：
        Tensor，形状为 (num_frames, C, H, W)
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
            # 将 BGR 转为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if transform:
                img = transform(img)
            else:
                img = transforms.ToTensor()(img)
            frames.append(img)
        frame_id += 1
    cap.release()
    # 如果采样帧不足，则复制最后一帧
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.stack(frames)  # 形状为 (num_frames, C, H, W)

# ===================== 自定义视频数据集 =====================
class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, num_frames=16, transform=None):
        """
        初始化视频数据集
        参数：
            csv_file: CSV 文件路径，包含 "filename", "category", "emo_label" 三列
            video_dir: 视频文件所在目录
            num_frames: 采样帧数（默认16帧）
            transform: 视频帧预处理的变换
        """
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform

        # 读取 CSV 文件
        df = pd.read_csv(csv_file)
        # 过滤掉 emo_label 为 8 的样本（无需训练预测）
        df = df[df['emo_label'] != 8].reset_index(drop=True)
        # 只保留 .mp4 文件（排除其他格式，如 .wav）
        df = df[df['filename'].str.lower().str.endswith('.mp4')].reset_index(drop=True)
        
        # 将剩余的 emo_label 标签构建连续索引映射（例如原标签 {0, 1, 3, 5, 6} 映射为 {0, 1, 2, 3, 4}）
        unique_labels = sorted(df['emo_label'].unique())
        self.label_mapping = {orig: idx for idx, orig in enumerate(unique_labels)}
        
        # 构造样本列表：每个样本为 (完整视频路径, 重新映射后的标签)
        self.samples = []
        for _, row in df.iterrows():
            filename = row['filename']
            label_orig = row['emo_label']
            label = self.label_mapping[label_orig]
            video_path = os.path.join(video_dir, filename)
            self.samples.append((video_path, label))
        
        self.num_classes = len(unique_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        # 读取视频帧，形状为 (num_frames, C, H, W)
        video = load_video_frames(video_path, num_frames=self.num_frames, transform=self.transform)
        # Timesformer 模型要求输入格式为 (T, C, H, W)
        return video, label

# ===================== 定义基于 Timesformer 的视频分类网络 =====================
class TimesformerClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        初始化 Timesformer 分类模型
        参数：
            num_classes: 分类数（根据 CSV 文件重新映射后的类别数）
        """
        super(TimesformerClassifier, self).__init__()
        # 调用 Transformers API 加载预训练的 Timesformer 模型，
        # 使用 fine-tuned 模型 "facebook/timesformer-base-finetuned-k400"
        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400", output_hidden_states=True
        )
        hidden_size = self.backbone.config.hidden_size  # 通常为768
        # 分类头，用于 finetune 分类任务
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 输入 x 形状应为 (batch_size, T, C, H, W)
        outputs = self.backbone(x)
        # 获取 last_hidden_state，形状为 (batch_size, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state
        # 对序列维度做平均池化，得到 (batch_size, hidden_size)
        pooled = last_hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# ===================== 训练与评估函数 =====================
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    模型训练与评估
    每个 epoch 计算训练和验证指标，并记录下来用于绘图展示
    返回：记录训练/验证损失及准确率的字典
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # 使用 tqdm 显示训练进度
        epoch_iter = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")
        for videos, labels in epoch_iter:
            videos = videos.to(device)  # videos 形状: (batch, T, C, H, W)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

            # 更新进度条描述信息
            epoch_iter.set_postfix(loss=loss.item())
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * videos.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        # 打印每个 epoch 的训练和验证结果，确保输出刷新
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}", flush=True)
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Val   Acc: {epoch_val_acc:.4f}", flush=True)

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    return model, metrics

# ===================== 主函数 =====================
def main():
    # ---------------------- 路径设置 ----------------------
    # CSV 文件所在目录（多个 CSV 文件）
    csv_dir = "/media/data1/ningtong/wzh/datasets/RAVDESS/csv/"
    # 视频文件所在目录
    video_dir = "/media/data1/ningtong/wzh/datasets/RAVDESS"
    # 保存 backbone 权重及训练曲线图像的目录
    backbone_save_dir = "/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/visual"
    os.makedirs(backbone_save_dir, exist_ok=True)

    # ---------------------- 训练参数设置 ----------------------
    batch_size = 4
    num_frames = 32
    num_epochs = 15
    learning_rate = 1e-6
    val_split = 0.2  # 验证集比例

    # ---------------------- 视频预处理设置 ----------------------
    video_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------------------- 设备设置 ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------- 遍历 CSV 文件进行训练 ----------------------
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for csv_file_name in csv_files:
        full_csv_path = os.path.join(csv_dir, csv_file_name)
        print(f"Processing CSV file: {csv_file_name}")

        # 构造数据集
        dataset = VideoDataset(csv_file=full_csv_path, video_dir=video_dir, num_frames=num_frames, transform=video_transform)
        print(f"  Total samples: {len(dataset)}, Number of classes: {dataset.num_classes}")

        # 划分训练集和验证集
        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # 构造模型（每个 CSV 的类别数可能不同）
        model = TimesformerClassifier(num_classes=dataset.num_classes)
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 训练并评估模型
        print(f"  Training on {csv_file_name} ...", flush=True)
        model, metrics = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

        # 绘制训练过程曲线，保存图片到 backbone 保存目录
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
        print(f"  Saved training curve to {plot_save_path}", flush=True)

        # 训练结束后，替换分类头为 Identity，仅保存视觉 backbone 权重
        model.classifier = nn.Identity()
        backbone_save_path = os.path.join(backbone_save_dir, f"{os.path.splitext(csv_file_name)[0]}_timesformer_backbone.pth")
        torch.save(model.backbone.state_dict(), backbone_save_path)
        print(f"  Saved backbone weights to {backbone_save_path}\n", flush=True)

if __name__ == '__main__':
    main()
