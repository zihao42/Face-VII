import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import TimesformerModel


def load_video_frames(video_path, num_frames=32, transform=None):

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if transform:
                img = transform(img)
            else:
                img = transforms.ToTensor()(img)
            frames.append(img)
        frame_id += 1
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.stack(frames)

class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, num_frames=32, transform=None):

        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform

        df = pd.read_csv(csv_file)

        df = df[df['emo_label'] != 8].reset_index(drop=True)

        df = df[df['filename'].str.lower().str.endswith('.mp4')].reset_index(drop=True)

        unique_labels = sorted(df['emo_label'].unique())
        self.label_mapping = {orig: idx for idx, orig in enumerate(unique_labels)}

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
        video = load_video_frames(video_path, num_frames=self.num_frames, transform=self.transform)
        return video, label


class TimesformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TimesformerClassifier, self).__init__()

        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400", output_hidden_states=True
        )
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        outputs = self.backbone(x)
        last_hidden = outputs.last_hidden_state
        pooled = last_hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


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
        for videos, labels in epoch_iter:
            videos = videos.to(device)
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

            epoch_iter.set_postfix(loss=loss.item())
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

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

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(BASE_DIR, "../../datasets/RAVDESS/csv")
    video_dir = os.path.join(BASE_DIR, "../../datasets/RAVDESS")
    backbone_save_dir = os.path.join(BASE_DIR, "weights/backbones/visual")
    os.makedirs(backbone_save_dir, exist_ok=True)

    batch_size = 4
    num_frames = 32
    num_epochs = 15
    learning_rate = 1e-6
    val_split = 0.2

    video_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for csv_file_name in csv_files:
        full_csv_path = os.path.join(csv_dir, csv_file_name)
        print(f"Processing CSV file: {csv_file_name}")

        dataset = VideoDataset(csv_file=full_csv_path, video_dir=video_dir, num_frames=num_frames, transform=video_transform)
        print(f"  Total samples: {len(dataset)}, Number of classes: {dataset.num_classes}")

        num_val = int(len(dataset) * val_split)
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = TimesformerClassifier(num_classes=dataset.num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(f"  Training on {csv_file_name} ...", flush=True)
        model, metrics = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

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

        model.classifier = nn.Identity()
        backbone_save_path = os.path.join(backbone_save_dir, f"{os.path.splitext(csv_file_name)[0]}_timesformer_backbone.pth")
        torch.save(model.backbone.state_dict(), backbone_save_path)
        print(f"  Saved backbone weights to {backbone_save_path}\n", flush=True)

if __name__ == '__main__':
    main()
