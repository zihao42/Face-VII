import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Import the pretrained wav2vec2 model from transformers
from transformers import Wav2Vec2Model

# Reuse the audio loading function from your data.py
from data import load_audio_file

def collate_fn_audio(batch):
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
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        
        df = pd.read_csv(csv_file)
        df = df[df['emo_label'] != 8].reset_index(drop=True)
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
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# ===================== Define Wav2Vec Classifier with Regularization and Feature Alignment =====================
class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes, target_time=32, dropout_rate=0.3):
        super(Wav2VecClassifier, self).__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        self.target_time = target_time
        
    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        outputs = self.backbone(x)
        hidden_states = outputs.last_hidden_state  # (batch, T, hidden_size)
        hidden_states_t = hidden_states.transpose(1, 2)  # (batch, hidden_size, T)
        aligned_features = nn.functional.adaptive_avg_pool1d(hidden_states_t, self.target_time)
        aligned_features = aligned_features.transpose(1, 2)  # (batch, target_time, hidden_size)
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
    csv_dir = "/media/data1/ningtong/wzh/datasets/RAVDESS/csv/"
    audio_dir = "/media/data1/ningtong/wzh/datasets/RAVDESS"
    backbone_save_dir = "/media/data1/ningtong/wzh/projects/Face-VII/weights/backbones/audio"
    os.makedirs(backbone_save_dir, exist_ok=True)

    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-5
    val_split = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        model = Wav2VecClassifier(num_classes=dataset.num_classes, target_time=32, dropout_rate=0.3)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        print(f"  Training on {csv_file_name} ...")
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
        print(f"  Saved training curve to {plot_save_path}")

        model.classifier = nn.Identity()
        backbone_save_path = os.path.join(backbone_save_dir, f"{os.path.splitext(csv_file_name)[0]}_wav2vec_backbone.pth")
        torch.save(model.backbone.state_dict(), backbone_save_path)
        print(f"  Saved backbone weights to {backbone_save_path}\n")

if __name__ == '__main__':
    main()
