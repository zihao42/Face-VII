import torch
from torch.utils.data import Dataset, DataLoader
import random


class DummyTimeSformerDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=5, C=3, T=32, H=224, W=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.C = C
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TimeSformer expects (C, T, H, W)
        video = torch.randn(self.T, self.C, self.H, self.W)
        label = random.randint(0, self.num_classes - 1)
        return video, label


class DummyWav2Vec2Dataset(Dataset):
    def __init__(self, num_samples=100, num_classes=5, audio_len=50000):
        """
        :param num_samples: number of audio clips
        :param num_classes: number of classification labels
        :param audio_len: number of audio samples per clip (e.g., 16000 = 1 sec at 16kHz)
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.audio_len = audio_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy raw audio waveform (1D float tensor)
        waveform = torch.randn(self.audio_len)
        label = random.randint(0, self.num_classes - 1)
        return waveform, label
