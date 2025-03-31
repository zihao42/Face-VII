# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import os
# import random
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
# Label Mapping Functions
#############################
def load_label_mapping(label_file):
    """
    Load label mapping from an XLSX file.
    Assumes the file has columns 'filename' and 'label'. 
    Returns a dictionary mapping filename to label.
    """
    df = pd.read_excel(label_file)
    mapping = {}
    # Adjust column names as needed
    for _, row in df.iterrows():
        mapping[row['filename']] = row['label']
    return mapping

#############################
# Parsing Functions
#############################
def parse_ravdess_info(filename):
    """
    Parse a RAVDESS filename and return a dictionary with the relevant fields:
      - modality: first field (e.g., "02")
      - vocal_channel: second field (e.g., "01")
      - emotion: third field, converted to a zero-based integer (e.g., "06" becomes 5)
      
    Example:
        For filename "02-01-06-01-02-01-12.mp4":
            modality: "02"
            vocal_channel: "01"
            emotion: int("06") - 1 = 5  (fearful)
    """
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    parts = name.split('-')
    if len(parts) < 3:
        raise ValueError("Filename does not have the required 7 parts")
    info = {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion': int(parts[2]) - 1  # zero-based
    }
    return info

#############################
# Video Processing Functions
#############################
def load_video_frames(video_path, num_frames=16):
    """
    Load video frames from video_path using OpenCV.
    Uniformly sample num_frames frames from the entire video.
    Returns a list of frames (as numpy arrays in RGB format).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame_rgb)
        frame_id += 1
    cap.release()

    # Duplicate last frame if fewer frames were captured
    while len(sampled_frames) < num_frames:
        sampled_frames.append(sampled_frames[-1])
    return sampled_frames

#############################
# Audio Processing Functions
#############################
def load_audio_file(audio_path, target_sample_rate=16000):
    """
    Load an audio file using torchaudio and resample it to the target_sample_rate if needed.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                   new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

#############################
# Dataset Class
#############################
class RAVDESSDataset(Dataset):
    def __init__(self, video_dir, audio_dir, file_list, label_mapping,
                 modality='both', num_frames=16, video_transform=None,
                 audio_transform=None, target_sample_rate=16000):
        """
        Args:
            video_dir (str): Directory containing video (.mp4) files.
            audio_dir (str or None): Directory containing audio (.wav) files.
                If None, audio is loaded from the video file.
            file_list (list): List of filenames (e.g., "02-01-06-01-02-01-12.mp4").
            label_mapping (dict): Dictionary mapping filename to label.
            modality (str): 'video', 'audio', or 'both'.
            num_frames (int): Number of video frames to sample.
            video_transform: Transform to be applied on each video frame.
            audio_transform: Transform to be applied on the audio waveform.
            target_sample_rate (int): Target sample rate for audio.
        """
        self.video_dir = video_dir
        self.audio_dir = audio_dir
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
        
        # Retrieve label from mapping (ensuring efficient lookup)
        if filename not in self.label_mapping:
            raise KeyError(f"Label for {filename} not found in the label mapping.")
        label = self.label_mapping[filename]

        # Optionally, also parse the RAVDESS info if needed
        info = parse_ravdess_info(filename)
        data = {}

        # Load video data if required
        if self.modality in ['video', 'both']:
            video_path = os.path.join(self.video_dir, filename)
            frames = load_video_frames(video_path, self.num_frames)
            if self.video_transform:
                frames = [self.video_transform(Image.fromarray(frame)) for frame in frames]
            else:
                frames = [transforms.ToTensor()(Image.fromarray(frame)) for frame in frames]
            video_tensor = torch.stack(frames)  # Shape: (num_frames, C, H, W)
            data['video'] = video_tensor

        # Load audio data if required
        if self.modality in ['audio', 'both']:
            if self.audio_dir is not None:
                base_name, _ = os.path.splitext(filename)
                audio_filename = base_name + '.wav'
                audio_path = os.path.join(self.audio_dir, audio_filename)
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file {audio_path} not found.")
            else:
                audio_path = os.path.join(self.video_dir, filename)
            waveform = load_audio_file(audio_path, self.target_sample_rate)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
            data['audio'] = waveform

        return data, label, info

#############################
# DataLoader Helper Functions
#############################
def get_file_list(video_dir, extension='.mp4'):
    """
    Retrieve a list of files with the given extension from video_dir,
    filtering to include only speech files (i.e. vocal_channel == "01").
    """
    file_list = []
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(extension):
            parts = f.split('-')
            if len(parts) < 2:
                continue
            # Only include files where the vocal channel (second field) is "01" (speech)
            if parts[1] == '01':
                file_list.append(f)
    random.shuffle(file_list)
    return file_list

def split_file_list(file_list, test_ratio=0.1, eval_ratio=0.2):
    """
    Split the file list into training, evaluation, and test sets.
    """
    num_samples = len(file_list)
    test_size = int(num_samples * test_ratio)
    eval_size = int(num_samples * eval_ratio)
    
    test_files = file_list[:test_size]
    eval_files = file_list[test_size:test_size + eval_size]
    train_files = file_list[test_size + eval_size:]
    return train_files, eval_files, test_files

def get_dataloaders(video_dir, audio_dir, label_file, modality='both', batch_size=16,
                    num_frames=16, test_ratio=0.1, eval_ratio=0.2,
                    video_transform=None, audio_transform=None,
                    target_sample_rate=16000, num_workers=4):
    """
    Build and return the train, evaluation, and test dataloaders for the RAVDESS dataset.
    Only speech files (vocal_channel "01") are included.
    
    The label_file (XLSX) is loaded once to create a mapping, ensuring efficient label lookups.
    """
    # Load label mapping from XLSX file
    label_mapping = load_label_mapping(label_file)
    
    # Define a default video transform if none is provided.
    if video_transform is None:
        video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    file_list = get_file_list(video_dir, extension='.mp4')
    train_files, eval_files, test_files = split_file_list(file_list, test_ratio, eval_ratio)
    
    train_dataset = RAVDESSDataset(video_dir, audio_dir, train_files, label_mapping,
                                   modality, num_frames, video_transform,
                                   audio_transform, target_sample_rate)
    eval_dataset = RAVDESSDataset(video_dir, audio_dir, eval_files, label_mapping,
                                  modality, num_frames, video_transform,
                                  audio_transform, target_sample_rate)
    test_dataset = RAVDESSDataset(video_dir, audio_dir, test_files, label_mapping,
                                  modality, num_frames, video_transform,
                                  audio_transform, target_sample_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, eval_loader, test_loader

#############################
# Example Usage
#############################
# video_directory = "/path/to/ravdess/videos"
# audio_directory = "/path/to/ravdess/audios"  # Set to None to load audio from the video file
# label_file = "/path/to/labels.xlsx"
# train_loader, eval_loader, test_loader = get_dataloaders(video_directory, audio_directory, label_file, modality='both')





# # Define RAF-DB dataset class
# class RAFDBDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]

#         image = Image.open(img_path).convert("L").convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# # Load dataset paths and labels (Modify with actual dataset location)
# def load_rafdb_dataset(img_dir, label_file_path, uk_mode, uk):
#     if uk == "sur":
#         uk_conv = "1"
#     elif uk == "fea":
#         uk_conv = "2"
#     elif uk == "dis":
#         uk_conv = "3"
#     elif uk == "hap":
#         uk_conv = "4"
#     elif uk == "sad":
#         uk_conv = "5"
#     elif uk == "ang":
#         uk_conv = "6"
#     elif uk == "neu":
#         uk_conv = "7"
#     else:
#         uk_conv = None

#     train_image_paths = []
#     train_labels = []
#     test_image_paths = []
#     test_labels = []
#     with open(label_file_path, "r") as f:
#         for line in f:
#             img_name, label = line.strip().split()
#             name, suffix = img_name.split('.')
#             img_name = name + '_aligned.' + suffix
#             if "train" in img_name:
#                 if label == uk_conv:
#                     continue
#                 else:
#                     train_image_paths.append(os.path.join(img_dir, img_name))
#                     train_labels.append(int(label) - 1)  # Convert to zero-based index
#                     #train_labels.append(int(label))
#             elif "test" in img_name:
#                 test_image_paths.append(os.path.join(img_dir, img_name))
#                 if label == uk_conv:
#                     #test_labels.append(int(label) - 1)  # Convert to zero-based index
#                     test_labels.append(8)
#                 else:
#                     test_labels.append(int(label)-1)

            
#     return train_image_paths, train_labels, test_image_paths, test_labels


# <<<<<<< Updated upstream
# def get_dataloaders(img_dir, label_file_path, uk_mode, uk):
#     train_image_paths, train_labels, test_image_paths, test_labels = load_rafdb_dataset(img_dir, label_file_path, uk_mode, uk)
#     combined = list(zip(train_image_paths, train_labels))
#     random.shuffle(combined)
#     train_image_paths, train_labels = zip(*combined)
#     eval_size = int(len(train_labels) * 0.2)
#     eval_image_paths, eval_labels = train_image_paths[:eval_size], train_labels[:eval_size]
#     train_image_paths, train_labels = train_image_paths[eval_size:], train_labels[eval_size:]
    

#     print("Number of train samples: " + str(len(train_image_paths)))
#     print("Number of train labels: " + str(len(train_labels)))
#     print("Number of eval samples: " + str(len(eval_image_paths)))
#     print("Number of eval labels: " + str(len(eval_labels)))
#     print("Number of test samples: " + str(len(test_image_paths)))
#     print("Number of test labels: " + str(len(test_labels)))
# =======
# def get_dataloaders(img_dir, label_file_path, uk_mode, uk):
#     train_image_paths, train_labels, test_image_paths, test_labels = load_rafdb_dataset(img_dir, label_file_path, uk_mode, uk)
#     combined = list(zip(train_image_paths, train_labels))
#     random.shuffle(combined)
#     train_image_paths, train_labels = zip(*combined)
#     eval_size = int(len(train_labels) * 0.2)
#     eval_image_paths, eval_labels = train_image_paths[:eval_size], train_labels[:eval_size]
#     train_image_paths, train_labels = train_image_paths[eval_size:], train_labels[eval_size:]
#     #test_image_paths, test_labels = test_image_paths[:10], test_labels[:10]
    

#     # print("Number of train samples: " + str(len(train_image_paths)))
#     # print("Number of train labels: " + str(len(train_labels)))
#     # print("Number of eval samples: " + str(len(eval_image_paths)))
#     # print("Number of eval labels: " + str(len(eval_labels)))
#     # print("Number of test samples: " + str(len(test_image_paths)))
#     # print("Number of test labels: " + str(len(test_labels)))
# >>>>>>> Stashed changes

#     # Define transformations
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     transform_eval = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     train_dataset = RAFDBDataset(train_image_paths, train_labels, transform=transform)
#     eval_dataset = RAFDBDataset(eval_image_paths, eval_labels, transform=transform_eval)
#     test_dataset = RAFDBDataset(test_image_paths, test_labels, transform=transform_eval)
#     dataloader_train = DataLoader(train_dataset, batch_size=256, shuffle=True)
#     dataloader_eval = DataLoader(eval_dataset, batch_size=256, shuffle=False)
#     dataloader_test = DataLoader(test_dataset, batch_size=256, shuffle=False)

#     return dataloader_train, dataloader_eval, dataloader_test
