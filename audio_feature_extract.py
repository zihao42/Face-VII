import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

def load_audio_backbone(weights: str, device: torch.device) -> Wav2Vec2Model:
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    backbone = Wav2Vec2Model(config)
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    return backbone

def extract_audio_features(
    audio_batch: torch.Tensor,
    combination: int,
    weights_dir: str = os.path.join("weights", "backbones", "audio"),
    device: torch.device = None
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    weights_fname = f"openset_split_combination_{combination}_wav2vec_backbone.pth"
    weights_path = os.path.join(weights_dir, weights_fname)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find weights files: {weights_path}")

    backbone = load_audio_backbone(weights_path, device)
    audio_batch = audio_batch.to(device)
    outputs = backbone(audio_batch)
    hidden_states = outputs.last_hidden_state  # [B, seq_len, D]
    B, _, D = hidden_states.shape

    x = hidden_states.transpose(1, 2)          # [B, D, seq_len]
    pooled = nn.functional.adaptive_avg_pool1d(x, 32)  # [B, D, 32]
    features = pooled.transpose(1, 2)          # [B, 32, D]
    return features

def extract_audio_features_from_backbone(
    audio_batch: torch.Tensor,
    backbone: Wav2Vec2Model,
    target_frames: int = 32
) -> torch.Tensor:
    device = next(backbone.parameters()).device
    audio_batch = audio_batch.to(device)
    outputs = backbone(audio_batch)
    hidden_states = outputs.last_hidden_state  # [B, seq_len, D]
    B, _, D = hidden_states.shape

    x = hidden_states.transpose(1, 2)               # [B, D, seq_len]
    pooled = nn.functional.adaptive_avg_pool1d(x, target_frames)  # [B, D, target_frames]
    features = pooled.transpose(1, 2)               # [B, target_frames, D]
    return features

def extract_audio_features_from_backbone_eva(
    audio_batch: torch.Tensor,
    backbone: Wav2Vec2Model,
    target_frames: int = 32
) -> torch.Tensor:
    device = next(backbone.parameters()).device
    B = audio_batch.size(0)
    audio_flat = audio_batch.view(B, -1).to(device)  # [B, T_total]

    outputs = backbone(audio_flat).last_hidden_state  # [B, seq_len', D]

    x = outputs.transpose(1, 2)                   # [B, D, seq_len']
    pooled = nn.functional.adaptive_avg_pool1d(x, target_frames)  # [B, D, target_frames]
    features = pooled.transpose(1, 2)             # [B, target_frames, D]
    return features
