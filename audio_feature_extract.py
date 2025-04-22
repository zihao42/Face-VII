#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

def load_audio_backbone(weights: str, device: torch.device) -> Wav2Vec2Model:
    """
    加载预训练的音频 backbone（Wav2Vec2）模型，并使用外部 .pth 权重初始化。
    """
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    backbone = Wav2Vec2Model(config)
    # 安全地只加载权重
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
    """
    按原逻辑：每次加载 backbone 并提取特征（兼容旧调用）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    weights_fname = f"openset_split_combination_{combination}_wav2vec_backbone.pth"
    weights_path = os.path.join(weights_dir, weights_fname)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")

    backbone = load_audio_backbone(weights_path, device)
    audio_batch = audio_batch.to(device)
    outputs = backbone(audio_batch)
    hidden_states = outputs.last_hidden_state  # [B, seq_len, D]
    B, _, D = hidden_states.shape

    # 池化到 32 帧（与视频对齐）
    x = hidden_states.transpose(1, 2)          # [B, D, seq_len]
    pooled = nn.functional.adaptive_avg_pool1d(x, 32)  # [B, D, 32]
    features = pooled.transpose(1, 2)          # [B, 32, D]
    return features

def extract_audio_features_from_backbone(
    audio_batch: torch.Tensor,
    backbone: Wav2Vec2Model,
    target_frames: int = 32
) -> torch.Tensor:
    """
    使用已加载 backbone（只加载一次）提取音频特征并池化到 target_frames。
    """
    device = next(backbone.parameters()).device
    audio_batch = audio_batch.to(device)
    outputs = backbone(audio_batch)
    hidden_states = outputs.last_hidden_state  # [B, seq_len, D]
    B, _, D = hidden_states.shape

    x = hidden_states.transpose(1, 2)               # [B, D, seq_len]
    pooled = nn.functional.adaptive_avg_pool1d(x, target_frames)  # [B, D, target_frames]
    features = pooled.transpose(1, 2)               # [B, target_frames, D]
    return features
