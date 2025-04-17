#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

def load_audio_backbone(weights: str, device: torch.device) -> Wav2Vec2Model:
    """
    加载预训练的音频 backbone（Wav2Vec2）模型，并使用外部 .pth 权重初始化。

    输入:
        weights: str，.pth 权重文件，仅包含 backbone 的 state_dict
        device: torch.device，用于模型加载

    返回:
        已加载并设为 eval 模式的 Wav2Vec2Model
    """
    # 从官方仓库获取配置，不下载模型权重
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    # 根据配置初始化一个干净的模型，实现不包含 masked_spec_embed
    backbone = Wav2Vec2Model(config)

    # 加载并替换自己的权重
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    backbone.load_state_dict(state_dict)

    # 部署到设备并设置为评估模式
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
    使用 Wav2Vec2 backbone 提取音频特征，并自适应池化到固定的 32 帧。

    输入:
        audio_batch: Tensor，[B, num_samples]
        combination: int，对应权重文件编号 N，会加载 openset_split_combination_{N}_wav2vec_backbone.pth
        weights_dir: str，可选，权重文件目录
        device: str 或 torch.device，可选；None 则自动选择 GPU 否则 CPU

    返回:
        features: Tensor，[B, 32, hidden_dim]
    """
    # 设备选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    # 构造权重路径
    weights_fname = f"openset_split_combination_{combination}_wav2vec_backbone.pth"
    weights_path = os.path.join(weights_dir, weights_fname)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到权重文件: {weights_path}")

    # 加载 backbone
    backbone = load_audio_backbone(weights_path, device)

    # 准备输入
    audio_batch = audio_batch.to(device)

    # 前向计算
    outputs = backbone(audio_batch)
    hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
    B, seq_len, hidden_dim = hidden_states.shape

    # 固定池化到 32 帧
    target_frames = 32
    x = hidden_states.transpose(1, 2)            # [B, hidden_dim, seq_len]
    pool = nn.AdaptiveAvgPool1d(target_frames)
    pooled = pool(x)                            # [B, hidden_dim, target_frames]
    features = pooled.transpose(1, 2)           # [B, target_frames, hidden_dim]

    return features
