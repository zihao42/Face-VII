#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from transformers import Wav2Vec2Model

def load_audio_backbone(weights: str, device: torch.device) -> Wav2Vec2Model:
    """
    加载预训练的音频 backbone 模型，并使用外部权重进行初始化。
    
    输入:
        weights: 预训练权重文件路径（.pth 文件），该文件仅包含 backbone 的 state_dict。
        device: torch.device 对象，用于指定模型加载设备（如 torch.device("cuda")）。
        
    返回:
        backbone 模型 (Wav2Vec2Model)，设置为 eval 模式，可用于特征提取。
        
    注意:
        - 此函数的结构与视频端的 load_timesformer_backbone 非常类似，
          先使用 Hugging Face 的 from_pretrained 加载预训练模型，再加载外部的 state_dict。
    """
    # 这里选用 facebook/wav2vec2-base-960h 作为示例模型
    backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)
    state_dict = torch.load(weights, map_location=device)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    return backbone

def extract_audio_features(audio_batch: torch.Tensor, backbone: Wav2Vec2Model, target_frames: int) -> torch.Tensor:
    """
    利用音频 backbone 提取特征，并自适应池化到 target_frames 个时间步，
    返回对齐后的特征，形状为 [B, target_frames, hidden_dim]。

    参数:
        audio_batch: 输入音频，形状为 [B, num_samples]（或 [B, 1, num_samples]）
        backbone: 加载了预训练权重的 Wav2Vec2 模型
        target_frames: 目标时间步数（应与视频特征一致，如 32 或 16）

    返回:
        aligned_features: 对齐后的音频特征张量，形状为 [B, target_frames, hidden_dim]
    """
    outputs = backbone(audio_batch)
    last_hidden = outputs.last_hidden_state  # shape: [B, L, hidden_dim]
    # 将 last_hidden 转置为 (B, hidden_dim, L)，方便自适应池化
    hidden_states_t = last_hidden.transpose(1, 2)  # shape: [B, hidden_dim, L]
    aligned_features = nn.functional.adaptive_avg_pool1d(hidden_states_t, target_frames)
    aligned_features = aligned_features.transpose(1, 2)  # shape: [B, target_frames, hidden_dim]
    return aligned_features

# -------------------------------------------------------------------
# 示例调用（请在实际代码中使用时传入正确参数）：
#
# 假设：
#   - batch size = 4
#   - target_frames = 32 （必须与视频端的 T 参数保持一致）
#   - 每段音频的采样点数为 num_samples，根据采样率和时长确定（例如 16000 表示 1 秒 16kHz 音频）
#
# 示例：
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weights_path = "path_to_your_audio_backbone_weights.pth"  # 替换为实际权重路径
# backbone = load_audio_backbone(weights_path, device)
#
# 构造一个随机的 audio_batch，假设每段音频有 16000 个采样点：
# audio_batch = torch.randn(4, 16000).to(device)
#
# 调用提取函数：
# features = extract_audio_features(audio_batch, backbone, target_frames=32)
# print("对齐后的音频特征形状:", features.shape)
