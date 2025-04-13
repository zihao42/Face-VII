#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import TimesformerModel

def load_timesformer_backbone(weights: str, device: torch.device) -> TimesformerModel:
    """
    加载预训练的 Timesformer backbone 模型，并使用 .pth 文件中的权重进行初始化。
    
    输入:
        weights: 字符串，预训练权重的路径（.pth文件），该文件仅包含 backbone 的 state_dict
        device: torch.device 对象，用于模型加载设备（如 torch.device("cuda")）
        
    返回:
        backbone 模型 (TimesformerModel)，处于 eval 模式，可用于特征提取
    """
    # 使用 Hugging Face 提供的 fine-tuned 模型
    backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", output_hidden_states=True)
    state_dict = torch.load(weights, map_location=device)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    return backbone

def extract_video_features(video_batch: torch.Tensor, backbone: TimesformerModel) -> torch.Tensor:
    """
    利用 Timesformer backbone 提取视频特征，返回形状为 [batch_size, seq_length, hidden_dim] 的张量，
    其中 seq_length 对应于每个视频帧提取后的时序特征（未进行时间维度的平均池化）。
    
    输入:
        video_batch: 视频数据张量，形状为 [batch_size, T, C, H, W]，
                     T 表示帧数，C/H/W 为通道/高度/宽度。
        backbone: 加载好权重的 Timesformer backbone 模型
        
    处理流程:
        1. 对输入视频数据进行前向传播，获得模型输出（last_hidden_state）。
        2. 输出 last_hidden_state 的形状为 [B, L, hidden_dim]，其中第一个 token 通常为 CLS token，
           剩余 token 数为 L-1，应满足 L-1 = T * num_patches（num_patches 为每帧中 spatial patch 数量）。
        3. 去除 CLS token 后，将剩余 token reshape 为 [B, T, num_patches, hidden_dim]。
        4. 对每一帧的 spatial tokens（维度 num_patches）进行平均池化，得到形状 [B, T, hidden_dim]。
    
    返回:
        frame_features: 视频特征张量，形状为 [batch_size, T, hidden_dim]，其中 T 对应视频的时序特征，
                        可用于与音频特征进行时序对齐和特征融合。
    """
    
    # video_batch 的形状: [B, T, C, H, W]
    outputs = backbone(video_batch)  # 输出为一个模型输出对象，包含 last_hidden_state
    last_hidden = outputs.last_hidden_state  # shape: [B, L, hidden_dim]
    B, L, hidden_dim = last_hidden.shape
    # 假定第一个 token 为 CLS token
    L_without_cls = L - 1
    T = video_batch.shape[1]  # 帧数 T
    if L_without_cls % T != 0:
        raise ValueError("Token 数量与视频帧数不匹配，无法正确进行空间平均池化。")
    num_patches = L_without_cls // T
    # 去除 CLS token，并 reshape 为 [B, T, num_patches, hidden_dim]
    tokens = last_hidden[:, 1:, :].reshape(B, T, num_patches, hidden_dim)
    # 对每一帧的 spatial tokens（沿 num_patches 维度）进行平均池化，得到 [B, T, hidden_dim]
    frame_features = tokens.mean(dim=2)
    return frame_features
