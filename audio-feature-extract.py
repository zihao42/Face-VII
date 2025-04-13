#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
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

def extract_audio_features(audio_batch: torch.Tensor, backbone: Wav2Vec2Model) -> torch.Tensor:
    """
    利用音频 backbone 提取特征，并将输出的时间步长按照 target_frames 划分、池化，实现与视频特征的对齐。
    
    输入:
        audio_batch: 音频数据张量，形状为 [batch_size, num_samples]，
                     例如，batch size = 4，num_samples 根据采样率和时长确定。
        backbone: 已加载权重的 Wav2Vec2 backbone 模型。
        target_frames: 目标帧数 T，应与视频特征提取中视频的帧数一致（例如 32），
                       → **重要：必须与视频端的参数对齐，否则后续多模态融合会出错。**
                       
    处理流程:
        1. 将 audio_batch 输入 backbone，获得输出 last_hidden_state，形状为 [B, L, hidden_dim]，
           其中 B 为批次大小，L 为 backbone 经过下采样后的时间步数。
        2. 检查 L 是否能被 target_frames 整除，如不满足请检查音频预处理或采样率设定。
        3. 计算每个目标帧对应的 token 数：chunk_size = L // target_frames。
        4. 将 last_hidden_state 重塑为形状 [B, target_frames, chunk_size, hidden_dim]，再在 chunk_size 维度上取平均，
           得到形状为 [B, target_frames, hidden_dim] 的对齐特征。
    
    返回:
        aligned_features: 对齐后的音频特征张量，形状为 [batch_size, target_frames, hidden_dim]。
    """
    outputs = backbone(audio_batch)
    last_hidden = outputs.last_hidden_state  # shape: [B, L, hidden_dim]
    B, L, hidden_dim = last_hidden.shape
    target_frames = audio_batch.shape[1]

    # 确保 L 能被 target_frames 整除
    if L % target_frames != 0:
        raise ValueError(
            f"音频输出的 token 数 L={L} 不能被目标帧数 target_frames={target_frames} 整除。"
            "请检查音频预处理或调整采样率。"
        )

    # 计算每个目标帧对应的 token 数（chunk_size）
    chunk_size = L // target_frames
    # 重塑为 [B, target_frames, chunk_size, hidden_dim] 后，沿 chunk_size 维度取平均获得对齐特征
    aligned_features = last_hidden.reshape(B, target_frames, chunk_size, hidden_dim).mean(dim=2)
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
